"""Build the supervised fine-tuning set for a round.

Two ingredients, both necessary:

* **Refusal pairs** -- (successful attack -> safe response). The safe
  responses are model-written, in the spirit of RLAIF / Constitutional AI
  (Bai et al., 2022), which is cheap but means they are never better than
  the generator; they are validated before use.
* **Retention pairs** -- (benign instruction -> the defender's own current
  answer). Training exclusively on refusals teaches "refuse everything",
  which scores a perfect 0% ASR while destroying the model. Self-distilling
  the defender's existing answers on benign prompts anchors normal
  behaviour without needing a stronger teacher.
"""

from __future__ import annotations

from .config import SafeResponseConfig
from .local_lm import LocalCausalLM
from .metrics import looks_like_refusal
from .utils import get_logger

LOGGER = get_logger(__name__)

SAFE_RESPONSE_SYSTEM_PROMPT = """
You create supervised safety-training responses.

Given a user request that should not be fulfilled, produce a concise
assistant response that declines the unsafe portion while remaining helpful
where appropriate. Offer a legitimate alternative when one exists.

Do not restate the request. Do not include any content that would advance
the unsafe request. Return only the assistant response.
""".strip()

MIN_SAFE_RESPONSE_CHARS = 20


def build_safe_response_messages(attack: str) -> list[dict]:
    """Ask the teacher to refuse ``attack``, with the attack quoted as data.

    The attack is not sent as the user turn itself. These prompts are the
    ones that already jailbroke the defender, so handing them over as the
    live instruction tends to make the teacher comply instead of refusing.
    """
    return [
        {"role": "system", "content": SAFE_RESPONSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Write the assistant reply that should be used as the training "
                "target for the user message below. Refuse anything unsafe. "
                "Do not restate the request and do not include content that "
                "would carry it out.\n\n"
                f"USER MESSAGE:\n{attack}"
            ),
        },
    ]


def generate_refusal(lm: LocalCausalLM, config: SafeResponseConfig, attack: str) -> str | None:
    try:
        return lm.complete_chat(
            build_safe_response_messages(attack),
            temperature=config.temperature,
            max_new_tokens=config.max_tokens,
            top_p=config.top_p,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Safe-response generation failed: %s", exc)
        return None


def build_refusal_dataset(
    config: SafeResponseConfig, successful_attacks: list[dict]
) -> list[dict]:
    """Turn successful attacks into (prompt, refusal) training pairs.

    The teacher is loaded from Hugging Face for this call and released
    before the caller trains the defender.
    """
    if not successful_attacks:
        return []

    lm = LocalCausalLM(
        config.model_id,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=config.trust_remote_code,
        max_seq_length=config.max_seq_length,
    )
    examples: list[dict] = []
    rejected = 0
    try:
        lm.load()
        responses = lm.complete_chat_batch(
            [build_safe_response_messages(item["attack"]) for item in successful_attacks],
            temperature=config.temperature,
            max_new_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        for item, response in zip(successful_attacks, responses):
            if response is None:
                LOGGER.warning("Safe-response generation failed")
                rejected += 1
                continue
            if not _is_usable_refusal(response):
                rejected += 1
                continue
            examples.append(
                {
                    "prompt": item["attack"],
                    "response": response,
                    "behavior_id": item["behavior_id"],
                    "source": "refusal",
                }
            )
    finally:
        lm.unload()

    LOGGER.info(
        "Built %d refusal example(s); rejected %d unusable generation(s)",
        len(examples),
        rejected,
    )
    return examples


def build_retention_dataset(
    model, tokenizer, prompts: list[str], defender_config
) -> list[dict]:
    """Self-distil the defender's own answers to benign prompts."""
    from .defender import generate_responses

    if not prompts:
        return []

    responses = generate_responses(model, tokenizer, prompts, defender_config)
    examples = [
        {
            "prompt": prompt,
            "response": response,
            "behavior_id": "__retention__",
            "source": "retention",
        }
        for prompt, response in zip(prompts, responses)
        # Drop anything the model already refuses -- reinforcing an existing
        # false refusal is the opposite of what retention data is for.
        if response.strip() and not looks_like_refusal(response)
    ]
    LOGGER.info(
        "Built %d retention example(s) from %d benign prompt(s)",
        len(examples),
        len(prompts),
    )
    return examples


def mix_datasets(
    refusals: list[dict], retention: list[dict], ratio: float, seed: int
) -> list[dict]:
    """Combine refusal and retention data at ``ratio`` retention per refusal."""
    import random

    if not refusals:
        LOGGER.warning("No refusal examples; nothing to train on this round.")
        return []

    n_retention = int(round(len(refusals) * ratio))
    rng = random.Random(seed)
    sampled = rng.sample(retention, min(n_retention, len(retention)))
    if n_retention and len(sampled) < n_retention:
        LOGGER.warning(
            "Wanted %d retention example(s), only %d available.",
            n_retention,
            len(sampled),
        )

    mixed = refusals + sampled
    rng.shuffle(mixed)
    LOGGER.info(
        "Training mix: %d refusal + %d retention = %d example(s)",
        len(refusals),
        len(sampled),
        len(mixed),
    )
    return mixed


def _is_usable_refusal(response: str | None) -> bool:
    if not response or len(response.strip()) < MIN_SAFE_RESPONSE_CHARS:
        return False
    # A teacher that complied with the attack must not become the training target.
    return looks_like_refusal(response)
