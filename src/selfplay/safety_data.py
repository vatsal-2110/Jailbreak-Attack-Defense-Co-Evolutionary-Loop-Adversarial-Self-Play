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

from .llm_client import LLMClient
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


def generate_refusal(client: LLMClient, model_id: str, attack: str) -> str | None:
    try:
        return client.complete(
            model=model_id,
            user_prompt=attack,
            system_prompt=SAFE_RESPONSE_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=300,
            retries=3,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Safe-response generation failed: %s", exc)
        return None


def build_refusal_dataset(
    client: LLMClient, model_id: str, successful_attacks: list[dict]
) -> list[dict]:
    """Turn successful attacks into (prompt, refusal) training pairs."""
    examples: list[dict] = []
    rejected = 0

    for item in successful_attacks:
        response = generate_refusal(client, model_id, item["attack"])
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
    return True
