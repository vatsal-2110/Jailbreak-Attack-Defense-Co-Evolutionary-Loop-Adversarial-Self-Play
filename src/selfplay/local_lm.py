"""Load a causal LM from Hugging Face, generate, then release the GPU.

Attacker, classifier, safe-response teacher, and defender are never resident
together. Each caller loads one model, uses it, and calls ``unload``.
"""

from __future__ import annotations

from .utils import configure_cpu_parallelism, free_gpu, get_logger

LOGGER = get_logger(__name__)

# Independent completions share one padded forward. Chunked so a long
# attacker sample cannot exhaust the GPU; each prompt is still decoded alone.
_GENERATE_CHUNK = 8


class LocalCausalLM:
    def __init__(
        self,
        model_id: str,
        *,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        max_seq_length: int = 2048,
    ) -> None:
        self.model_id = model_id
        self._load_in_4bit = load_in_4bit
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return

        configure_cpu_parallelism()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = None
        if self._load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        LOGGER.info("Loading %s from Hugging Face", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self._trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {
            "quantization_config": quant_config,
            "device_map": "auto",
            "trust_remote_code": self._trust_remote_code,
        }
        if not self._load_in_4bit:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        free_gpu()

    def complete_chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float = 0.95,
    ) -> str:
        """Chat-template generation. The template already includes special tokens."""
        self.load()
        return self._generate(
            self._render_chat(messages),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

    def complete_chat_batch(
        self,
        conversations: list[list[dict]],
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float = 0.95,
    ) -> list[str | None]:
        """Same decoding settings as ``complete_chat``, one string per conversation.

        Order matches the input. A failed row is ``None`` so one bad prompt
        does not drop the rest. A single conversation uses the unbatched path.
        """
        if not conversations:
            return []
        self.load()
        from .utils import thread_map

        prompts = thread_map(self._render_chat, conversations)
        outputs: list[str | None] = []
        for start in range(0, len(prompts), _GENERATE_CHUNK):
            chunk = prompts[start : start + _GENERATE_CHUNK]
            if len(chunk) == 1:
                try:
                    outputs.append(
                        self._generate(
                            chunk[0],
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            top_p=top_p,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Generation failed: %s", exc)
                    outputs.append(None)
                continue
            try:
                outputs.extend(
                    generate_texts(
                        self._model,
                        self._tokenizer,
                        chunk,
                        max_new_tokens=max_new_tokens,
                        max_length=self._max_seq_length,
                        temperature=temperature,
                        top_p=top_p,
                        add_special_tokens=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Batched generation failed (%s); retrying one by one.", exc)
                for prompt in chunk:
                    try:
                        outputs.append(
                            self._generate(
                                prompt,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p,
                            )
                        )
                    except Exception as item_exc:  # noqa: BLE001
                        LOGGER.warning("Generation failed: %s", item_exc)
                        outputs.append(None)
        return outputs

    def _render_chat(self, messages: list[dict]) -> str:
        tokenizer = self._tokenizer
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **template_kwargs)

    def _generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        texts = generate_texts(
            self._model,
            self._tokenizer,
            [prompt],
            max_new_tokens=max_new_tokens,
            max_length=self._max_seq_length,
            temperature=temperature,
            top_p=top_p,
            add_special_tokens=False,
        )
        return texts[0]


def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    max_length: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    add_special_tokens: bool = True,
) -> list[str]:
    """Greedy or sampled generation. Several prompts are one padded batch.

    Left padding keeps each row's new tokens aligned, matching single-prompt
    decoding. One prompt is not padded.
    """
    import torch

    if not prompts:
        return []

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left" if len(prompts) > 1 else old_padding
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
            padding=len(prompts) > 1,
        )
    finally:
        tokenizer.padding_side = old_padding

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    return [
        tokenizer.decode(row[input_length:], skip_special_tokens=True).strip()
        for row in outputs
    ]
