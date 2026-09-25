"""Defender model: 4-bit load, LoRA attach, batched chat generation.

Fixes carried over from the notebook review:

* ``use_cache`` is toggled per phase instead of being pinned to ``False``.
  Generating with the KV cache disabled was several times slower for no gain.
* Chat prompts are truncated from the **left**. Right-truncating a chat
  template at ``max_length`` cuts off the trailing ``<|im_start|>assistant``
  header and the model no longer knows it is its turn.
* ``apply_chat_template(..., return_dict=True)`` is explicit, so the batch
  carries an ``attention_mask``. With left padding and no mask the pad tokens
  are attended to and the generations are wrong.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import DefenderConfig, LoraConfigSpec
from .utils import get_logger

LOGGER = get_logger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_defender(config: DefenderConfig):
    """Load the base defender and its tokenizer."""
    quant_config = None
    if config.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_DTYPES[config.bnb_4bit_compute_dtype],
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Chat templates put the assistant header at the end; truncating from the
    # right would remove it.
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    LOGGER.info("Loaded defender %s", config.model_id)
    return model, tokenizer


def attach_lora(model, lora: LoraConfigSpec, for_training: bool = True):
    """Prepare a quantised model for training and attach a fresh adapter."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if for_training:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_adapter(base_model, adapter_dir: str | Path, is_trainable: bool = False):
    """Attach a saved adapter. ``is_trainable`` continues LoRA training."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model, str(adapter_dir), is_trainable=is_trainable
    )
    if is_trainable:
        model.train()
    else:
        model.eval()
    LOGGER.info("Loaded adapter from %s", adapter_dir)
    return model


@contextmanager
def generation_mode(model, tokenizer):
    """Switch a model into fast, correct generation settings, then restore.

    Left padding for decoder-only batched generation; KV cache on; eval mode.
    """
    was_training = model.training
    old_padding_side = tokenizer.padding_side
    old_use_cache = getattr(model.config, "use_cache", True)

    model.eval()
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    try:
        yield
    finally:
        tokenizer.padding_side = old_padding_side
        model.config.use_cache = old_use_cache
        if was_training:
            model.train()


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    config: DefenderConfig,
    progress_every: int = 1,
) -> list[str]:
    """Batched single-turn chat generation. Greedy decoding for determinism."""
    responses: list[str] = []
    batch_size = config.generation_batch_size

    with generation_mode(model, tokenizer), torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            messages = [[{"role": "user", "content": p}] for p in batch]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Left padding makes every sequence the same length, so a single
            # offset is correct for the whole batch.
            input_length = inputs["input_ids"].shape[1]
            for row in outputs:
                text = tokenizer.decode(
                    row[input_length:], skip_special_tokens=True
                ).strip()
                responses.append(text)

            if progress_every and (start // batch_size) % progress_every == 0:
                LOGGER.info(
                    "Generated %d/%d responses",
                    min(start + batch_size, len(prompts)),
                    len(prompts),
                )
    return responses


def run_attacks(model, tokenizer, attacks: list[dict], config: DefenderConfig) -> list[dict]:
    """Run attack prompts through the defender and attach the responses."""
    if not attacks:
        return []
    prompts = [item["attack"] for item in attacks]
    responses = generate_responses(model, tokenizer, prompts, config)
    return [{**item, "response": response} for item, response in zip(attacks, responses)]
