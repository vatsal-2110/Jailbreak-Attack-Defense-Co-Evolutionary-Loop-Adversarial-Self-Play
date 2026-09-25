"""LoRA supervised fine-tuning for one self-play round.

Two correctness fixes over the notebook:

* **Prompt masking is clamped.** ``labels[:prompt_len] = [-100] * prompt_len``
  *extends* the list when ``prompt_len > len(labels)`` (Python slice
  assignment), producing a labels tensor longer than ``input_ids``. When both
  halves hit ``max_length`` every label became ``-100`` and the loss went
  NaN. Examples with no supervised tokens are dropped.
* **Round restarts are explicit.** The notebook reused one mutated global
  model, so "D2 trained on rounds 0-1" was really D1 trained *again* on data
  it had already seen. ``restart_policy`` makes the choice explicit:
  ``from_base`` (fresh adapter, cumulative data) or ``incremental``
  (keep the adapter, new data only).
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .config import TrainConfig
from .utils import ensure_dir, get_logger, thread_map

LOGGER = get_logger(__name__)

IGNORE_INDEX = -100


def tokenize_example(example: dict, tokenizer, max_seq_length: int) -> dict | None:
    """Tokenise one (prompt, response) pair, masking the prompt tokens.

    Returns ``None`` when nothing is left to supervise, so the caller can
    drop the example instead of feeding the trainer an all-``-100`` row.
    """
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(full_text, truncation=True, max_length=max_seq_length)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_length)

    labels = list(full["input_ids"])
    # Clamp: slice assignment past the end would lengthen `labels`.
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    if all(label == IGNORE_INDEX for label in labels):
        return None

    return {
        "input_ids": list(full["input_ids"]),
        "attention_mask": list(full["attention_mask"]),
        "labels": labels,
    }


def tokenize_dataset(examples: list[dict], tokenizer, max_seq_length: int) -> list[dict]:
    rows = thread_map(
        lambda example: tokenize_example(example, tokenizer, max_seq_length),
        examples,
    )
    tokenized: list[dict] = []
    dropped = 0
    for row in rows:
        if row is None:
            dropped += 1
            continue
        tokenized.append(row)
    if dropped:
        LOGGER.warning(
            "Dropped %d example(s) with no supervised tokens after truncation "
            "(prompt alone filled max_seq_length=%d).",
            dropped,
            max_seq_length,
        )
    return tokenized


class SafetyCollator:
    """Right-pads a batch; pads labels with the ignore index."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id; set one before training.")

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            assert len(feature["labels"]) == len(feature["input_ids"]), (
                "labels/input_ids length mismatch: "
                f"{len(feature['labels'])} vs {len(feature['input_ids'])}"
            )
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad_len)
            attention_mask.append(list(feature["attention_mask"]) + [0] * pad_len)
            labels.append(list(feature["labels"]) + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_round(
    model,
    tokenizer,
    examples: list[dict],
    output_dir: str | Path,
    config: TrainConfig,
    max_seq_length: int,
):
    """Fine-tune ``model`` on ``examples`` and save the adapter."""
    if not examples:
        raise ValueError(
            "No training examples. With a working judge this means no attack "
            "succeeded; with a broken judge it means the labels are missing. "
            "Check the round's judge_detail fields before assuming the former."
        )

    output_dir = ensure_dir(output_dir)
    tokenized = tokenize_dataset(examples, tokenizer, max_seq_length)
    if not tokenized:
        raise ValueError("Every training example was dropped during tokenisation.")

    # Training needs the cache off; generation turns it back on.
    model.config.use_cache = False
    tokenizer.padding_side = "right"

    # Tokenisation already ran on a thread pool. DataLoader workers are left
    # at 0 because this process owns the CUDA context; forked workers deadlock.
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        fp16=config.fp16,
        optim=config.optim,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=SafetyCollator(tokenizer),
    )

    LOGGER.info("Training on %d example(s) -> %s", len(tokenized), output_dir)
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    LOGGER.info("Saved adapter to %s", output_dir)
    return trainer
