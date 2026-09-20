"""Regression tests for prompt masking and collation.

`labels[:prompt_len] = [-100] * prompt_len` *extends* the list when
prompt_len exceeds len(labels) -- Python slice-assignment semantics. That
produced a labels tensor longer than input_ids, and when both halves hit
max_length every label was -100 and the loss went NaN.
"""

import pytest

from selfplay.train import IGNORE_INDEX, SafetyCollator, tokenize_example


class FakeTokenizer:
    """Word-level stand-in: no model download needed for these invariants."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts)

    def __call__(self, text, truncation=False, max_length=None):
        ids = list(range(1, len(text.split()) + 1))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_labels_match_input_ids_length():
    row = tokenize_example(
        {"prompt": "a b c", "response": "d e f"}, FakeTokenizer(), max_seq_length=64
    )
    assert row is not None
    assert len(row["labels"]) == len(row["input_ids"])


def test_prompt_tokens_are_masked_and_response_is_not():
    row = tokenize_example(
        {"prompt": "a b c", "response": "d e f"}, FakeTokenizer(), max_seq_length=64
    )
    assert row["labels"][0] == IGNORE_INDEX
    assert any(label != IGNORE_INDEX for label in row["labels"])


def test_truncation_that_removes_the_response_drops_the_example():
    # max_seq_length below the prompt length: nothing left to supervise.
    row = tokenize_example(
        {"prompt": "a b c d e f g h", "response": "i j k"},
        FakeTokenizer(),
        max_seq_length=3,
    )
    assert row is None


def test_collator_pads_labels_with_ignore_index():
    collator = SafetyCollator(FakeTokenizer())
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
    )
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["labels"][1].tolist() == [4, IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][1].tolist() == [1, 0, 0]


def test_collator_rejects_length_mismatch():
    collator = SafetyCollator(FakeTokenizer())
    with pytest.raises(AssertionError):
        collator([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2, 3]}])
