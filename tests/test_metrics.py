"""ASR and over-refusal accounting."""

import pytest

from selfplay.metrics import (
    attack_success_rate,
    looks_like_refusal,
    overrefusal_rate,
    per_category_asr,
)


def test_asr_basic():
    results = [{"success": 1}, {"success": 1}, {"success": 0}, {"success": 0}]
    assert attack_success_rate(results).asr_percent == pytest.approx(50.0)


def test_asr_empty_is_zero_not_error():
    assert attack_success_rate([]).asr_percent == 0.0


def test_per_category_asr_splits_buckets():
    results = [
        {"success": 1, "functional_category": "standard"},
        {"success": 0, "functional_category": "standard"},
        {"success": 1, "functional_category": "contextual"},
    ]
    by_cat = per_category_asr(results)
    assert by_cat["standard"]["asr_percent"] == pytest.approx(50.0)
    assert by_cat["contextual"]["asr_percent"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that request.",
        "I cannot provide that information.",
        "I'm not able to assist with this.",
        "I must decline.",
    ],
)
def test_detects_refusals(text):
    assert looks_like_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here are three tips for staying healthy.",
        "The capital of France is Paris.",
        "You can kill a Python process with `kill -9 <pid>`.",
    ],
)
def test_does_not_flag_normal_answers(text):
    assert not looks_like_refusal(text)


def test_overrefusal_rate():
    responses = ["I can't help with that.", "Sure, here you go.", "Paris."]
    stats = overrefusal_rate(responses)
    assert stats["n"] == 3
    assert stats["n_refused"] == 1
    assert stats["refusal_percent"] == pytest.approx(100 / 3)
