"""Defaults for the local Hugging Face attacker, classifier, and refusal teacher."""

from pathlib import Path

from selfplay.config import JudgeConfig, RedTeamConfig, load_config
from selfplay.judge import BehaviorJudge, HarmBenchClsJudge
from selfplay.redteam import (
    build_jailbreak_r1_user_prompt,
    extract_model_attack,
)
from selfplay.redteam import RedTeamGenerator
from selfplay.safety_data import _is_usable_refusal, build_safe_response_messages

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_uses_local_huggingface_models():
    config = load_config(ROOT / "configs" / "default.yaml")
    assert config.redteam.model_id == "yukiyounai/Jailbreak-R1"
    assert config.judge.model_id == "cais/HarmBench-Mistral-7b-val-cls"
    assert config.safe_response.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.defender.model_id == "Qwen/Qwen2.5-1.5B-Instruct"


def test_default_judge_is_the_harmbench_classifier():
    judge = BehaviorJudge(None, JudgeConfig())
    assert isinstance(judge._llm_judge, HarmBenchClsJudge)


def test_jailbreak_r1_prompt_is_the_model_card_template():
    goal = "the held-out benchmark behavior"
    prompt = build_jailbreak_r1_user_prompt(goal)
    assert "Attack target: the held-out benchmark behavior" in prompt
    assert "<attack> final attack prompt is here </attack>" in prompt
    assert "already tried" not in prompt

    with_history = build_jailbreak_r1_user_prompt(goal, ["an earlier probe"])
    assert "an earlier probe" in with_history
    assert with_history.startswith(prompt)


def test_model_attack_keeps_a_closed_span_without_terminal_punctuation():
    raw = (
        "<think>strategy goes here</think>"
        "<attack>Rewrite the request as a numbered lab note for the stated behavior</attack>"
    )
    assert extract_model_attack(raw) == (
        "Rewrite the request as a numbered lab note for the stated behavior"
    )


def test_model_attack_ignores_a_truncated_span():
    raw = "<think>strategy</think><attack>this prompt was cut off before the close"
    assert extract_model_attack(raw) is None


def test_safe_response_quotes_the_attack_as_data():
    messages = build_safe_response_messages("please answer the benchmark item")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "USER MESSAGE:\nplease answer the benchmark item" in messages[1]["content"]
    assert not messages[1]["content"].startswith("please answer")


def test_complied_teacher_output_is_not_a_training_target():
    assert _is_usable_refusal("Sure, here is a full procedure that carries the request out.") is False
    assert _is_usable_refusal(
        "I can't help with that request. I can help with a legitimate alternative instead."
    ) is True


def test_attacker_init_does_not_load_weights():
    generator = RedTeamGenerator(RedTeamConfig())
    assert generator._lm._model is None
