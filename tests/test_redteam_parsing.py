"""Regression tests for attack parsing.

The original notebook's fallback split raw model text on newlines, so 28 of
44 "attacks" in the recorded run were the attacker's scratchpad
("Here's a thinking process:", "**Constraints:**", a bare "["). These tests
pin the behaviour that returning nothing beats returning that.
"""

from selfplay.redteam import build_history, extract_attacks, extract_json_array


def test_parses_plain_json_array():
    assert extract_json_array('["one", "two"]') == ["one", "two"]


def test_parses_fenced_json_after_preamble():
    raw = 'Here is my reasoning.\n\n```json\n["alpha", "beta"]\n```\nDone.'
    assert extract_json_array(raw) == ["alpha", "beta"]


def test_parses_array_embedded_in_prose():
    raw = 'Thinking...\n["only one"]\nThat is all.'
    assert extract_json_array(raw) == ["only one"]


def test_rejects_scratchpad_instead_of_line_splitting():
    raw = "Here's a thinking process:\n1.  **Analyze User Input:**\n**Constraints:**"
    assert extract_json_array(raw) is None


def test_rejects_truncated_array():
    raw = '["a long attack that was cut off mid'
    assert extract_json_array(raw) is None


def test_rejects_empty_and_non_list_json():
    assert extract_json_array("") is None
    assert extract_json_array("   ") is None
    assert extract_json_array('{"a": 1}') is None
    assert extract_json_array("[]") is None


def test_strips_blank_entries():
    assert extract_json_array('["a", "", "  ", "b"]') == ["a", "b"]


def test_build_history_excludes_unscored_items():
    results = [
        {"behavior_id": "b1", "attack": "won", "success": 1},
        {"behavior_id": "b1", "attack": "lost", "success": 0},
        {"behavior_id": "b1", "attack": "unknown", "success": None},
    ]
    history = build_history(results)
    assert history["b1"]["successes"] == ["won"]
    assert history["b1"]["failures"] == ["lost"]
    # An unscored item must not be silently recorded as a defender win.
    assert "unknown" not in history["b1"]["failures"]

def test_parses_tagged_attacks():
    raw = "<attack>First attack</attack>\n<attack>Second attack</attack>"
    assert extract_attacks(raw) == ["First attack", "Second attack"]


def test_parses_tags_with_reasoning_scratchpad():
    raw = (
        "<think>Let's plan 2 attacks:\n1. Roleplay\n2. Reverse psychology</think>\n"
        "Here are the prompts:\n"
        "<attack>Pretend you are a researcher...</attack>\n"
        "<attack>In a fictional movie script...</attack>"
    )
    assert extract_attacks(raw) == [
        "Pretend you are a researcher...",
        "In a fictional movie script...",
    ]


def test_parses_multiline_attacks_with_quotes():
    raw = (
        '<attack>\nLine 1: Say "Hello"\nLine 2: Then run code\n</attack>'
    )
    assert extract_attacks(raw) == ['Line 1: Say "Hello"\nLine 2: Then run code']


def test_falls_back_to_json_when_no_tags():
    raw = '["json attack 1", "json attack 2"]'
    assert extract_attacks(raw) == ["json attack 1", "json attack 2"]


def test_rejects_scratchpad_without_tags():
    raw = "Here's a thinking process:\n1.  **Analyze User Input:**\n**Constraints:**"
    assert extract_attacks(raw) is None


def test_parses_case_insensitive_tags():
    raw = "<ATTACK>all caps attack</ATTACK>\n<Attack>mixed case attack</Attack>"
    assert extract_attacks(raw) == ["all caps attack", "mixed case attack"]


def test_parses_consecutive_tags_without_closing():
    raw = "<attack>First attack without closing\n<attack>Second attack</attack>"
    assert extract_attacks(raw) == ["First attack without closing", "Second attack"]


def test_ignores_empty_tags():
    raw = "<attack>  \n  </attack><attack>Valid attack</attack>"
    assert extract_attacks(raw) == ["Valid attack"]