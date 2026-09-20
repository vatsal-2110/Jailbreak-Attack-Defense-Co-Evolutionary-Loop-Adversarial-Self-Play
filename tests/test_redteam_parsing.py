"""Regression tests for attack parsing.

The original notebook's fallback split raw model text on newlines, so 28 of
44 "attacks" in the recorded run were the attacker's scratchpad
("Here's a thinking process:", "**Constraints:**", a bare "["). These tests
pin the behaviour that returning nothing beats returning that.
"""

from selfplay.redteam import build_history, extract_json_array


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
