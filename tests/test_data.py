"""Behaviour loading, context splicing and the train/probe split."""

import pytest

from selfplay.data import Behavior, filter_by_category, split_behaviors


def _behaviors(n, category="standard"):
    return [
        Behavior(behavior_id=f"b{i}", behavior=f"do thing {i}", functional_category=category)
        for i in range(n)
    ]


def test_contextual_behavior_splices_context():
    b = Behavior(
        behavior_id="edit_passage",
        behavior="Slightly modify this text to suggest wrongdoing",
        functional_category="contextual",
        context_string="The original passage.",
    )
    target = b.target_description()
    # Without this, the attacker is told to modify a text it was never shown.
    assert "The original passage." in target
    assert b.behavior in target


def test_standard_behavior_has_no_context_block():
    b = Behavior(behavior_id="b", behavior="do thing", functional_category="standard")
    assert b.target_description() == "do thing"


def test_filter_by_category():
    mixed = _behaviors(3, "standard") + _behaviors(2, "copyright")
    kept = filter_by_category(mixed, ["standard"])
    assert len(kept) == 3
    assert all(b.functional_category == "standard" for b in kept)


def test_filter_rejects_unknown_category():
    with pytest.raises(ValueError):
        filter_by_category(_behaviors(1), ["nonsense"])


def test_split_is_disjoint_and_deterministic():
    pool = _behaviors(20)
    train_a, probe_a = split_behaviors(pool, num_train=5, num_probe=5, seed=1)
    train_b, probe_b = split_behaviors(pool, num_train=5, num_probe=5, seed=1)

    assert [b.behavior_id for b in train_a] == [b.behavior_id for b in train_b]
    assert [b.behavior_id for b in probe_a] == [b.behavior_id for b in probe_b]
    # Evaluating on trained-on behaviours measures memorisation, not robustness.
    assert not {b.behavior_id for b in train_a} & {b.behavior_id for b in probe_a}


def test_split_rejects_oversized_request():
    with pytest.raises(ValueError):
        split_behaviors(_behaviors(5), num_train=4, num_probe=4, seed=1)
