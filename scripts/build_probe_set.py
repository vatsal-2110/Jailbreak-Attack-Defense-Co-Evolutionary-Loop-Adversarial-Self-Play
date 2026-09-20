#!/usr/bin/env python3
"""Generate and freeze the held-out evaluation probe set.

Why this exists: the original loop measured each checkpoint against freshly
generated, history-adapted attacks, so a change in "ASR" conflated a change
in the defender with a change in the attacker. Nothing was comparable across
rounds.

This script generates attacks **once**, against behaviours the defender is
never trained on, with no attack history, and writes them to disk. Every
checkpoint is then evaluated on that identical fixed set.

Run once per experiment, commit the hash, never regenerate mid-experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import _bootstrap  # noqa: F401  (sys.path side effect)

from selfplay.config import load_config, require_env
from selfplay.data import filter_by_category, load_harmbench, split_behaviors
from selfplay.llm_client import LLMClient
from selfplay.redteam import RedTeamGenerator
from selfplay.utils import ensure_dir, get_logger, set_seed, write_json

LOGGER = get_logger("build_probe_set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--attacks-per-behavior",
        type=int,
        default=None,
        help="Defaults to redteam.attacks_per_behavior from the config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing probe set (invalidates prior results).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    probe_path = ensure_dir(config.probe_dir) / "probe_attacks.json"
    if probe_path.exists() and not args.force:
        raise SystemExit(
            f"{probe_path} already exists. Regenerating it makes new results "
            f"incomparable with old ones. Pass --force if that is intended."
        )

    behaviors = filter_by_category(
        load_harmbench(config.data.harmbench_csv),
        config.data.functional_categories,
    )
    _, probe_behaviors = split_behaviors(
        behaviors,
        num_train=config.data.num_train_behaviors,
        num_probe=config.data.num_probe_behaviors,
        seed=config.data.behavior_split_seed,
    )

    client = LLMClient(
        api_key=require_env("OPENROUTER_API_KEY"),
        base_url=config.openrouter_base_url,
    )
    generator = RedTeamGenerator(client, config.redteam)

    n = args.attacks_per_behavior or config.redteam.attacks_per_behavior
    # round_idx=-1 and no history: the probe set must not adapt to any
    # checkpoint, or it stops being a fixed yardstick.
    records = generator.generate_round(
        round_idx=-1, behaviors=probe_behaviors, num_attacks=n, history=None
    )

    payload = [r.to_dict() for r in records]
    write_json(payload, probe_path)

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(
        {
            "sha256": digest,
            "n_attacks": len(payload),
            "n_behaviors": len(probe_behaviors),
            "behavior_ids": [b.behavior_id for b in probe_behaviors],
            "attacks_per_behavior": n,
            "redteam_model": config.redteam.model_id,
        },
        config.probe_dir / "probe_manifest.json",
    )

    LOGGER.info("Wrote %d probe attack(s) to %s", len(payload), probe_path)
    LOGGER.info("Probe set sha256: %s", digest)


if __name__ == "__main__":
    main()
