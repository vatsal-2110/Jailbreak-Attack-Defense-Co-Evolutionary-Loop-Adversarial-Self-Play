#!/usr/bin/env python3
"""Run the red-team / defender self-play loop.

Each round:
  1. Jailbreak-R1 generates prompts for the training behaviours, conditioned
     on which of its earlier prompts worked;
  2. the current defender answers them;
  3. the HarmBench classifier labels each answer;
  4. Qwen3 writes a refusal for each successful attack, mixed with benign
     retention data;
  5. the defender is fine-tuned with LoRA and checkpointed.

Attacker, defender, classifier, and refusal teacher are all loaded from
Hugging Face, one at a time, so they do not share the GPU.

Reported robustness numbers do NOT come from this script -- run
`scripts/evaluate.py` against the frozen probe set for those. The in-loop
ASR printed here is on adaptive training-behaviour attacks and is a progress
signal only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path side effect)

from selfplay.config import ExperimentConfig, load_config, require_env
from selfplay.data import filter_by_category, load_harmbench, load_retention_prompts, split_behaviors
from selfplay.defender import attach_lora, load_adapter, load_defender, run_attacks
from selfplay.judge import BehaviorJudge, is_local_harmbench_classifier
from selfplay.metrics import summarize_round
from selfplay.redteam import RedTeamGenerator, build_history
from selfplay.safety_data import build_refusal_dataset, build_retention_dataset, mix_datasets
from selfplay.train import train_round
from selfplay.utils import ensure_dir, free_gpu, get_logger, set_seed, write_json

LOGGER = get_logger("run_selfplay")


def _load_for_generation(config: ExperimentConfig, adapter_dir: Path | None):
    model, tokenizer = load_defender(config.defender)
    if adapter_dir is not None:
        model = load_adapter(model, adapter_dir)
    return model, tokenizer


def _load_for_training(config: ExperimentConfig, adapter_dir: Path | None):
    """Fresh adapter on the base model, or the previous adapter if incremental."""
    model, tokenizer = load_defender(config.defender)
    continue_adapter = (
        config.train.restart_policy == "incremental" and adapter_dir is not None
    )
    if continue_adapter:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
        model = load_adapter(model, adapter_dir, is_trainable=True)
    else:
        model = attach_lora(model, config.lora, for_training=True)
    return model, tokenizer


def _make_judge(config: ExperimentConfig) -> BehaviorJudge:
    if is_local_harmbench_classifier(config.judge.model_id):
        return BehaviorJudge(None, config.judge)
    from selfplay.llm_client import LLMClient

    client = LLMClient(
        api_key=require_env("OPENROUTER_API_KEY"),
        base_url=config.openrouter_base_url,
    )
    return BehaviorJudge(client, config.judge)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.rounds is not None:
        config.num_rounds = args.rounds
    set_seed(config.seed)

    ensure_dir(config.rounds_dir)
    ensure_dir(config.checkpoint_dir)
    write_json({"config": args.config, "num_rounds": config.num_rounds}, config.run_dir / "run_meta.json")

    behaviors = filter_by_category(
        load_harmbench(config.data.harmbench_csv), config.data.functional_categories
    )
    train_behaviors, probe_behaviors = split_behaviors(
        behaviors,
        num_train=config.data.num_train_behaviors,
        num_probe=config.data.num_probe_behaviors,
        seed=config.data.behavior_split_seed,
    )
    write_json(
        {
            "train": [b.to_dict() for b in train_behaviors],
            "probe": [b.to_dict() for b in probe_behaviors],
        },
        config.run_dir / "behavior_split.json",
    )

    generator = RedTeamGenerator(config.redteam)
    judge = _make_judge(config)

    LOGGER.info("Building retention set from the untrained defender ...")
    model, tokenizer = load_defender(config.defender)
    try:
        retention_prompts = load_retention_prompts(
            config.data.retention_dataset,
            n=config.data.num_train_behaviors * config.redteam.attacks_per_behavior * 2,
            seed=config.seed,
        )
        retention_examples = build_retention_dataset(
            model, tokenizer, retention_prompts, config.defender
        )
    finally:
        del model, tokenizer
        free_gpu()
    write_json(retention_examples, config.run_dir / "retention_examples.json")

    all_results: list[dict] = []
    cumulative_refusals: list[dict] = []
    # Checkpoint the defender answered with on this round. None is the base model.
    active_checkpoint: Path | None = None

    for round_idx in range(config.num_rounds + 1):
        LOGGER.info("=" * 70)
        LOGGER.info("ROUND %d", round_idx)
        LOGGER.info("=" * 70)

        history = build_history(all_results)
        attacks = generator.generate_round(
            round_idx=round_idx,
            behaviors=train_behaviors,
            num_attacks=config.redteam.attacks_per_behavior,
            history=history,
        )
        if not attacks:
            LOGGER.error("Round %d produced no parseable attacks; stopping.", round_idx)
            break
        write_json([a.to_dict() for a in attacks], config.rounds_dir / f"round_{round_idx}_attacks.json")

        model, tokenizer = _load_for_generation(config, active_checkpoint)
        try:
            results = run_attacks(model, tokenizer, [a.to_dict() for a in attacks], config.defender)
        finally:
            del model, tokenizer
            free_gpu()

        try:
            results = judge.score_all(results)
        finally:
            judge.unload()
        write_json(results, config.rounds_dir / f"round_{round_idx}_results.json")

        summary = summarize_round(round_idx, results)
        write_json(summary, config.rounds_dir / f"round_{round_idx}_summary.json")
        LOGGER.info(
            "Round %d in-loop (adaptive, train behaviours): %s",
            round_idx,
            summary["overall"],
        )

        all_results.extend(results)

        if round_idx == config.num_rounds:
            LOGGER.info("Final round reached; no further training.")
            break

        successful = [x for x in results if x.get("success") == 1]
        LOGGER.info("Round %d: %d successful attack(s) to train on", round_idx, len(successful))
        if not successful:
            LOGGER.warning("No successful attacks; skipping training for this round.")
            continue

        new_refusals = build_refusal_dataset(config.safe_response, successful)
        cumulative_refusals.extend(new_refusals)

        if config.train.restart_policy == "from_base":
            refusals_for_round = cumulative_refusals
        elif config.train.restart_policy == "incremental":
            refusals_for_round = new_refusals
        else:
            raise ValueError(
                f"Unknown restart_policy {config.train.restart_policy!r}; "
                f"expected 'from_base' or 'incremental'."
            )

        checkpoint_dir = config.checkpoint_dir / f"D{round_idx + 1}"
        model, tokenizer = _load_for_training(config, active_checkpoint)
        try:
            train_examples = mix_datasets(
                refusals_for_round,
                retention_examples,
                ratio=config.train.retention_ratio,
                seed=config.seed + round_idx,
            )
            if not train_examples:
                LOGGER.warning(
                    "Round %d produced no usable refusal examples; skipping training.",
                    round_idx,
                )
                continue
            write_json(train_examples, checkpoint_dir / "train_examples.json")
            train_round(
                model=model,
                tokenizer=tokenizer,
                examples=train_examples,
                output_dir=checkpoint_dir,
                config=config.train,
                max_seq_length=config.defender.max_seq_length,
            )
            active_checkpoint = checkpoint_dir
        finally:
            del model, tokenizer
            free_gpu()

    write_json(all_results, config.run_dir / "all_results.json")
    LOGGER.info("Self-play complete. Run scripts/evaluate.py for reportable numbers.")


if __name__ == "__main__":
    main()
