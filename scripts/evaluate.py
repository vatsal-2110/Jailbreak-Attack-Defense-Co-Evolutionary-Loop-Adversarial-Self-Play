#!/usr/bin/env python3
"""Evaluate checkpoints on the frozen probe set. These are the numbers to report.

Two metrics, always together:

* **ASR** on the frozen probe attacks (held-out behaviours, fixed prompts).
  Lower is better.
* **Over-refusal** on XSTest-style safe prompts. Lower is better.

Reporting ASR alone is meaningless -- a model that refuses every input scores
0% ASR and 100% over-refusal. A checkpoint only improved if ASR fell and
over-refusal did not rise to meet it.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \\
        --checkpoints base runs/jailbreak_selfplay/checkpoints/D1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path side effect)

from selfplay.config import load_config, require_env
from selfplay.data import load_overrefusal_prompts
from selfplay.defender import generate_responses, load_adapter, load_defender, run_attacks
from selfplay.judge import BehaviorJudge, is_local_harmbench_classifier
from selfplay.llm_client import LLMClient
from selfplay.metrics import attack_success_rate, overrefusal_rate, per_category_asr
from selfplay.utils import ensure_dir, get_logger, read_json, set_seed, write_json

LOGGER = get_logger("evaluate")


def evaluate_checkpoint(
    checkpoint: str,
    probe_attacks: list[dict],
    overrefusal_prompts: list[str],
    config,
    judge: BehaviorJudge,
) -> dict:
    model, tokenizer = load_defender(config.defender)
    label = checkpoint
    if checkpoint != "base":
        model = load_adapter(model, checkpoint)
        label = Path(checkpoint).name

    LOGGER.info("[%s] scoring %d probe attack(s)", label, len(probe_attacks))
    results = run_attacks(model, tokenizer, probe_attacks, config.defender)

    over = {"n": 0, "n_refused": 0, "refusal_percent": 0.0}
    if overrefusal_prompts:
        LOGGER.info("[%s] scoring %d benign prompt(s)", label, len(overrefusal_prompts))
        benign_responses = generate_responses(
            model, tokenizer, overrefusal_prompts, config.defender
        )
        over = overrefusal_rate(benign_responses)

    del model
    _free_gpu()

    # Local HarmBench classifiers need the GPU; score after the defender is gone.
    results = judge.score_all(results)
    asr = attack_success_rate(results)

    return {
        "checkpoint": label,
        "path": checkpoint,
        "asr": asr.to_dict(),
        "per_category_asr": per_category_asr(results),
        "overrefusal": over,
        "raw_results": results,
    }


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def render_table(reports: list[dict]) -> str:
    header = (
        "| Checkpoint | Probe ASR % | scored/total | Over-refusal % |\n"
        "| --- | ---: | ---: | ---: |\n"
    )
    rows = "".join(
        "| {c} | {a:.2f} | {s}/{t} | {o:.2f} |\n".format(
            c=r["checkpoint"],
            a=r["asr"]["asr_percent"],
            s=r["asr"]["n_scored"],
            t=r["asr"]["n_total"],
            o=r["overrefusal"]["refusal_percent"],
        )
        for r in reports
    )
    return header + rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="'base' for the untrained model, otherwise adapter directories.",
    )
    parser.add_argument("--probe", default=None, help="Defaults to <probe_dir>/probe_attacks.json")
    parser.add_argument(
        "--skip-overrefusal",
        action="store_true",
        help="Skip the benign-prompt metric (not recommended).",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    probe_path = Path(args.probe or (config.probe_dir / "probe_attacks.json"))
    if not probe_path.exists():
        raise SystemExit(
            f"No probe set at {probe_path}. Build one first:\n"
            f"    python scripts/build_probe_set.py --config {args.config}"
        )
    probe_attacks = read_json(probe_path)
    LOGGER.info("Loaded %d frozen probe attack(s) from %s", len(probe_attacks), probe_path)

    overrefusal_prompts: list[str] = []
    if not args.skip_overrefusal:
        try:
            overrefusal_prompts = load_overrefusal_prompts(config.data.overrefusal_dataset)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load over-refusal set (%s); continuing without it.", exc)

    if is_local_harmbench_classifier(config.judge.model_id):
        judge = BehaviorJudge(None, config.judge)
    else:
        client = LLMClient(
            api_key=require_env("OPENROUTER_API_KEY"), base_url=config.openrouter_base_url
        )
        judge = BehaviorJudge(client, config.judge)

    reports = [
        evaluate_checkpoint(ckpt, probe_attacks, overrefusal_prompts, config, judge)
        for ckpt in args.checkpoints
    ]

    out_dir = ensure_dir(Path(args.out) if args.out else config.run_dir / "eval")
    write_json(reports, out_dir / "eval_reports.json")
    table = render_table(reports)
    (out_dir / "eval_table.md").write_text(table, encoding="utf-8")

    print("\n" + table)
    LOGGER.info("Wrote evaluation to %s", out_dir)


if __name__ == "__main__":
    main()
