"""HarmBench behaviour loading, category routing and behaviour splits.

HarmBench (Mazeika et al., 2024) ships three functional categories that are
*not* interchangeable:

``standard``
    Self-contained requests. Scored with an LLM judge.
``contextual``
    The ``Behavior`` text alone is incomplete -- it refers to a passage held
    in the ``ContextString`` column ("Slightly modify this text ..."). The
    context must be supplied to the attacker and to the judge, or the task
    is meaningless.
``copyright``
    Verbatim-reproduction requests. HarmBench scores these with a separate
    n-gram/hash classifier, not a harm classifier; a safety judge labels a
    reproduced book passage "safe" every time.

Routing behaviours to the right judge is the job of :mod:`selfplay.judge`;
this module just carries the category through faithfully.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .utils import get_logger

LOGGER = get_logger(__name__)

HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_test.csv"
)

STANDARD = "standard"
CONTEXTUAL = "contextual"
COPYRIGHT = "copyright"


@dataclass(frozen=True)
class Behavior:
    behavior_id: str
    behavior: str
    functional_category: str
    semantic_category: str = ""
    tags: str = ""
    context_string: str = ""

    @property
    def needs_context(self) -> bool:
        return self.functional_category == CONTEXTUAL

    def target_description(self) -> str:
        """The full target spec handed to the attacker and the judge.

        For contextual behaviours this splices in ``ContextString``; omitting
        it leaves a dangling referent and produces unusable attacks.
        """
        if self.needs_context and self.context_string:
            return (
                f"{self.behavior}\n\n"
                f"--- CONTEXT THE REQUEST REFERS TO ---\n"
                f"{self.context_string}\n"
                f"--- END CONTEXT ---"
            )
        return self.behavior

    def to_dict(self) -> dict:
        return asdict(self)


def load_harmbench(csv_path: str | Path) -> list[Behavior]:
    """Load a HarmBench ``*_behaviors_text_*.csv`` file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Fetch it with:\n"
            f"    scripts/fetch_harmbench.sh\n"
            f"or download {HARMBENCH_CSV_URL}"
        )
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    required = {"Behavior", "FunctionalCategory", "BehaviorID"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{csv_path} is missing column(s): {sorted(missing)}")

    behaviors = [
        Behavior(
            behavior_id=row["BehaviorID"],
            behavior=row["Behavior"],
            functional_category=row["FunctionalCategory"],
            semantic_category=row.get("SemanticCategory", ""),
            tags=row.get("Tags", ""),
            context_string=row.get("ContextString") or "",
        )
        for row in rows
    ]
    LOGGER.info("Loaded %d behaviours from %s", len(behaviors), csv_path)
    return behaviors


def filter_by_category(
    behaviors: list[Behavior], allowed: list[str]
) -> list[Behavior]:
    allowed_set = set(allowed)
    unknown = allowed_set - {STANDARD, CONTEXTUAL, COPYRIGHT}
    if unknown:
        raise ValueError(f"Unknown functional category: {sorted(unknown)}")

    kept = [b for b in behaviors if b.functional_category in allowed_set]
    dropped = len(behaviors) - len(kept)
    if dropped:
        LOGGER.info(
            "Dropped %d behaviour(s) outside categories %s",
            dropped,
            sorted(allowed_set),
        )
    missing_ctx = [
        b.behavior_id for b in kept if b.needs_context and not b.context_string
    ]
    if missing_ctx:
        LOGGER.warning(
            "%d contextual behaviour(s) have an empty ContextString: %s",
            len(missing_ctx),
            missing_ctx[:5],
        )
    return kept


def split_behaviors(
    behaviors: list[Behavior],
    num_train: int,
    num_probe: int,
    seed: int,
) -> tuple[list[Behavior], list[Behavior]]:
    """Draw disjoint train and probe behaviour sets.

    The probe set is never trained on. Reporting attack success on behaviours
    the defender was fine-tuned against measures memorisation, not robustness.
    """
    if num_train + num_probe > len(behaviors):
        raise ValueError(
            f"Requested {num_train} train + {num_probe} probe behaviours but "
            f"only {len(behaviors)} are available after filtering."
        )
    pool = list(behaviors)
    random.Random(seed).shuffle(pool)
    train = pool[:num_train]
    probe = pool[num_train : num_train + num_probe]

    overlap = {b.behavior_id for b in train} & {b.behavior_id for b in probe}
    assert not overlap, f"train/probe overlap: {overlap}"

    LOGGER.info(
        "Behaviour split: %d train, %d probe (seed=%d)",
        len(train),
        len(probe),
        seed,
    )
    return train, probe


def category_counts(behaviors: list[Behavior]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in behaviors:
        counts[b.functional_category] = counts.get(b.functional_category, 0) + 1
    return dict(sorted(counts.items()))


def load_retention_prompts(dataset_id: str, n: int, seed: int) -> list[str]:
    """Benign instructions mixed into training to limit over-refusal.

    Training only on (attack -> refusal) pairs teaches the model that
    refusing is always correct. Retention data is what keeps the model
    usable; see README "Over-refusal".
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional at test time
        raise RuntimeError(
            "`datasets` is required for retention data: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_id, split="train")
    column = _first_present(ds.column_names, ["instruction", "prompt", "text"])
    rows = ds.shuffle(seed=seed).select(range(min(n * 3, len(ds))))
    prompts: list[str] = []
    for row in rows:
        # Alpaca rows with a non-empty `input` need it appended to make sense.
        text = (row[column] or "").strip()
        extra = (row.get("input") or "").strip() if "input" in ds.column_names else ""
        if extra:
            text = f"{text}\n\n{extra}"
        if text:
            prompts.append(text)
        if len(prompts) >= n:
            break
    LOGGER.info("Loaded %d retention prompts from %s", len(prompts), dataset_id)
    return prompts


def load_overrefusal_prompts(dataset_id: str, n: int | None = None) -> list[str]:
    """XSTest-style prompts that *look* unsafe but are safe to answer.

    Used to measure exaggerated safety (Roettger et al., 2024). Column names
    differ between XSTest mirrors, so detect rather than hardcode.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional at test time
        raise RuntimeError(
            "`datasets` is required for over-refusal eval: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_id, split="train")
    column = _first_present(ds.column_names, ["prompt", "instruction", "text"])
    label_col = _first_present(
        ds.column_names, ["label", "type", "category"], required=False
    )
    prompts = []
    for row in ds:
        # Keep only the safe half of the suite; the contrast set is genuinely
        # unsafe and refusing those is correct behaviour.
        if label_col and "unsafe" in str(row[label_col]).lower():
            continue
        text = (row[column] or "").strip()
        if text:
            prompts.append(text)
    if n is not None:
        prompts = prompts[:n]
    LOGGER.info("Loaded %d over-refusal prompts from %s", len(prompts), dataset_id)
    return prompts


def _first_present(
    columns: list[str], candidates: list[str], required: bool = True
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(
            f"None of {candidates} found in dataset columns {columns}. "
            f"Point the config at a dataset with a prompt-like column."
        )
    return None
