"""Shared helpers: logging, seeding, JSON/JSONL I/O, CPU parallelism."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_T = TypeVar("_T")
_R = TypeVar("_R")

# Spawn a process pool only when there is enough pure-Python work to pay for it.
_PROCESS_MAP_MIN = 256


def available_cpus() -> int:
    """Logical CPUs available to this process."""
    return max(1, os.cpu_count() or 1)


def _export_thread_env() -> None:
    """Point BLAS/OpenMP at every core before native libraries are loaded."""
    n = str(available_cpus())
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


_export_thread_env()


def configure_cpu_parallelism() -> int:
    """Use every core for CPU work. Safe to call more than once."""
    n = available_cpus()
    _export_thread_env()
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            # Interop threads can only be set once, and only before parallel work.
            pass
    except ImportError:
        pass
    return n


def thread_map(fn: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
    """Apply ``fn`` in a thread pool, preserving input order.

    Threads cover I/O and tokenizers that release the GIL. One item stays
    on the caller thread.
    """
    seq = list(items)
    if len(seq) <= 1:
        return [fn(item) for item in seq]
    workers = min(available_cpus(), len(seq))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, seq))


def process_map(fn: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
    """Apply a picklable ``fn`` across processes, preserving input order.

    Short inputs stay in-process: spawning workers would cost more than the work.
    """
    seq = list(items)
    if len(seq) < _PROCESS_MAP_MIN:
        return [fn(item) for item in seq]
    workers = min(available_cpus(), len(seq))
    # spawn, not fork: the parent may already have initialised CUDA.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(fn, seq))


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def free_gpu() -> None:
    """Release cached CUDA memory after the caller has dropped its model refs."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. torch/numpy are seeded only if importable."""
    configure_cpu_parallelism()
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunked(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
