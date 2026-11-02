"""Isolated Table 2 development, using the previously frozen physical plant."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / "outputs/online_stability_pilot_20261031/source"
MACRO_STUDY = PROJECT / "outputs/high_macro_recovery_pilot_20261031_v3"
PREVIOUS = PROJECT / "outputs/physical_recovery_eval_20261031"
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
sys.path.append(str(MACRO_STUDY))

import numpy as np  # noqa: E402


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n"
    )
    temporary.replace(path)


def case_key(cell: dict) -> str:
    return (
        f"d{int(cell['start_day'])}_s{int(cell['seed'])}_"
        f"b{float(cell['regd_load_bias']):g}"
    ).replace(".", "p")


def contracts() -> dict:
    previous = read_json(PREVIOUS / "heldout_registration.json")
    macro = read_json(MACRO_STUDY.parent / (
        "high_macro_recovery_pilot_20261031_v2/registration.json"
    ))
    low = next(
        item for item in previous["policies"] if item["id"] == "old_low_1007"
    )["hardware_checkpoint"]
    return {
        "config_path": previous["config_path"],
        "service_gate": previous["selection_gate"],
        "recovery": previous["recovery"],
        "hardware_reference": low,
        "normalization": macro["normalization"],
        "train_days": macro["train_days"],
        "train_biases": macro["train_biases"],
        "development_cells": macro["development_cells"],
        "sealed_cells": macro["sealed_test_cells"],
    }


def cap_cpu(cpu: int | None = None) -> None:
    import torch

    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
