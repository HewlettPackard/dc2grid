"""Shared identities and frozen dependencies; never change CUDA visibility."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT
PREVIOUS = PROJECT / "outputs/physical_recovery_eval_20261031"
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _key in THREAD_KEYS:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))

import numpy as np  # noqa: E402

TASK_SCHEMA = "cash-constrained-quantity-macro-v1"
EXTRA_FEATURES = (
    "remaining_service_fraction",
    "current_signed_regd",
    "last_window_signed_regd_mean",
    "signed_requested_mwh_over_bess_capacity",
    "centered_cash_in_10000_usd",
    "thermal_warning_budget_used",
    "backlog_fraction_of_arrived_work",
    "performance_weighted_capacity_ratio",
    "maximum_accepted_capacity_fraction",
)
OBSERVATION_DIM = 19 + len(EXTRA_FEATURES)
VARIANTS = {"cash": 0.0, "reserve_potential": 0.0003}


def load_json(path: Path):
    return json.loads(path.read_text())


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def cell_key(cell: dict) -> str:
    for key in ("start_day", "seed"):
        value = cell[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"Invalid integer condition field {key}")
    if not np.isfinite(cell["regd_load_bias"]):
        raise ValueError("Nonfinite condition bias")
    bias = f"{float(cell['regd_load_bias']):g}".replace(".", "p")
    return f"d{int(cell['start_day'])}_s{int(cell['seed'])}_b{bias}"


def inherited_contract() -> tuple[dict, dict, dict]:
    registration = load_json(PREVIOUS / "heldout_registration.json")
    policy = next(
        item for item in registration["policies"] if item["id"] == "old_low_1007"
    )
    return (
        policy["hardware_checkpoint"],
        registration["selection_gate"],
        registration["recovery"],
    )


def verify_frozen_imports() -> None:
    for name, module in list(sys.modules.items()):
        if name in ("c2g_env", "baselines") or name.startswith(
            ("c2g_env.", "baselines.")
        ):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(SOURCE):
                raise RuntimeError(f"Unexpected simulator module origin: {name}")


def verify_previous_evidence() -> dict:
    matched_root = PROJECT / "outputs/matched_validation_20261031"
    matched = load_json(matched_root / "registration.json")
    body = {k: v for k, v in matched.items() if k != "registration_id"}
    if identity(body) != matched["registration_id"]:
        raise RuntimeError("Original matched registration changed")
    if file_hash(matched_root / "run_matrix.py") != matched["runner_sha256"]:
        raise RuntimeError("Original matched runner changed")
    for name, expected in matched["source_files"].items():
        if file_hash(SOURCE / name) != expected:
            raise RuntimeError(f"Frozen physics source changed: {name}")
    for name, expected in matched["data_files"].items():
        if file_hash(SOURCE / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    previous = load_json(PREVIOUS / "heldout_registration.json")
    body = {k: v for k, v in previous.items() if k != "registration_id"}
    if identity(body) != previous["registration_id"]:
        raise RuntimeError("Previous held-out registration changed")
    for name, expected in previous["code_files"].items():
        if file_hash(PREVIOUS / name) != expected:
            raise RuntimeError(f"Previous registered implementation changed: {name}")
    for name, expected in previous["protected_inputs"].items():
        if file_hash(name) != expected:
            raise RuntimeError(f"Previous protected evidence changed: {name}")
    if file_hash(PREVIOUS / "selection.json") != previous["selection_sha256"]:
        raise RuntimeError("Previous locked selection changed")
    hardware, _, _ = inherited_contract()
    if file_hash(hardware["path"]) != hardware["sha256"]:
        raise RuntimeError("Frozen SAC1007 hardware checkpoint changed")
    return {
        "physics_source_files": matched["source_files"],
        "data_files": matched["data_files"],
        "prior_registration": previous["registration_id"],
        "hardware": hardware,
    }


def tensor_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenHardware:
    def __init__(self, artifact: dict):
        import time

        from stable_baselines3 import SAC

        from c2g_env.experiments.identity import check_policy_semantics

        started = time.perf_counter()
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Hardware checkpoint hash mismatch")
        self.model = SAC.load(artifact["path"], device="cpu")
        check_policy_semantics(self.model, "physical_v1")
        if self.model.observation_space.shape != (
            18,
        ) or self.model.action_space.shape != (4,):
            raise ValueError("Expected the registered 18D/4D hardware SAC")
        self.model.policy.set_training_mode(False)
        for parameter in self.model.policy.parameters():
            parameter.requires_grad_(False)
        self.digest = tensor_digest(self.model.policy.state_dict())
        self.load_seconds = time.perf_counter() - started
        self.predictions = 0
        verify_frozen_imports()

    def __call__(self, observation, macro_action):
        self.predictions += 1
        return self.model.predict(observation, deterministic=True)[0]

    def verify(self) -> str:
        current = tensor_digest(self.model.policy.state_dict())
        if current != self.digest:
            raise RuntimeError("Frozen hardware tensors changed")
        return current
