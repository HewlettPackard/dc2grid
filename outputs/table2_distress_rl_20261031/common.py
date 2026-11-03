"""Isolated Table 2 development, using the previously frozen physical plant."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT
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
    macro = read_json(
        MACRO_STUDY.parent / ("high_macro_recovery_pilot_20261031_v2/registration.json")
    )
    low = next(item for item in previous["policies"] if item["id"] == "old_low_1007")[
        "hardware_checkpoint"
    ]
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


def assert_default_plant(fast) -> None:
    from c2g_env.physics.weather import WEATHER_PRESETS

    thermal, electrical, bess = fast._thermal, fast._elec, fast._bess
    checks = {
        "simulation_mode": (fast.simulation_mode, "physical_v1"),
        "market": (fast._grid._market.zone_csv, "NYC"),
        "facility_cap_kw": (fast._facility_cap_kw, 250000),
        "dt_seconds": (fast._dt, 5),
        "commitment_max_mw": (fast._committed_mw_max, 30),
        "liquid_capacitance": (thermal.C_A, 27000),
        "liquid_conductance": (thermal.K_liq, 35),
        "liquid_supply": (thermal.T_supply_A, 30),
        "air_supply": (thermal.T_supply_B, 20),
        "liquid_cop": (thermal.cop_liquid(), 30),
        "air_capacitance": (thermal.C_B, 10000),
        "air_conductance": (thermal.K_air, 13),
        "air_cop_base": (thermal.COP_base, 3.5),
        "air_cop_temperature_slope": (thermal.COP_alpha, 0.02),
        "envelope_A": (thermal.K_env_A, 0.5),
        "envelope_B": (thermal.K_env_B, 0.5),
        "safe_temperature": (thermal.T_safe, 35),
        "warning_A": (fast._rcfg["T_warn_A"], 33),
        "warning_B": (fast._rcfg["T_warn_B"], 33),
        "weather_preset": (fast._weather._preset, WEATHER_PRESETS["nyiso_nyc"]),
        "max_hvac": (thermal.max_hvac_mw, 50),
        "max_pump": (thermal.P_PUMP_MAX_MW, 1.5),
        "fault_factor": (thermal.fault_factor, 1),
        "transformer_mva": (electrical.S_xfmr_mva, 300),
        "transformer_iron": (electrical.p_iron_mw, 0.15),
        "transformer_copper": (electrical.p_copper_frac, 0.006),
        "zone_A_nameplate": (
            electrical.n_racks_A * electrical.p_max_rack_A_kw / 1000,
            150,
        ),
        "battery_mw": (bess.P_MAX_MW, 5),
        "battery_mwh": (bess.E_NOM_MWH, 15),
        "battery_soc_min": (bess.SOC_MIN, 0.10),
        "battery_soc_max": (bess.SOC_MAX, 0.95),
        "initial_reference_A": (fast._reference_thermal.temp_A, thermal.temp_A),
        "initial_reference_B": (fast._reference_thermal.temp_B, thermal.temp_B),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Unsupported study plant: {name}={actual}, expected {expected}"
            )
