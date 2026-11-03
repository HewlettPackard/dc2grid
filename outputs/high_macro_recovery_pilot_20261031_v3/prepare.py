"""Prepare the approved bounded pilot; registration follows validation and review."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

# isort: off
from study_common import (
    ROOT,
    SOURCE,
    TASK_SCHEMA,
    VARIANTS,
    file_hash,
    inherited_contract,
    load_json,
    verify_previous_evidence,
    write_json,
)
import numpy as np
import yaml
from c2g_env import C2GMacroEnv
# isort: on


def environment_config(directory: Path, *, ticks: int = 17280) -> Path:
    config = yaml.safe_load((SOURCE / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = ticks
    for key in ("energy_dir", "weather_dir", "trace_dir"):
        config["global"][key] = str((SOURCE / config["global"][key]).resolve())
    path = directory / "environment.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Different existing environment configuration: {path}")
    if not path.exists():
        path.write_text(text)
    return path


def normalization_bound(config: Path, day: int) -> dict:
    env = C2GMacroEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=config,
        start_day=day,
    )
    try:
        env.reset(seed=314159)
        fast = env._fast_env
        thermal, electrical = fast._thermal, fast._elec
        if (
            fast._facility_cap_kw != 250000
            or fast._dt != 5
            or fast._committed_mw_max != 30
            or thermal.T_safe != 35
            or thermal.T_supply_A != 30
            or thermal.COP_base_liq != 30
        ):
            raise ValueError(
                "Cash enclosure is valid only for the frozen default plant"
            )
        ambient_abs = 60.0
        if np.abs(fast._weather._temps_c).max() > ambient_abs:
            raise ValueError("Weather exceeds the declared enclosure support")
        it_a = electrical.n_racks_A * electrical.p_max_rack_A_kw / 1000
        it_b = electrical.n_racks_B * electrical.p_max_rack_B_kw / 1000
        overshoot_a = (
            fast._dt * (it_a + thermal.K_env_A * 2 * ambient_abs) / thermal.C_A
        )
        overshoot_b = (
            fast._dt * (it_b + thermal.K_env_B * 2 * ambient_abs) / thermal.C_B
        )
        if max(overshoot_a, overshoot_b) >= 1:
            raise ValueError("First-fault temperature bound does not hold")
        reference_a_equilibrium = (
            it_a
            + thermal.K_liq * 0.7 * thermal.T_supply_A
            + thermal.K_env_A * ambient_abs
        ) / (thermal.K_liq * 0.7 + thermal.K_env_A)
        if reference_a_equilibrium >= 40 or thermal.temp_A >= 40:
            raise ValueError("Reference temperature enclosure does not hold")
        ups_bounds = []
        for zone in ("A", "B"):
            nameplate = (
                getattr(electrical, f"n_racks_{zone}")
                * getattr(electrical, f"p_max_rack_{zone}_kw")
                / 1000
            )
            capacity = 1.1 * nameplate
            minimum_it = (
                getattr(electrical, f"n_racks_{zone}")
                * getattr(electrical, f"p_idle_rack_{zone}_kw")
                / 1000
            )
            pdu = 1 + getattr(electrical, f"pdu_loss_frac_{zone}")
            x_min, x_max = minimum_it * pdu / capacity, nameplate * pdu / capacity
            k = getattr(electrical, f"ups_k_loss_{zone}")
            if not 0.01 <= x_min <= x_max <= 1 or k >= 0.5:
                raise ValueError("UPS monotone-envelope assumptions do not hold")
            ups_bounds.append(
                capacity
                * (1 + getattr(electrical, f"ups_k_noload_{zone}"))
                / getattr(electrical, f"ups_eta_peak_{zone}")
            )
        cooling_a = thermal.K_liq * (40 - thermal.T_supply_A) / thermal.cop_liquid()
        power_bound = (
            sum(ups_bounds)
            + cooling_a
            + thermal.max_hvac_mw
            + thermal.P_PUMP_MAX_MW
            + electrical.p_lighting_mw
            + electrical.p_network_mw
            + electrical.p_iron_mw
            + electrical.p_copper_frac * electrical.S_xfmr_mva * 1.5**2
            + fast._bess.P_MAX_MW
        )
        if power_bound >= 400:
            raise ValueError(
                "Registered 400 MW absolute import envelope is insufficient"
            )
        lmp_max = fast._grid._load_to_lmp(fast._grid._load_max)
        cash_bound = math.ceil((24 * 500 * 30 + 26 * lmp_max * 800) / 1000) * 1000
        return {
            "cash_prefix_bound_usd": cash_bound,
            "import_abs_mw": 400.0,
            "nominal_abs_mw": 400.0,
            "lmp_abs_usd_mwh": lmp_max,
            "ambient_abs_C": ambient_abs,
            "formula": "ceil_to_1000(24h*500USD/MWh*30MW + 26h*LMPmax*(400MW+400MW))",
            "derivation": {
                "it_nameplate_A_mw": it_a,
                "it_nameplate_B_mw": it_b,
                "first_fault_temperature_overshoot_A_C": overshoot_a,
                "first_fault_temperature_overshoot_B_C": overshoot_b,
                "reference_A_equilibrium_upper_C": reference_a_equilibrium,
                "temperature_A_bound_C": 40.0,
                "ups_input_upper_mw_by_zone": ups_bounds,
                "chiller_A_upper_mw": cooling_a,
                "derived_import_absolute_upper_mw": power_bound,
                "UPS_proof": "x in [.01,1]; x+k(1-x)^2+k0 increasing for k<.5",
                "scope": (
                    "All permitted prefixes including the first fault step; "
                    "no continuation after faults"
                ),
            },
        }
    finally:
        env.close()


def choose_cpu_groups() -> list[list[int]]:
    def counters():
        result = {}
        for line in Path("/proc/stat").read_text().splitlines():
            name, *values = line.split()
            if name.startswith("cpu") and name[3:].isdigit():
                v = list(map(int, values))
                result[int(name[3:])] = (sum(v[:8]), v[3] + v[4])
        return result

    first = counters()
    time.sleep(2)
    second = counters()
    allowed = os.sched_getaffinity(0)
    seen, candidates = set(), []
    for cpu in sorted(allowed):
        siblings = (
            (
                Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                / "thread_siblings_list"
            )
            .read_text()
            .strip()
        )
        if siblings in seen:
            continue
        seen.add(siblings)
        group = []
        for interval in siblings.split(","):
            ends = interval.split("-")
            group.extend(range(int(ends[0]), int(ends[-1]) + 1))
        usable = sorted(set(group) & allowed)
        utilization = {}
        for item in usable:
            total = second[item][0] - first[item][0]
            utilization[item] = (
                1 - (second[item][1] - first[item][1]) / total if total else 1.0
            )
        representative = min(usable, key=lambda item: (utilization[item], item))
        candidates.append((max(utilization.values()), representative))
    if len(candidates) < 42:
        raise RuntimeError(
            "Insufficient physical CPUs for forty assigned cores and coordination"
        )
    selected = [cpu for _, cpu in sorted(candidates)[:40]]
    return [selected[index : index + 5] for index in range(0, 40, 5)]


def draft() -> dict:
    path = ROOT / "protocol_draft.json"
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing draft")
    inherited = verify_previous_evidence()
    hardware, gate, recovery = inherited_contract()
    split = load_json(ROOT / "audits/data_split.json")["recommended_split"]
    config = environment_config(ROOT)
    protocol = {
        "schema": TASK_SCHEMA,
        "implementation_revision": 3,
        "forward_fix_for_registration": (
            "60757aabef71aca114f3992aaae491aa7a06f1b7b30831ba343887dfac333f5f"
        ),
        "supersedes_invalidated_registration": (
            "990d33de42fec3c0164aadc703ed696f989f1bc5bd0e8c8f72b76703538f7607"
        ),
        "user_approved": {"objective": "regulation_cash", "service_floor_percent": 90},
        "hardware": hardware,
        "service_gate": gate,
        "recovery": recovery,
        "config_path": str(config),
        "normalization": normalization_bound(config, split["train"]["start_days"][0]),
        "minimum_service_fraction": 0.9,
        "service_definition": (
            "Performance-weighted committed capacity-hours divided by the "
            "maximum-residual zero-price offer's accepted capacity-hours with "
            "unit tracking and identical market draws. Not actual battery energy."
        ),
        "potential": {"reserve_target_soc": 0.3, "margin": 0.1},
        "observation_scales": {"cash_usd": 10000.0, "requested_energy_mwh": 15.0},
        "variants": VARIANTS,
        "train_days": split["train"]["start_days"],
        "train_biases": split["train"]["biases"],
        "development_cells": split["development"]["cells"],
        "sealed_test_cells": split["sealed_test"]["cells"],
        "sealed_test_execution_in_this_pilot": False,
        "retired_peak_days": [207, 248, 249],
        "macro_seeds": [5000, 5001, 5002, 5003],
        "n_envs": 4,
        "threads": 1,
        "cpu_groups": choose_cpu_groups(),
        "gpus": list(range(8)),
        "total_timesteps": 20480,
        "milestones": [4096, 8192, 12288, 16384, 20480],
        "learning_starts": 1024,
        "teacher_fraction": 0.98,
        "teacher_exploration": {
            "every_local_steps": 20,
            "uniform_fraction_range": [0.0, 1.0],
            "staggered_per_worker": True,
            "warmup_exploration_actions": 52,
        },
        "supervised_bias_updates": 256,
        "critic_calibration_updates": 512,
        "initial_actor_std": 0.1,
        "algorithm": {
            "learning_rate": 0.0003,
            "buffer_size": 50000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 1.0,
            "ent_coef": 1e-9,
            "train_freq": 8,
            "gradient_steps": -1,
            "policy_kwargs": {"net_arch": [256, 256]},
        },
        "baseline_fractions": [0.9, 0.95, 0.98, 1.0],
        "include_zero_price_capped_rule": True,
        "selection": {
            "require_every_development_cell_qualified": True,
            "score": "mean_closed_26h_cashflow",
            "select_per_training_seed": True,
            "imitation_initialization_is_not_an_RL_checkpoint": True,
            "no_survivor_only_arm_mean": True,
        },
        "input_hashes": {
            "data_split": file_hash(ROOT / "audits/data_split.json"),
            "prior_results_adversary": file_hash(
                ROOT / "audits/prior_results_adversary.json"
            ),
            "market_objective": file_hash(ROOT / "audits/market_objective.json"),
        },
        "inherited_evidence": inherited,
        "study_scope": (
            "Sequential quantity-control feasibility/sample-efficiency pilot with "
            "eight independent macro fits in four paired seeds, one frozen hardware "
            "policy, common analytic market and fixed-quantity baselines. "
            "Not evidence of superiority over an unimplemented robust physical MPC."
        ),
    }
    write_json(path, protocol)
    print("Prepared approved draft:", path)
    print(
        "Cash normalization bound USD:",
        protocol["normalization"]["cash_prefix_bound_usd"],
    )
    print("CPU groups:", protocol["cpu_groups"])
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    draft()
