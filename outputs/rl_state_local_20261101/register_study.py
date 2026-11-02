"""Freeze the paired improvement protocol before any new development outcomes."""

from __future__ import annotations

import copy
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# isort: off
from study import (
    ROOT,
    BASELINE,
    PROJECT,
    HIGH_BASELINE,
    BASELINE_COMMIT,
    VARIANTS,
    baseline_protocol,
    file_hash,
    identity,
    read_json,
    write_json,
    verify_registration,
    FEATURE_CONTRACT,
)
from prepare import choose_cpu_groups
import yaml
# isort: on


def cells(days, seeds, biases):
    return [
        {"start_day": day, "seed": seed, "regd_load_bias": bias}
        for day in days
        for seed in seeds
        for bias in biases
    ]


def validate_split(split, prior_seeds):
    days = split["new_development_days"] + split["new_final_days"]
    previous = split["previously_consumed_days"]
    if len(previous) != 71 or len(set(previous)) != 71:
        raise ValueError("Prior exposure inventory is not the audited 71-start set")
    if len(days) != 12 or any(
        type(day) is not int or not 0 <= day <= 363 for day in days
    ):
        raise ValueError("New start days must be twelve in-year integer starts")
    if any(abs(day - prior) < 2 for day in days for prior in previous):
        raise ValueError("A new 26-hour window overlaps prior exposure")
    if any(
        abs(first - second) < 2
        for index, first in enumerate(days)
        for second in days[index + 1 :]
    ):
        raise ValueError("New 26-hour windows overlap each other")
    coverage = split["coverage_audit"]
    invalid = set(coverage["load_coverage_invalid_days"]) | set(
        coverage["weather_coverage_invalid_days"]
    )
    blocked = {
        day for day in range(365) if any(abs(day - prior) < 2 for prior in previous)
    }
    eligible = set(range(365)) - blocked - invalid
    if (
        len(blocked) != coverage["blocked_by_prior_closed_windows"]
        or len(eligible) != coverage["eligible_unconsumed_starts"]
        or not set(days) <= eligible
    ):
        raise ValueError("Exogenous coverage/exposure inventory does not reconcile")
    for panel, expected in (
        (
            "development",
            {"medium": 2, "low": 2, "p90-high": 1, "p75-high-tail-not-p90-high": 1},
        ),
        ("final", {"medium": 2, "low": 2, "p90-high": 2}),
    ):
        tiers = split[f"{panel}_load_tiers"]
        if (
            set(map(int, tiers)) != set(split[f"new_{panel}_days"])
            or Counter(tiers.values()) != expected
        ):
            raise ValueError("Calendar strata differ from the declared panel design")
    dev, final = split["new_development_seeds"], split["new_final_seeds"]
    if (
        len(dev) != 2
        or len(final) != 3
        or len(set(dev + final)) != 5
        or set(dev + final) & set(prior_seeds)
    ):
        raise ValueError("Evaluation noise-stream namespaces collide")


def draft():
    if (
        Path(choose_cpu_groups.__code__.co_filename).resolve()
        != HIGH_BASELINE / "prepare.py"
    ):
        raise RuntimeError("The frozen CPU-allocation helper was shadowed")
    baseline = baseline_protocol()
    split = read_json(ROOT / "split.json")
    lock = read_json(BASELINE / "final_candidate_lock.json")
    selected_families = {
        "nominal",
        "pid",
        "old_sac",
        "cooling_first",
        "bc_only",
        "guided_sac",
    }
    baselines = [
        {
            **{
                key: policy[key]
                for key in ("id", "family", "hardware", "training_seed", "checkpoint")
                if key in policy
            },
            "macro": "rule",
        }
        for policy in lock["inventory"]
        if policy["available"] and policy["family"] in selected_families
    ]
    source_paths = [
        *sorted(ROOT.glob("*.py")),
        *(BASELINE / name for name in baseline["code_hashes"]),
        *(Path(path) for path in baseline["input_hashes"] if path.endswith(".py")),
    ]
    input_paths = [
        ROOT / "split.json",
        BASELINE / "registration.json",
        BASELINE / "final_candidate_lock.json",
        *[Path(value["path"]) for value in baseline["initializers"].values()],
        *[
            Path(policy["checkpoint"]["path"])
            for policy in baselines
            if "checkpoint" in policy
        ],
        *[
            BASELINE / f"runs/guided_sac_{seed}/initialization.json"
            for seed in baseline["seeds"]
        ],
        *(PROJECT / name for name in split["prior_registration_paths"]),
    ]
    protocol = copy.deepcopy(
        {
            key: baseline[key]
            for key in (
                "config_path",
                "service_gate",
                "recovery",
                "hardware_reference",
                "normalization",
                "train_days",
                "train_biases",
                "initializers",
                "corpus_path",
                "n_envs",
                "algorithm",
                "architecture",
                "learning_starts",
                "critic_calibration_updates",
                "frozen_physics",
            )
        }
    )
    config = yaml.safe_load(Path(protocol["config_path"]).read_text())
    groups = choose_cpu_groups()
    used = {cpu for group in groups for cpu in group}
    used_physical = {
        Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        .read_text()
        .strip()
        for cpu in used
    }
    secondary = []
    for cpu in sorted(os.sched_getaffinity(0)):
        physical = (
            Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            .read_text()
            .strip()
        )
        if physical not in used_physical:
            secondary.append(cpu)
            used_physical.add(physical)
            if len(secondary) == 8:
                break
    if len(secondary) != 8:
        raise RuntimeError(
            "Need eight additional distinct cores for independent macro evaluations"
        )
    protocol.update(
        schema="state-local-reference-SAC-and-causal-quantity-v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        baseline_commit=BASELINE_COMMIT,
        baseline_registration_id=baseline["registration_id"],
        implementation_commit_before_runs_required=True,
        feature_contract=FEATURE_CONTRACT,
        variants=list(VARIANTS),
        seeds=[1100, 1101, 1102, 1103],
        online_seed_by_initializer={
            str(seed): seed + 5000 for seed in range(1100, 1104)
        },
        initializer_actor_digests={
            str(seed): read_json(
                BASELINE / f"runs/guided_sac_{seed}/initialization.json"
            )["actor"]
            for seed in range(1100, 1104)
        },
        reference_weight=0.1,
        reference_sigma=0.01,
        reference_prior={
            "direction": "KL(current pre-tanh Gaussian || frozen BC Gaussian)",
            "sigma_is_fixed_design_prior_not_estimated_uncertainty": True,
            "same_nominal_weight_not_same_realized_gradient_strength": True,
            "intervention": "fixed demonstration versus current replay state source",
            "includes_state_coverage_and_distribution_changes": True,
            "reward_and_scalar_SAC_critics_unchanged": True,
            "forced_policy_deviation_or_action_distance_floor": False,
            "fixed_state_support": (
                "same 72 demonstration episodes on 24 BC-training dates as the "
                "published Guided regularizer; six internal BC-validation "
                "dates excluded"
            ),
            "replay_window": (
                "all collected transitions until capacity; thereafter trailing "
                "200000-transition circular buffer, unchanged for both arms"
            ),
        },
        stage_milestones={
            "smoke": [4096],
            "pilot": [4096, 16384, 69120],
            "extended": [4096, 16384, 69120, 153600, 276480, 552960],
        },
        extension={
            "decision_checkpoint": 69120,
            "all_four_fits_in_at_least_one_arm_must_pass": True,
            "require_all_24_physical_and_recovery_gates_each_fit": True,
            "maximum_rmse_ratio_to_own_BC": 1.25,
            "maximum_rmse_ratio_to_locked_Guided_same_initializer": 0.8,
            "rmse_statistic": "unweighted arithmetic mean of 24 complete-case RMSEs",
            "run_both_arms_if_triggered": True,
            "implementation": (
                "rerun from exact initializer/seed, never replay-only resume"
            ),
            "repeated_prefix_counts_as_compute_not_additional_replication": True,
            "compare_prefix_parameter_digests_to_pilot": True,
            "incomplete_reference_comparator": (
                "halt before RL and preserve the failure; any alternative requires "
                "a new explicit preregistration, never a fabricated full-case mean"
            ),
            "prefix_results_for_publication_if_rerun_differs": "original pilot",
        },
        acceptance={
            "primary_checkpoint": (
                "final registered step only; no early-best substitution"
            ),
            "milestones": (
                "diagnostics, including p75/p90 case-RMSE and qualification counts"
            ),
            "stability_vs_Guided": (
                "all physical gates and RMSE <= 0.8 times matching frozen Guided"
            ),
            "increment_vs_BC": (
                "all physical gates; RMSE <= 0.9 times own BC; biased relief and "
                "26h cash no lower than own BC except roundoff (1e-9 MW, 1e-6 USD)"
            ),
            "BC_retention_is_not_a_control_increment_over_BC": True,
            "guided_reference_selection": (
                "previous development; frozen without reselection here"
            ),
            "threshold_values": (
                "computed from immutable matched Dev-B baselines before training"
            ),
            "no_independent_seed_or_date_inference_or_p_values": True,
        },
        online_exposure={
            "pilot_healthy_path": (
                "4 complete condition episodes and 4 zero-step autoresets"
            ),
            "extended_healthy_path": (
                "32 complete condition episodes and 4 zero-step autoresets"
            ),
            "pilot_is_not_a_90_cell_or_30_day_curriculum_cycle": True,
            "actual_conditions_and_physical_terminal_channels_are_recorded": True,
        },
        development_cells=cells(
            split["new_development_days"],
            split["new_development_seeds"],
            split["biases"],
        ),
        final_cells=cells(
            split["new_final_days"], split["new_final_seeds"], split["biases"]
        ),
        regression_cells=baseline["development_cells"],
        smoke_cells=[{"start_day": 16, "seed": 730002, "regd_load_bias": 0.5}],
        split=split,
        frozen_baseline_policies=baselines,
        service_floor=0.9,
        high_level={
            "ridge_penalty": 10.0,
            "decay_rho": 0.9963,
            "rmcp_normalizer": float(config["handshake"]["rmcp_max"]),
            "guard_ratio": 0.92,
            "planning_performance": 0.98,
            "margin_thresholds_usd_mw_h": [0.0, 5.0],
            "candidate_grid": [["decay", 0.0], ["ridge", 0.0], ["ridge", 5.0]],
            "fixed_hardware_for_selection": "bc_only_1100",
            "minimum_mean_paired_cash_gain_usd_for_advancement": 5.0,
            "require_nonnegative_mean_gain_in_both_bias_strata": True,
            "require_all_physical_recovery_and_90_percent_service_gates": True,
            "current_price_persistence_is_an_approximation": True,
            "forecast_accuracy_alone_cannot_establish_cash_headroom": True,
            "service_floor_definition": (
                "performance-weighted committed capacity-hours divided by maximum "
                "admissible accepted capacity-hours under matching market draws; "
                "standing 5 MW on rejection; not battery energy"
            ),
        },
        gpus=list(range(8)),
        cpu_groups=groups,
        secondary_evaluation_cpus=secondary,
        threads_per_process=1,
        source_hashes={str(path): file_hash(path) for path in source_paths},
        input_hashes={str(path): file_hash(path) for path in input_paths},
        final_guard=(
            "A separate self-hashed final lock is required before final rollouts"
        ),
    )
    validate_split(
        split,
        [
            *range(100, 105),
            *range(1000, 1003),
            *range(1100, 1104),
            *range(6100, 6104),
            *range(90001, 90005),
            91001,
            91002,
            92001,
            92002,
            92003,
            730001,
            730002,
        ],
    )
    return protocol


def register():
    destination = ROOT / "registration.json"
    if destination.exists():
        raise FileExistsError("Never overwrite a scientific registration")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        cwd=PROJECT,
        check=True,
    )
    protocol = draft()
    protocol["registration_id"] = identity(protocol)
    write_json(destination, protocol)
    verify_registration(protocol)
    print(protocol["registration_id"], flush=True)


if __name__ == "__main__":
    register()
