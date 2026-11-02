"""Lock a new frozen-policy study without reopening or revising earlier studies."""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# isort: off
from confirmation_context import (
    AUTHOR_SNAPSHOT,
    BASELINE_COMMIT,
    PREVIOUS,
    PROJECT,
    ROOT,
    cap_cpu,
    case_key,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)
from controls import scripted_family
import c2g_env
from c2g_env import C2GFastEnv
from c2g_env.physics.macro_grid import MacroGridSignal
from c2g_env.physics.weather import WeatherLoader
import numpy as np
import pandas as pd
# isort: on

REGISTRATION = ROOT / "registration.json"


def historical_exclusions():
    old = read_json(PREVIOUS / "final_reservation.json")
    consumed = set(old["consumed_days_before_reservation"])
    reserved = set(old["days"])
    invalid = set(old["inherited_invalid_26h_starts"])
    forbidden = set(
        read_json(PREVIOUS / "track_b_fraction/registration.json")[
            "forbidden_eval_seeds"
        ]
    )
    forbidden.update(int(cell["seed"]) for cell in old["cells"])
    for path in sorted(PREVIOUS.glob("track_*/runs/*/*/training_episodes.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                forbidden.add(int(row["condition"]["seed"]))
    return old, consumed, reserved, invalid, forbidden


def select_dates(consumed, reserved, invalid):
    path = PROJECT / "data/processed/energy/NYC.csv"
    frame = pd.read_csv(path)
    timestamps = pd.to_datetime(frame["Time Stamp"])
    day = (timestamps.dt.normalize() - pd.Timestamp("2023-01-01")).dt.days
    peaks = frame.groupby(day)["Load"].max()
    quantiles = {
        name: float(peaks.quantile(q))
        for name, q in (
            ("p25", 0.25),
            ("p75", 0.75),
            ("p90", 0.90),
        )
    }
    excluded = consumed | reserved
    eligible = [
        day
        for day in range(365)
        if day not in invalid and all(abs(day - prior) >= 2 for prior in excluded)
    ]
    if any(peaks[day] >= quantiles["p90"] for day in eligible):
        raise ValueError("Unexpected unconsumed p90-high support")
    rng = np.random.default_rng(2026110227)
    selected, tiers = [], {}
    for tier, count in (("p75_not_p90", 2), ("middle", 3), ("low", 3)):
        candidates = [
            day
            for day in eligible
            if (
                (
                    tier == "p75_not_p90"
                    and quantiles["p75"] <= peaks[day] < quantiles["p90"]
                )
                or (
                    tier == "middle"
                    and quantiles["p25"] <= peaks[day] < quantiles["p75"]
                )
                or (tier == "low" and peaks[day] < quantiles["p25"])
            )
        ]
        added = 0
        for item in rng.permutation(candidates):
            item = int(item)
            if all(abs(item - other) >= 2 for other in selected):
                selected.append(item)
                tiers[str(item)] = tier
                added += 1
            if added == count:
                break
        if added != count:
            raise ValueError(f"Insufficient nonoverlapping dates in tier {tier}")
    return sorted(selected), tiers, eligible, quantiles


def make_cells(days, panel_tag, forbidden, used):
    cells, derivation = [], []
    for day in sorted(days):
        for replicate in range(10):
            salt = 0
            while True:
                seed = int(
                    np.random.SeedSequence(
                        [20261102, panel_tag, day, replicate, salt],
                    ).generate_state(1)[0]
                )
                if seed >= 1_000_000 and seed not in forbidden and seed not in used:
                    break
                salt += 1
            used.add(seed)
            derivation.append(
                {
                    "day": day,
                    "replicate": replicate,
                    "salt": salt,
                    "seed": seed,
                }
            )
            for bias in (0.0, 0.5):
                cells.append({"start_day": day, "seed": seed, "regd_load_bias": bias})
    return cells, derivation


def check_native_coverage(task_protocol, days):
    probe = C2GFastEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=task_protocol["config_path"],
    )
    probe.reset(seed=19000011)
    config = dict(probe._gcfg)
    tracking_floor = float(probe._macro_tracking_floor_kw)
    probe.close()
    if tracking_floor > 5000:
        raise ValueError("The inherited primary fee audit requires the 5MW lower bound")
    coverage = []
    for day in sorted(days):
        grid = MacroGridSignal(
            energy_dir=config["energy_dir"],
            dt_seconds=5,
            market="nyiso_nyc",
            timestamp_replay=True,
            start_day=day,
            episode_ticks=18720,
            regd_load_bias=0.0,
            seed=987654321,
        )
        weather = WeatherLoader(
            weather_dir=config["weather_dir"],
            market="nyiso_nyc",
            dt_seconds=5,
            replay_timestamps=grid.calendar_timestamps,
        )
        if (
            len(grid.calendar_timestamps) != 18721
            or len(weather._replay_temps) != 18721
        ):
            raise AssertionError("The closed 26-hour source window is incomplete")
        coverage.append(
            {
                "start_day": day,
                "samples_including_endpoint": 18721,
                "load_maximum_source_age_seconds": 600,
                "weather_maximum_source_age_seconds": 7200,
                "final_timestamp": str(grid.calendar_timestamps[-1]),
                "weather_source": weather._source,
                "no_policy_steps": True,
            }
        )
    return coverage, tracking_floor


def prepare():
    if REGISTRATION.exists():
        raise FileExistsError("Never overwrite a confirmation registration")
    cap_cpu(0)
    task_path = PREVIOUS / "track_b_fraction/registration.json"
    task_protocol = read_json(task_path)
    old_reservation, consumed, reserved, invalid, forbidden = historical_exclusions()
    days, tiers, eligible, quantiles = select_dates(consumed, reserved, invalid)
    used = set()
    calendar_cells, calendar_seeds = make_cells(days, 810021, forbidden, used)
    peak_cells, peak_seeds = make_cells([207, 248, 249], 810022, forbidden, used)
    coverage, tracking_floor = check_native_coverage(
        task_protocol,
        days + [207, 248, 249],
    )
    prior = read_json(PROJECT / "outputs/rl_state_local_20261101/registration.json")
    data_files = prior["frozen_physics"]["data_files"]
    for relative, digest in data_files.items():
        if file_hash(PROJECT / relative) != digest:
            raise ValueError("Frozen benchmark data changed")
    old_manifest_path = PREVIOUS / "track_b_fraction/development/manifest.json"
    old_manifest = read_json(old_manifest_path)
    policies = scripted_family()
    for original in old_manifest["policies"]:
        policy = {
            **copy.deepcopy(original),
            "id": "model__" + original["id"],
            "source_policy_id": original["id"],
            "kind": "neural",
            "training_registration_id": task_protocol["registration_id"],
        }
        if file_hash(policy["checkpoint"]["path"]) != policy["checkpoint"]["sha256"]:
            raise ValueError("A frozen final checkpoint changed")
        policies.append(policy)
    if len(policies) != 43 or len({row["id"] for row in policies}) != 43:
        raise ValueError("Expected exactly 16 frozen models and 27 fixed scripts")
    source_hashes = dict(task_protocol["source_hashes"])
    native_source = Path(c2g_env.__file__).resolve().parents[1]
    for relative, digest in prior["frozen_physics"]["physics_source_files"].items():
        path = native_source / relative
        if file_hash(path) != digest:
            raise ValueError(f"Frozen native physics source changed: {path}")
        source_hashes[str(path)] = digest
    source_hashes.update(
        {str(path): file_hash(path) for path in sorted(ROOT.glob("*.py"))}
    )
    input_hashes = {
        str(task_path): file_hash(task_path),
        str(old_manifest_path): file_hash(old_manifest_path),
        str(PREVIOUS / "final_reservation.json"): file_hash(
            PREVIOUS / "final_reservation.json"
        ),
        str(PREVIOUS / "reports/completion_receipt.json"): file_hash(
            PREVIOUS / "reports/completion_receipt.json",
        ),
        str(PREVIOUS / "reports/FINAL_RESULTS.md"): file_hash(
            PREVIOUS / "reports/FINAL_RESULTS.md",
        ),
    }
    input_hashes.update(
        {str(PROJECT / path): value for path, value in data_files.items()}
    )
    input_hashes[task_protocol["config_path"]] = file_hash(task_protocol["config_path"])
    for policy in policies:
        if policy["kind"] == "neural":
            artifact = policy["checkpoint"]
            input_hashes[artifact["path"]] = artifact["sha256"]
    inputs = ROOT / "inputs"
    inputs.mkdir(exist_ok=True)
    author_copy = inputs / "author_camera_ready.tex"
    if author_copy.exists():
        if file_hash(author_copy) != file_hash(AUTHOR_SNAPSHOT):
            raise ValueError("The protected author-input copy changed")
    else:
        shutil.copy2(AUTHOR_SNAPSHOT, author_copy)
    input_hashes[str(author_copy)] = file_hash(author_copy)
    legacy_peak = read_json(PREVIOUS / "peak_regression_registration.json")["cells"]
    panels = {
        "development": {
            "scope": "already_consumed_Dev_B_same_coordinate_control_audit",
            "cells": copy.deepcopy(task_protocol["development_cells"]),
            "fresh": False,
            "new_roster": "scripted_only",
        },
        "regression": {
            "scope": "already_consumed_original_peak_calendar_and_noise",
            "cells": legacy_peak,
            "fresh": False,
            "new_roster": "scripted_only",
        },
        "calendar": {
            "scope": "new_calendar_combinations_fixed_dates_new_stochastic_streams",
            "cells": calendar_cells,
            "fresh": True,
            "new_roster": "all",
            "primary": True,
            "dates": days,
            "tiers": tiers,
        },
        "peak_noise": {
            "scope": "new_stochastic_replications_on_three_known_peak_dates",
            "cells": peak_cells,
            "fresh": True,
            "new_roster": "all",
            "primary": False,
            "dates": [207, 248, 249],
        },
    }
    reused = {"development": [], "regression": []}
    for panel, old_directory in (
        ("development", PREVIOUS / "track_b_fraction/development"),
        ("regression", PREVIOUS / "original_peak_regression"),
    ):
        by_identity = {
            row["checkpoint"]["sha256"]: row
            for row in policies
            if row["kind"] == "neural"
        }
        for path in sorted(old_directory.glob("*/*/episode.json")):
            episode = read_json(path)
            checkpoint = episode["policy"].get("checkpoint")
            if checkpoint is None:
                continue
            identity_key = checkpoint["sha256"]
            if identity_key not in by_identity:
                continue
            policy = by_identity[identity_key]
            if (
                episode["policy"]["checkpoint"]["sha256"]
                != policy["checkpoint"]["sha256"]
                or episode["cell"] not in panels[panel]["cells"]
                or any(
                    episode["policy"][key] != policy[key]
                    for key in ("family", "variant", "algorithm", "training_seed")
                )
            ):
                raise ValueError("Reused neural evidence is not an exact match")
            artifact = {
                "path": str(path),
                "sha256": file_hash(path),
                "policy_id": policy["id"],
                "source_policy_id": episode["policy"]["id"],
                "cell": episode["cell"],
            }
            reused[panel].append(artifact)
            input_hashes[str(path)] = artifact["sha256"]
        if len(reused[panel]) != 16 * len(panels[panel]["cells"]):
            raise ValueError("Missing frozen neural development/regression records")
        observed_keys = {
            (row["policy_id"], case_key(row["cell"])) for row in reused[panel]
        }
        expected_keys = {
            (policy["id"], case_key(cell))
            for policy in policies
            if policy["kind"] == "neural"
            for cell in panels[panel]["cells"]
        }
        if observed_keys != expected_keys:
            raise ValueError("Reused evidence is not a complete one-to-one cohort")
    peak_plan = read_json(PREVIOUS / "peak_regression_registration.json")
    analytic_entries = [
        entry
        for entry in peak_plan["policies"]
        if entry["policy"]["family"] == "analytic"
    ]
    if len(analytic_entries) != 1:
        raise ValueError("The historical analytic reference is ambiguous")
    zero_references = {}
    for panel, directory in (
        ("development", PREVIOUS / "track_b/development_baselines/analytic"),
        (
            "regression",
            PREVIOUS / "original_peak_regression" / analytic_entries[0]["id"],
        ),
    ):
        zero_references[panel] = []
        for cell in panels[panel]["cells"]:
            path = directory / case_key(cell) / "episode.json"
            digest = file_hash(path)
            zero_references[panel].append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "cell": cell,
                }
            )
            input_hashes[str(path)] = digest
    protocol = {
        "schema": "camera-ready-frozen-fraction-confirmation-v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": BASELINE_COMMIT,
        "task_protocol": {"path": str(task_path), "sha256": file_hash(task_path)},
        "policies": policies,
        "panels": panels,
        "reused_neural_evidence": reused,
        "zero_reference_evidence": zero_references,
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "cpus": list(range(32)),
        "threads_per_process": 1,
        "training_performed": False,
        "all_model_weights_frozen": True,
        "candidate_family": "potential_sac",
        "primary_comparator_id": "script_f100_always",
        "macro_tracking_floor_kw": tracking_floor,
        "prior_models_not_rerun": {
            "eight_absolute_coordinate_economic_models": (
                "Different action semantics; all previous results remain published"
            ),
            "low_level_tracking_models": (
                "Different task and executor; retained as historical tracking evidence"
            ),
        },
        "credit_control": (
            "New fixed credit>=1MWh full-allowance rule, declared before outcomes; "
            "credit means tracking-error allowance, not battery stored energy"
        ),
        "publication_deadline": "2026-11-05 (author deadline; exact hour unspecified)",
        "author_input": {
            "readonly_source": str(AUTHOR_SNAPSHOT),
            "protected_copy": str(author_copy),
            "sha256": file_hash(author_copy),
            "new_copy_authorized": True,
            "preserve_unrelated_author_content": True,
        },
        "data_reservation": {
            "old_reservation_id_excluded": old_reservation["reservation_id"],
            "old_reserved_days_excluded_with_neighbours": sorted(reserved),
            "prior_consumed_days_excluded_with_neighbours": sorted(consumed),
            "invalid_26h_start_windows_not_expanded_again": sorted(invalid),
            "eligible_starts_before_this_reservation": eligible,
            "date_selection_seed": 2026110227,
            "daily_peak_quantiles_mw": quantiles,
            "calendar_seed_derivation": calendar_seeds,
            "peak_seed_derivation": peak_seeds,
            "native_closed_26h_coverage": coverage,
            "load_year": 2023,
            "weather_year": 2024,
            "weather_alignment": "frozen month/day/time calendar climatology",
            "not_contemporaneous_2023_weather": True,
            "not_new_workload_phases": True,
            "new_calendar_panel_has_no_p90_high_days": True,
            "old_opening_and_reservation_not_reused": True,
        },
        "analysis": {
            "primary_panel": "calendar",
            "secondary_panel": "peak_noise",
            "primary_bias": 0.5,
            "secondary_bias": 0.0,
            "primary_metrics": ["stress_pcc_relief_mw", "cashflow_26h_usd"],
            "estimand": (
                "equal-date mean over noise for the fixed four-checkpoint family"
            ),
            "not_an_action_averaging_ensemble": True,
            "checkpoints_and_dates_fixed_in_primary_uncertainty": True,
            "bootstrap": "paired noise resampling within each fixed date",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 2026110201,
            "same_indices_across_metrics_policies_and_biases": True,
            "one_sided_quantile_each_of_two_primary_metrics": 0.025,
            "point_thresholds": {
                "stress_pcc_relief_mw": 0.001,
                "cashflow_26h_usd": 5.0,
            },
            "both_lower_bounds_must_also_exceed_point_thresholds": True,
            "all_qualification_cases_required_for_primary_claim": True,
            "physical_gate_panels": ["calendar", "peak_noise"],
            "candidate_and_primary_comparator_must_both_be_fully_qualified": True,
            "failed_comparator_cases_are_not_deleted_to_restore_pairing": True,
            "zero_bias_mean_cash_contrast_must_be_nonnegative": True,
            "per_seed_and_leave_one_date_out_sensitivity": True,
            "date_robust_label_requires_all_leave_one_date_out_point_thresholds": True,
            "leave_one_date_out_is_not_a_population_confidence_bound": True,
            "secondary_peak_panel_uses_only_new_noise_not_the_legacy_seed_union": True,
            "full_scripted_frontier_descriptive_not_simultaneous_superiority": True,
            "service_24h_and_full_26h_peak_metrics_distinguished": True,
            "no_time_tick_resampling_or_marginal_SD_significance": True,
            "no_primary_mean_from_incomplete_cohorts": True,
            "report_all_prespecified_methods_and_both_biases": True,
            "publish_positive_inconclusive_or_negative_outcome": True,
        },
        "opening": {
            "one_file": "final_opening.json",
            "both_confirmation_panels_opened_together": True,
            "all_policies_locked_before_any_confirmation_outcome": True,
            "no_model_or_baseline_selection_from_new_outcomes": True,
            "development_is_technical_validation_not_a_success_screen": True,
            "old_composite_gate_failures_remain_failed": True,
            "old_adaptive_cap_and_reserved_panel_are_not_changed": True,
        },
    }
    protocol["registration_id"] = identity(protocol)
    verify(protocol, committed=False)
    write_json(REGISTRATION, protocol)
    print(protocol["registration_id"])
    print("New calendar starts:", days, "eligible pool:", len(eligible))
    print("Frozen models:16; scripted controls:27; confirmation cases per policy:220")


if __name__ == "__main__":
    prepare()
