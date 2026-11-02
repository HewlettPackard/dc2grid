"""Bounded, exogenous-only calendar/split audit; never run a controller."""
from __future__ import annotations

import os
import sys

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS", "POLARS_MAX_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.sched_setaffinity(0, {61})

import argparse
import hashlib
import importlib.util
import json
import signal
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

signal.alarm(240)
PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT / "outputs/high_macro_recovery_pilot_20261031/audits/data_split.json"
SOURCE = PROJECT / "outputs/online_stability_pilot_20261031/source"
OLD = PROJECT / "outputs/corrected_sac_8gpu_20261030"
ONLINE = PROJECT / "outputs/online_stability_pilot_20261031"
MATCHED = PROJECT / "outputs/matched_validation_20261031"
RECOVERY = PROJECT / "outputs/physical_recovery_eval_20261031"
DT = 5
SERVICE_TICKS = 17280
TOTAL_TICKS = 18720
START = date(2023, 1, 1)


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay = load_source("c2g_env.replay", "c2g_env/replay.py")
workload_module = load_source("data_audit_workload", "c2g_env/physics/workload.py")
grid_module = load_source("data_audit_grid", "c2g_env/physics/macro_grid.py")
weather_module = load_source("data_audit_weather", "c2g_env/physics/weather.py")


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative(path):
    return str(path.relative_to(PROJECT))


def calendar(day):
    return (START + timedelta(days=int(day))).isoformat()


def rounded(value):
    return round(float(value), 6)


def age_seconds(times, targets):
    indices = np.searchsorted(times, targets, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("No previous sample")
    return (targets - times[indices]) / np.timedelta64(1, "s")


def get_inputs():
    manifest = read_json(OLD / "manifest.json")
    old_protocol = manifest["protocol"]
    online = read_json(ONLINE / "registration.json")
    matched = read_json(MATCHED / "registration.json")
    development = read_json(RECOVERY / "development_registration.json")
    heldout = read_json(RECOVERY / "heldout_registration.json")
    assert old_protocol["train_days"] == online["environment_protocol"]["train_days"]
    assert old_protocol["validation"] == online["environment_protocol"]["validation"]
    assert development["recovery"]["duration_seconds"] == 7200
    assert heldout["recovery"]["duration_seconds"] == 7200
    return manifest, online, matched, development, heldout


def exposure_windows(inputs):
    manifest, _, matched, development, heldout = inputs
    train = manifest["protocol"]["train_days"]
    val = sorted({c["start_day"] for c in matched["inventory"]["cells"]})
    recovered = sorted({c["start_day"] for c in development["cells"] + heldout["cells"]})
    consumed = sorted({c["start_day"] for c in heldout["cells"]})
    windows = []
    for day in sorted(set(train + val + consumed)):
        roles = []
        if day in train:
            roles.append("registered_training")
        if day in val:
            roles.append("observed_validation_and_development")
        if day in consumed:
            roles.append("consumed_test_regression_only")
        hours = 26 if day in recovered else 24
        windows.append({
            "start_day": day,
            "date": calendar(day),
            "start_hours": day * 24,
            "end_hours_inclusive": day * 24 + hours,
            "duration_hours": hours,
            "roles": roles,
        })
    return windows


def conflicts(day, windows):
    a, b = day * 24, day * 24 + 26
    return [w["start_day"] for w in windows
            if a <= w["end_hours_inclusive"] and w["start_hours"] <= b]


def exogenous():
    energy_path = SOURCE / "data/processed/energy/NYC.csv"
    weather_path = SOURCE / "data/processed/weather/NYC.csv"
    load = pd.read_csv(energy_path, parse_dates=["Time Stamp"])
    load = load.dropna(subset=["Load"]).sort_values("Time Stamp")
    times = load["Time Stamp"].to_numpy(dtype="datetime64[ns]")
    values = load["Load"].to_numpy(dtype=np.float64)
    median = float(np.median(values))
    instantaneous_load_p90 = float(np.quantile(values, .9))
    weather = pd.read_csv(weather_path, parse_dates=["timestamp_utc"])
    weather = weather.dropna(subset=["temp_c"]).sort_values("timestamp_utc")
    weather_times = pd.to_datetime(weather["timestamp_utc"], utc=True).dt.tz_localize(None)
    source_year = int(weather_times.iloc[0].year)
    assert source_year == 2024
    prior = weather_times.iloc[-1].replace(year=source_year - 1).to_datetime64()
    wt = np.concatenate(([prior], weather_times.to_numpy(dtype="datetime64[ns]")))
    wv = weather["temp_c"].to_numpy(dtype=np.float64)
    wv = np.r_[wv[-1], wv]
    workload = workload_module.WorkloadOrchestrator(
        trace_dir=SOURCE / "data/processed/workload_traces",
        dt_seconds=5, start_day=0, physical_accounting=True,
    )
    assert workload.horizon_ticks == 8640
    phase_summaries = []
    for phase in range(30):
        sl = slice(phase * 288, (phase + 1) * 288)
        nominal = (workload._p_flex_arr + workload._p_base_arr)[sl] / 1000
        flexible = (workload._p_flex_arr[sl] - workload.p_flex_idle_kw) / 1000
        phase_summaries.append({
            "phase": phase, "nominal_it_mean_mw": rounded(nominal.mean()),
            "nominal_it_peak_mw": rounded(nominal.max()),
            "flexible_arrival_work_mwh_per_day": rounded(flexible.sum() / 12),
            "rigid_it_mean_mw": rounded(workload._p_base_arr[sl].mean() / 1000),
            "recovery_phase": (phase + 1) % 30,
            "recovery_phase_prefix_hours": 2,
        })
    high_workload = float(np.quantile(
        [p["nominal_it_mean_mw"] for p in phase_summaries], .75))
    rows = []
    for day in range(365):
        day_date = START + timedelta(days=day)
        targets = np.datetime64(day_date.isoformat(), "ns") + (
            np.arange(TOTAL_TICKS + 1) * np.timedelta64(DT, "s")
        )
        mapped = np.concatenate([
            np.datetime64(day_date.replace(year=source_year).isoformat(), "ns")
            + np.arange(SERVICE_TICKS) * np.timedelta64(DT, "s"),
            np.datetime64((day_date + timedelta(days=1)).replace(
                year=source_year).isoformat(), "ns")
            + np.arange(TOTAL_TICKS - SERVICE_TICKS + 1) * np.timedelta64(DT, "s"),
        ])
        row = {"start_day": day, "date": calendar(day), "phase": day % 30,
               "month": day_date.month, "weekday": day_date.strftime("%A"),
               "weekend": day_date.weekday() >= 5, "coverage_errors": []}
        row.update(phase_summaries[day % 30])
        row["upper_quartile_workload_phase"] = row["nominal_it_mean_mw"] >= high_workload
        try:
            loads = replay.sample_previous(times, values, targets, max_age_seconds=600)
            service = loads[:SERVICE_TICKS]
            recovery_load = loads[SERVICE_TICKS:TOTAL_TICKS]
            lmp = np.clip(28 + 18 * np.maximum(0, service - median) / 1000, 0, 500)
            row.update(
                load_mean_mw=rounded(service.mean()),
                load_peak_mw=rounded(service.max()),
                load_p90_mw=rounded(np.quantile(service, .9)),
                recovery_load_mean_mw=rounded(recovery_load.mean()),
                recovery_load_peak_mw=rounded(recovery_load.max()),
                full_26h_load_peak_mw=rounded(loads[:TOTAL_TICKS].max()),
                proxy_lmp_mean_usd_mwh=rounded(lmp.mean()),
                proxy_lmp_peak_usd_mwh=rounded(lmp.max()),
                service_hours_above_source_sample_load_p90=rounded(
                    np.mean(service >= instantaneous_load_p90) * 24),
                load_max_age_seconds=rounded(age_seconds(times, targets).max()),
            )
        except ValueError as error:
            row["coverage_errors"].append({"series": "load", "error": str(error)})
        try:
            temps = replay.sample_previous(wt, wv, mapped, max_age_seconds=7200)
            row.update(
                temperature_mean_c=rounded(temps[:SERVICE_TICKS].mean()),
                temperature_peak_c=rounded(temps[:SERVICE_TICKS].max()),
                full_26h_temperature_peak_c=rounded(temps[:TOTAL_TICKS].max()),
                weather_max_age_seconds=rounded(age_seconds(wt, mapped).max()),
            )
        except ValueError as error:
            row["coverage_errors"].append({"series": "weather", "error": str(error)})
        rows.append(row)
    valid_load = [r for r in rows if "load_peak_mw" in r]
    cuts = {f"daily_peak_load_p{int(q * 100)}_mw":
            rounded(np.quantile([r["load_peak_mw"] for r in valid_load], q))
            for q in [.5, .75, .9, .95]}
    for row in rows:
        if "load_peak_mw" in row:
            row["load_stratum"] = (
                "high" if row["load_peak_mw"] >= cuts["daily_peak_load_p90_mw"] else
                "medium" if row["load_peak_mw"] >= cuts["daily_peak_load_p50_mw"] else
                "low"
            )
    dataset = {
        "load_rows": len(load), "load_first": str(times[0]), "load_last": str(times[-1]),
        "load_median_mw": rounded(median), "load_max_mw": rounded(values.max()),
        "weather_rows": len(weather), "weather_first": str(weather_times.iloc[0]),
        "weather_last": str(weather_times.iloc[-1]),
        "weather_synthetic_previous_timestamp": str(prior), "thresholds": cuts,
        "source_load_sample_p90_mw": rounded(instantaneous_load_p90),
        "phase_nominal_it_mean_p75_mw": rounded(high_workload),
        "daily_peak_quantile_reference_count": len(valid_load),
        "daily_peak_quantile_reference": "All start days with complete 26h load replay; weather validity and prior exposure do not affect this reference.",
    }
    return rows, phase_summaries, dataset


def order_key(role, day, attempt=0):
    return hashlib.sha256(f"20261031:{role}:{attempt}:{day}".encode()).hexdigest()


def new_window(day):
    return {"start_day": day, "start_hours": day * 24,
            "end_hours_inclusive": day * 24 + 26}


def select_split(rows, old_windows):
    valid = [r for r in rows if not r["coverage_errors"]]
    fresh = [r for r in valid if not conflicts(r["start_day"], old_windows)]
    selected = {}
    attempts = {}
    blocked = list(old_windows)
    slots = [
        ("high", [7]), ("high", [8, 9]), ("medium", [1, 2, 3]),
        ("low", [4, 5, 6]), ("medium", [10, 11, 12]),
        ("low", [1, 2, 3, 4, 5, 6, 10, 11, 12]),
    ]
    for role in ["sealed_test", "development"]:
        for attempt in range(200):
            picks = []
            tentative = list(blocked)
            for slot, (stratum, months) in enumerate(slots):
                candidates = sorted(fresh, key=lambda r: order_key(
                    f"{role}:{slot}", r["start_day"], attempt))
                candidates = [r for r in candidates
                              if r["load_stratum"] == stratum and r["month"] in months
                              and r["phase"] not in {x["phase"] for x in picks}
                              and not conflicts(r["start_day"], tentative)]
                if not candidates:
                    break
                row = candidates[0]
                picks.append(row)
                tentative.append(new_window(row["start_day"]))
            if (len(picks) == 6 and any(r["weekend"] for r in picks)
                    and any(r["upper_quartile_workload_phase"] for r in picks)):
                selected[role] = sorted(picks, key=lambda r: r["start_day"])
                blocked = tentative
                attempts[role] = attempt
                break
        else:
            raise ValueError(f"No varied calendar/workload {role} in 200 bounded attempts")

    protected = [w for w in old_windows if w["duration_hours"] == 26]
    protected += [new_window(r["start_day"]) for role in selected for r in selected[role]]
    train_pool = [r for r in valid if not conflicts(r["start_day"], protected)]
    for attempt in range(200):
        picks = []
        for stratum, number in [("high", 6), ("medium", 12), ("low", 12)]:
            for row in sorted(train_pool, key=lambda r: order_key(
                    f"train:{stratum}", r["start_day"], attempt)):
                if sum(x["load_stratum"] == stratum for x in picks) == number:
                    break
                if row["load_stratum"] != stratum:
                    continue
                if row["phase"] in {x["phase"] for x in picks}:
                    continue
                if conflicts(row["start_day"], [new_window(x["start_day"]) for x in picks]):
                    continue
                picks.append(row)
        if len(picks) == 30 and {r["month"] for r in picks} == set(range(1, 13)):
            selected["train"] = sorted(picks, key=lambda r: r["start_day"])
            attempts["train"] = attempt
            return selected, attempts
    raise ValueError("No phase/stratum/month-balanced curriculum in 200 bounded attempts")


def input_inventory(inputs):
    manifest, online, matched, development, heldout = inputs
    fingerprints = {}

    def record(path):
        fingerprints[relative(path)] = sha256(path)

    roots = [
        OLD / "manifest.json", OLD / "protocol.yaml", ONLINE / "registration.json",
        ONLINE / "protocol.yaml", MATCHED / "registration.json",
        RECOVERY / "development_registration.json", RECOVERY / "heldout_registration.json",
        RECOVERY / "recovery.py", RECOVERY / "run_evaluation.py",
        RECOVERY / "environment.yaml", MATCHED / "run_matrix.py",
    ]
    for path in roots:
        record(path)
    source_files = [
        "c2g_env/replay.py", "c2g_env/physics/workload.py", "c2g_env/physics/macro_grid.py",
        "c2g_env/physics/weather.py", "c2g_env/config.yaml", "c2g_env/env_low_level.py",
        "c2g_env/env_high_level.py", "baselines/train_corrected_sac.py",
        "baselines/online_stability.py",
    ]
    source_checks = {}
    for rel in source_files:
        actual = sha256(SOURCE / rel)
        expected = online["source_files"][rel]
        assert actual == expected, f"Frozen source changed: {rel}"
        source_checks[rel] = {"sha256": actual, "matches_online_registration": True}
        record(SOURCE / rel)
    data_checks = {}
    for rel, expected in matched["data_files"].items():
        actual = sha256(SOURCE / rel)
        assert actual == expected == manifest["data"][rel]["sha256"], rel
        data_checks[rel] = {
            "sha256": actual, "matches_matched_and_corrected_registrations": True,
        }
        record(SOURCE / rel)
    stages = []
    protocol = manifest["protocol"]
    protocol_paths = sorted(OLD.glob("seed_*/*/protocol.json"))
    assert len(protocol_paths) == 15
    for path in protocol_paths:
        stage_protocol = read_json(path)
        for key in ("train_days", "train_biases", "validation", "held_out_days"):
            assert stage_protocol[key] == protocol[key], (path, key)
        stage_manifest = read_json(path.parent / "manifest.json")
        status_path = path.parent / "status.json"
        status = read_json(status_path)
        assert stage_manifest["protocol_id"] == manifest["protocol_id"]
        stages.append({
            "path": relative(path.parent), "stage": stage_manifest["stage"],
            "seed": stage_manifest["seed"], "state": status.get("state"),
            "same_registered_dates_and_biases": True,
        })
        for item in (path, path.parent / "manifest.json", status_path):
            record(item)
    online_manifests = []
    for job in online["jobs"]:
        path = ONLINE / "runs" / job["id"] / "manifest.json"
        run = read_json(path)
        assert run["registration_id"] == online["registration_id"]
        online_manifests.append(relative(path))
        record(path)
    statuses = {}
    for name, path in [
        ("online_stability", ONLINE / "status.json"),
        ("matched_validation", MATCHED / "status.json"),
        ("physical_recovery_development", RECOVERY / "development/status.json"),
        ("physical_recovery_consumed_test", RECOVERY / "heldout/status.json"),
    ]:
        status = read_json(path)
        statuses[name] = {key: status[key] for key in
                          ("state", "registration_id", "planned_policy_cells",
                           "executable_policy_cells", "completed_tasks") if key in status}
        assert status["state"] == "completed"
        record(path)
    smoke = read_json(ONLINE / "smoke/registration.json")
    for key in ("train_days", "train_biases", "validation", "held_out_days"):
        assert smoke["environment_protocol"][key] == protocol[key]
    physical_smoke_path = RECOVERY / "smoke/macro1006_d196/episode.json"
    physical_smoke_cell = read_json(physical_smoke_path)["cell"]
    assert physical_smoke_cell["start_day"] == 196
    for path in [
        ONLINE / "smoke/registration.json", ONLINE / "smoke/run/manifest.json",
        ONLINE / "smoke/run_smoke.py", ONLINE / "smoke/environment.yaml",
        physical_smoke_path, MATCHED / "probe/acct_probe.py", MATCHED / "probe/probe2.py",
    ]:
        record(path)
    return {
        "corrected_protocol_id": manifest["protocol_id"],
        "corrected_source_id": manifest["source_id"],
        "registration_ids": {
            "online": online["registration_id"], "matched": matched["registration_id"],
            "recovery_development": development["registration_id"],
            "recovery_consumed_test": heldout["registration_id"],
        },
        "completed_campaign_bookkeeping": statuses,
        "corrected_stages": stages,
        "online_run_manifest_count": len(online_manifests),
        "online_run_manifests": online_manifests,
        "source_checks": source_checks, "data_checks": data_checks,
        "smoke_and_probe_exposures": {
            "online": "12-tick smoke uses the same registered 24 training dates and four validation cells; no additional calendar starts.",
            "matched_probes": "Both bounded probe source files declare day 196; already included.",
            "physical_recovery_smoke_cell": physical_smoke_cell,
        },
        "input_file_sha256": fingerprints,
        "fields_not_used": [
            "controller returns, rewards, tracking, qualification, rankings or settlement outcomes",
            "policy-selection and checkpoint-selection results embedded in registrations",
            "status.outcomes",
        ],
        "not_opened": [
            "replay buffers", "model or source archives", "hardware/macro/recovery trace files",
            "candidate policy performance on any proposed sealed-test window",
            "external demo directory or attachment",
        ],
    }


def describe_days(days):
    return [{"start_day": d, "date": calendar(d), "phase": d % 30} for d in days]


def summary(rows):
    return {
        "date_count": len(rows),
        "phase_count": len({r["phase"] for r in rows}),
        "phase_counts": dict(sorted(Counter(r["phase"] for r in rows).items())),
        "months": sorted({r["month"] for r in rows}),
        "load_strata": dict(Counter(r["load_stratum"] for r in rows)),
        "weekend_dates": [r["start_day"] for r in rows if r["weekend"]],
        "upper_quartile_workload_dates": [
            r["start_day"] for r in rows if r["upper_quartile_workload_phase"]],
        "service_daily_mean_load_range_mw": [
            min(r["load_mean_mw"] for r in rows), max(r["load_mean_mw"] for r in rows)],
        "service_daily_peak_load_range_mw": [
            min(r["load_peak_mw"] for r in rows), max(r["load_peak_mw"] for r in rows)],
        "service_proxy_lmp_daily_mean_range_usd_mwh": [
            min(r["proxy_lmp_mean_usd_mwh"] for r in rows),
            max(r["proxy_lmp_mean_usd_mwh"] for r in rows)],
        "service_proxy_lmp_peak_usd_mwh": max(r["proxy_lmp_peak_usd_mwh"] for r in rows),
        "service_temperature_range_of_daily_means_c": [
            min(r["temperature_mean_c"] for r in rows),
            max(r["temperature_mean_c"] for r in rows)],
        "nominal_it_day_mean_range_mw": [
            min(r["nominal_it_mean_mw"] for r in rows),
            max(r["nominal_it_mean_mw"] for r in rows)],
        "date_equal_weight_nominal_it_mean_mw": rounded(np.mean(
            [r["nominal_it_mean_mw"] for r in rows])),
        "total_service_hours_above_source_sample_load_p90": rounded(sum(
            r["service_hours_above_source_sample_load_p90"] for r in rows)),
    }


def check_selected_replay(selected, data):
    checks = {}
    for role, rows in selected.items():
        checks[role] = []
        for row in rows:
            grid = grid_module.MacroGridSignal(
                energy_dir=SOURCE / "data/processed/energy", dt_seconds=DT,
                market="nyiso_nyc", timestamp_replay=True, start_day=row["start_day"],
                episode_ticks=TOTAL_TICKS, regd_load_bias=0.0,
            )
            weather = weather_module.WeatherLoader(
                weather_dir=SOURCE / "data/processed/weather", market="nyiso_nyc",
                dt_seconds=DT, replay_timestamps=grid.calendar_timestamps,
            )
            assert len(grid.calendar_timestamps) == TOTAL_TICKS + 1
            assert len(weather._replay_temps) == TOTAL_TICKS + 1
            assert rounded(grid._calendar_load_mw[:SERVICE_TICKS].mean()) == row["load_mean_mw"]
            assert rounded(grid._calendar_load_mw[:SERVICE_TICKS].max()) == row["load_peak_mw"]
            assert rounded(weather._replay_temps[:SERVICE_TICKS].mean()) == row["temperature_mean_c"]
            assert rounded(weather._replay_temps[:SERVICE_TICKS].max()) == row["temperature_peak_c"]
            assert weather.source == "real:NYC.csv:calendar_climatology_2024"
            assert grid.lmp_base_usd == 28 and grid.lmp_slope == 18
            assert rounded(grid._load_median) == data["load_median_mw"]
            assert row["load_max_age_seconds"] <= 600
            assert row["weather_max_age_seconds"] <= 7200
            checks[role].append({
                "start_day": row["start_day"], "date": row["date"],
                "first_source_timestamp": str(grid.calendar_timestamps[0]),
                "service_end_timestamp": str(grid.calendar_timestamps[SERVICE_TICKS]),
                "last_observation_timestamp_inclusive": str(grid.calendar_timestamps[-1]),
                "first_weather_calendar_target": row["date"].replace("2023-", "2024-", 1),
                "points_including_terminal_observation": TOTAL_TICKS + 1,
                "load_max_age_seconds": row["load_max_age_seconds"],
                "weather_max_age_seconds": row["weather_max_age_seconds"],
                "load_array_sha256": hashlib.sha256(grid._calendar_load_mw.tobytes()).hexdigest(),
                "temperature_array_sha256": hashlib.sha256(weather._replay_temps.tobytes()).hexdigest(),
                "weather_source": weather.source,
                "dewpoint_replay_present": weather._replay_dewpoints is not None,
                "same_as_independent_vectorized_causal_audit": True,
                "policy_actions_executed": 0,
            })
    assert "c2g_env.env_low_level" not in sys.modules
    assert "stable_baselines3" not in sys.modules
    return checks


def create_report(inputs, windows, rows, phases, data, selected, attempts, began):
    inventory = input_inventory(inputs)
    manifest, online, matched, development, heldout = inputs
    old_protocol = manifest["protocol"]
    old_train = old_protocol["train_days"]
    old_val = sorted({c["start_day"] for c in development["cells"]})
    consumed = sorted({c["start_day"] for c in heldout["cells"]})
    prior_full_phases = sorted({d % 30 for d in old_train + old_val + consumed})
    tails = []
    for registration, role in [(development, "development"), (heldout, "consumed_test")]:
        for day in sorted({c["start_day"] for c in registration["cells"]}):
            tails.append({
                "service_start_day": day, "recovery_day": day + 1,
                "date": calendar(day + 1), "phase": (day + 1) % 30,
                "observed_window": "00:00:00 through 02:00:00, including endpoint",
                "registration_id": registration["registration_id"], "role": role,
            })
    partial_phases = sorted({t["phase"] for t in tails} - set(prior_full_phases))
    overlaps = []
    for i, a in enumerate(windows):
        for b in windows[i + 1:]:
            duration = min(a["end_hours_inclusive"], b["end_hours_inclusive"]) - max(
                a["start_hours"], b["start_hours"])
            if duration >= 0:
                overlaps.append({
                    "start_days": [a["start_day"], b["start_day"]],
                    "overlap_hours": duration, "roles": [a["roles"], b["roles"]],
                })
    exclusions = []
    valid_fresh = []
    for row in rows:
        old_conflicts = conflicts(row["start_day"], windows)
        if old_conflicts or row["coverage_errors"]:
            exclusions.append({
                "start_day": row["start_day"], "date": row["date"],
                "overlaps_prior_registered_start_days": old_conflicts,
                "coverage_errors": row["coverage_errors"],
            })
        else:
            valid_fresh.append(row["start_day"])
    all_new = [(role, r["start_day"]) for role, picks in selected.items() for r in picks]
    for i, (role_a, day_a) in enumerate(all_new):
        for role_b, day_b in all_new[i + 1:]:
            assert not conflicts(day_a, [new_window(day_b)]), (role_a, day_a, role_b, day_b)
    for role in ["development", "sealed_test"]:
        assert all(r["start_day"] in valid_fresh for r in selected[role])
    assert {r["phase"] for r in selected["train"]} == set(range(30))
    assert not any(conflicts(r["start_day"], [w for w in windows if w["duration_hours"] == 26])
                   for r in selected["train"])
    biases = {"train": [0.0, 0.25, 0.5], "development": [0.0, 0.5], "sealed_test": [0.0, 0.5]}
    seeds = {"development": [91001, 91002], "sealed_test": [92001, 92002, 92003]}
    split = {}
    for role in ["train", "development", "sealed_test"]:
        picks = selected[role]
        item = {
            "start_days": [r["start_day"] for r in picks],
            "dates": [r["date"] for r in picks],
            "phase_by_start_day": {str(r["start_day"]): r["phase"] for r in picks},
            "biases": biases[role], "summary": summary(picks), "exogenous_rows": picks,
            "service_hours": 24, "reserved_recovery_hours": 2, "reserved_total_hours": 26,
        }
        if role == "train":
            item["date_bias_cells_per_complete_curriculum_sweep"] = len(picks) * len(biases[role])
            item["conditions"] = [{"start_day": r["start_day"], "regd_load_bias": bias}
                                  for r in picks for bias in biases[role]]
            item["registered_old_training_starts_reused"] = sorted(
                set(item["start_days"]) & set(old_train))
            item["training_windows_intersecting_old_training_exposure"] = [
                {"start_day": r["start_day"],
                 "old_start_days": conflicts(r["start_day"], windows)}
                for r in picks if conflicts(r["start_day"], windows)]
            item["old_exposure_note"] = "Training may reuse training exposure; none of these intersections is called untouched. Old development and consumed-test 26h windows are excluded."
        else:
            item["matched_environment_seeds"] = seeds[role]
            item["cells"] = [
                {"start_day": r["start_day"], "regd_load_bias": bias, "seed": seed}
                for r in picks for seed in seeds[role] for bias in biases[role]]
            item["cells_per_controller"] = len(item["cells"])
            item["registered_prior_controller_window_overlap"] = False
        split[role] = item
    replay_checks = check_selected_replay(selected, data)
    exact_lists = {role: value["start_days"] for role, value in split.items()}
    proposal_identity = {
        "start_days": exact_lists, "biases": biases, "environment_seeds": seeds,
        "source_data": inventory["data_checks"], "source_code": inventory["source_checks"],
        "selection_rule": "calendar-strata-phase-and-overlap-v1",
    }
    proposal_id = hashlib.sha256(json.dumps(
        proposal_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {
        "schema": "high-macro-data-split-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "recommended_split_proposal_only_not_training_or_evaluation_authorization",
        "proposal_id": proposal_id,
        "scope": {
            "study": "NEW high-level SAC study",
            "frozen_source_root": relative(SOURCE),
            "audited_campaigns": [relative(p) for p in [OLD, ONLINE, MATCHED, RECOVERY]],
            "attachment": "Separate descriptive public receipt; not inspected, used as evidence, or treated as authorization.",
            "activity": "Read-only provenance and exogenous-data analysis; constructors of data loaders only.",
            "policy_actions_executed": 0, "training_executed": False,
            "checkpoint_selection_executed": False, "policy_evaluation_executed": False,
            "only_written_outputs": [relative(OUTPUT), relative(Path(__file__).resolve())],
            "holdout_claim": "No prior registered controller exposure in the four audited campaigns, including service, recovery and terminal-observation windows. This audit itself inspects only exogenous holdout data.",
        },
        "provenance": inventory,
        "prior_exposure_inventory": {
            "corrected_training": {
                "start_days": old_train, "dates_and_phases": describe_days(old_train),
                "biases": old_protocol["train_biases"],
                "summary": summary([rows[d] for d in old_train]),
                "stage_count": len(inventory["corrected_stages"]),
                "duration_hours": 24,
                "qualification_of_actuality": "15 completed stage manifests/protocols (8 low, 7 high) agree on these dates. They establish the registered sampler support, not each realized random episode. Conservatively reserve every supported full day, even if some episodes ended early.",
            },
            "original_validation": {
                "start_days": old_val, "dates_and_phases": describe_days(old_val),
                "cells": old_protocol["validation"], "duration_hours": 24,
                "summary": summary([rows[d] for d in old_val]),
                "phase_reuse": "Both dates are phase 16, already present at training day 226. The four original cells use different seeds between biases.",
            },
            "online_stability": {
                "registered_training_start_days": online["environment_protocol"]["train_days"],
                "biases": online["environment_protocol"]["train_biases"],
                "validation_cells": online["environment_protocol"]["validation"],
                "completed_run_manifest_count": inventory["online_run_manifest_count"],
                "new_calendar_dates": [],
            },
            "matched_validation": {
                "cells": matched["inventory"]["cells"], "duration_hours": 24,
                "new_calendar_dates": [],
                "note": "8 cells pair two environment seeds per date across biases 0/.5; still only two dates and one full workload phase.",
            },
            "physical_recovery_development": {
                "start_days": old_val, "cells": development["cells"],
                "duration_hours": 26, "recovery_seconds": 7200,
                "next_day_first_2h_exposure": [196 + 1, 256 + 1],
            },
            "consumed_tests_regression_only": {
                "start_days": consumed, "dates_and_phases": describe_days(consumed),
                "cells": heldout["cells"], "duration_hours": 26,
                "summary": summary([rows[d] for d in consumed]),
                "next_day_first_2h_exposure": [207 + 1, 248 + 1, 249 + 1],
                "interpretation": "207/248/249 were held out in the earlier protocol but the completed recovery campaign consumed them. Keep them for labeled regression only, never reseal them or use their performance to choose new cases.",
                "registration_coverage_note": "The registration's policy_actions_executed=0 applies only to its structural replay precheck. The later completed campaign status records 480 executed tasks, so it does not imply unconsumed tests.",
            },
            "conservative_prior_windows": windows,
            "recovery_tail_exposures": tails,
            "prior_26h_trajectory_overlaps": overlaps,
            "overlap_consequence": "d248 and d249 share two recovery/service hours; d249's recovery shares the first two hours of original training day 250. Their old labels do not make these 26h trajectories independent.",
            "full_service_workload_phases_ever_registered": prior_full_phases,
            "additional_partially_exposed_workload_phases": partial_phases,
            "workload_phases_without_registered_full_or_recovery_prefix_exposure": sorted(
                set(range(30)) - set(prior_full_phases) - set(partial_phases)),
        },
        "workload_cycle_audit": {
            "phase_definition": "start_day % 30 at source-calendar midnight; unchanged natural alignment",
            "trace_resolution_seconds": 300, "environment_resolution_seconds": DT,
            "batch_rows": 9504, "batch_days_available": 33,
            "dlrm_rows": 8640, "dlrm_days_available": 30,
            "common_rows_used": 8640, "common_cycle_days": 30,
            "batch_rows_outside_common_cycle_not_used": 864,
            "genai_rows": 288, "genai_cycle_days": 1, "genai_tiles_per_common_cycle": 30,
            "spot_trace": "Not used by the frozen WorkloadOrchestrator.",
            "original_training_phase_counts": dict(sorted(Counter(d % 30 for d in old_train).items())),
            "original_training_unique_phases": sorted({d % 30 for d in old_train}),
            "original_training_unique_phase_fraction": 6 / 30,
            "phase_summaries": phases,
            "summary_units": "Nominal incoming IT demand before control, queue dynamics, cooling or BESS; not measured PCC load or controller performance.",
            "novelty": "New 2023 calendar dates change real grid-load, proxy-price and calendar-climatology combinations; they do not create new workload realizations. GenAI repeats daily even across different 30-day phases.",
            "strict_phase_holdout_conflict": "Full 30-phase training leaves zero unseen workload phases. This recommendation targets unseen calendar combinations, not unseen workload phases. Reserving strict unseen phases instead would require reducing training support, also excluding predecessor-day recovery prefixes and accounting for the inherited low-level policy's prior exposures. Independent additional workload data is needed to satisfy both full-30-phase training and genuinely unseen workload-realization testing.",
        },
        "exogenous_data_audit": {
            **data,
            "calendar_start": "2023-01-01", "start_day_indexing": "zero-based, non-leap source year",
            "grid_market": "nyiso_nyc", "weather_market": "nyiso_nyc",
            "load_target_calendar": "Actual 2023 source-calendar timestamps, no wraparound.",
            "weather_mapping": "Replace the year of each 2023 grid target with 2024 while retaining month/day/time; use NOAA UTC timestamps with timezone removed, exactly as the frozen loader does. This is calendar climatology, not observed 2023 weather and not an ordinal-day shift across leap day.",
            "calendar_edge_rule": "The frozen loader prepends its final weather observation relabeled to 2023-12-31 23:00. This supplies January 1 midnight causally within the mapped climatology year; no substitute load data or stale-value tolerance was added.",
            "price_model": {
                "actual_historical_price_csv_used": False,
                "lmp_formula_usd_mwh": "clip(28 + 18 * max(Load_MW - 5406.2229, 0) / 1000, 0, 500)",
                "rmcp_conditional_mean_formula": "25 + 0.3 * (LMP_proxy - 28), before modeled noise/clipping",
                "scope": "High-price coverage means the frozen load-derived LMP proxy, not independently observed market prices. No RegD, RMCP noise, bids or controller outcomes were simulated.",
                "bias_meaning": "0/.25/.5 are RegD-load-anomaly coupling strengths, not multipliers on incoming IT demand or grid-load magnitude.",
            },
            "coverage": {
                "calendar_starts_checked": 365, "load_valid_26h_starts": 359,
                "both_load_weather_valid_26h_starts": sum(not r["coverage_errors"] for r in rows),
                "load_only_failure_start_days": [r["start_day"] for r in rows
                                                 if any(e["series"] == "load" for e in r["coverage_errors"])],
                "weather_failure_start_days": [r["start_day"] for r in rows
                                               if any(e["series"] == "weather" for e in r["coverage_errors"])],
                "failure_details": [{"start_day": r["start_day"], "date": r["date"],
                                     "errors": r["coverage_errors"]} for r in rows if r["coverage_errors"]],
            },
        },
        "exclusions_and_eligibility": {
            "window_rule": "Closed [start_day 00:00, next day 02:00] for every proposed split window. Earlier 24h runs include their terminal observation; endpoint-only contact is conservatively excluded from fresh evaluation.",
            "prior_window_overlap_excluded_start_days": [
                r["start_day"] for r in rows if conflicts(r["start_day"], windows)],
            "fresh_eligible_calendar_start_days": valid_fresh,
            "fresh_eligible_count": len(valid_fresh),
            "fresh_eligible_by_load_stratum": dict(Counter(
                rows[d]["load_stratum"] for d in valid_fresh)),
            "fresh_candidate_exclusions_with_provenance": exclusions,
            "within_and_across_new_split_26h_overlaps": [],
            "old_development_and_consumed_test_26h_windows_forbidden_in_new_training": True,
            "old_training_exposure_permitted_only_for_new_training": True,
            "minimum_spacing_between_new_start_days": min(
                b - a for a, b in zip(sorted(d for _, d in all_new), sorted(d for _, d in all_new)[1:])),
        },
        "recommended_split": split,
        "selection_and_curriculum": {
            "rule_id": "calendar-strata-phase-and-overlap-v1",
            "uses_policy_performance": False,
            "calendar_workload_alignment_changed": False,
            "ordering": "Reserve sealed test first, then development, then training. Within each bounded attempt, SHA256('20261031:role:attempt:day') orders candidates; evaluation role additionally includes slot index.",
            "evaluation_slots_in_order": [
                "high load, July", "high load, August/September",
                "medium load, January-March", "low load, April-June",
                "medium load, October-December",
                "low load, outside July-September",
            ],
            "evaluation_constraints": [
                "six distinct workload phases per fold",
                "two high, two medium, two low grid-load dates",
                "all four quarters, at least one weekend",
                "at least one upper-quartile nominal-IT workload phase",
                "no registered prior window overlap and no overlap with any other chosen 26h window",
            ],
            "training_constraints": [
                "30 dates, exactly one per workload phase",
                "6 high / 12 medium / 12 low grid-load dates",
                "all 12 source-calendar months",
                "no intra-curriculum 26h overlap, no development or sealed-test overlap",
                "no overlap with old development or consumed-test 26h windows",
            ],
            "strata": "High >= reference daily-peak P90; medium P50 <= peak < P90; low peak < P50. Workload upper quartile uses 30 nominal-IT phase means.",
            "bounded_attempts_used_zero_based": attempts,
            "why_attempts": "Only exogenous coverage and fixed calendar/workload constraints reject a proposal; no controller outcomes are queried.",
            "recommended_training_schedule": "Uniform shuffled complete sweeps of the exact 90 date-by-bias cells, before repeating cells, with seeds logged per episode. If retaining the existing with-replacement sampler instead, keep these supports and audit realized cell/phase counts; membership alone cannot prove coverage.",
            "changes_not_implemented": "No scheduler, learner, environment or recovery runner was changed by this audit.",
            "matched_evaluation": "Use each exact development/test seed at both biases and for every controller; same date/seed/bias cells, unchanged source and replay normalization. Never substitute seeds or days after observing failures.",
            "budget_units": "90 train condition cells per sweep; 24 development and 36 sealed-test cells per controller. Each evaluated cell reserves 24h service plus 2h recovery (18,720 five-second intervals). Seed replicates are nested within six calendar windows, not 24/36 independent dates.",
            "sealed_test_release_rule": "Freeze the full study protocol and candidate-selection decisions using development only, then execute the sealed matrix once under the prespecified failure-retention rule. Accessing controller performance consumes a window even if a run fails; do not silently replenish it.",
            "legacy_regression": "196/256 and consumed 207/248/249 may be labeled regression checks only. They are not the new development selection matrix or sealed tests.",
        },
        "selected_window_replay_verification": {
            "method": "Independent vectorized causal audit plus actual frozen MacroGridSignal and WeatherLoader construction for each selected 26h window; no environment, policy or step/reset methods invoked.",
            "load_max_age_seconds_allowed": 600, "weather_max_age_seconds_allowed": 7200,
            "all_selected_windows_pass": True, "selected_window_count": len(all_new),
            "terminal_observation_included": True,
            "service_recovery_normalization_warning": "These 26h constructors verify data coverage only. Execution must retain the frozen 24h service-load mean/peak for RegD-bias normalization, then append recovery as recovery.py does; do not recompute the service signal using the 26h mean/peak.",
            "checks": replay_checks,
        },
        "limitations": [
            "Performance-sealed within the four registered campaigns, not a claim that no undocumented outside run ever touched these dates. No broad trace-tree or outside-directory search was performed.",
            "The calendar dates are 2023 replay windows for future controller runs, not future real-world observations or a strict chronological-after-training forecasting split.",
            "Full-cycle training means all six sealed workload phases and their recovery prefixes are seen as workload patterns in training. Only calendar combinations are held out.",
            "One training calendar date per phase is a compact curriculum, not full crossing of phase, season and load. The cycle repeats every 30 days, GenAI every day, and calendar autocorrelation remains despite nonoverlapping trajectories.",
            "Exogenous holdout values were used for authorized strata and replay checks. They are not data-invisible; the frozen market model also uses the full source-year median.",
            "Each evaluation fold intentionally has 2/6 high-load dates versus a 10% reference tail; its unweighted mean is a balanced stress-panel estimate, not an unbiased annual average.",
            "NYC only. The 2023 load-derived LMP proxy cannot establish transfer to other markets or independent historical price spikes. Weather is mapped 2024 climatology; timezone-naive load alignment is reproduced rather than independently revalidated.",
            "Six evaluation dates per fold and a single processed workload cycle limit statistical independence and generalization. More seeds cannot substitute for more genuinely distinct dates or workloads.",
            "Causal replay here means previous-sample selection with stated age bounds. Full-year price calibration and the frozen full-service-day RegD anomaly statistics are existing exogenous modeling choices, not a proof that every modeled feature is causal at decision time.",
            "Missing load/weather windows and year-end overflow are excluded rather than interpolated, wrapped, or backfilled past the allowed ages. This coverage filter can itself alter calendar representativeness.",
            "The consumed annual peak tests are not resealed. New high-load windows provide P90-tail coverage but do not reproduce the annual maximum or authorize new performance claims from those old tests.",
        ],
        "alternatives": [],
        "decision_note": "Recommend this one natural-calendar split for broader high-level development and new controller-evaluation combinations. No user decision is needed unless the estimand is changed to strict unseen workload phases or chronological forecasting; neither can be relabeled as achieved by full-30-phase training.",
        "reproducibility": {
            "helper": relative(Path(__file__).resolve()),
            "helper_sha256": sha256(Path(__file__).resolve()),
            "run_from": str(PROJECT),
            "command": "taskset -c 61 .venv/bin/python -B outputs/high_macro_recovery_pilot_20261031/audits/data_split_audit.py",
            "python_executable": sys.executable,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "native_thread_environment": {name: os.environ[name] for name in
                                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "maximum_wall_seconds": 240, "maximum_source_start_days": 365,
            "maximum_sampling_attempts_per_role": 200,
            "elapsed_seconds": rounded(time.monotonic() - began),
        },
    }
    for rel, expected in inventory["input_file_sha256"].items():
        assert sha256(PROJECT / rel) == expected, f"Input changed during audit: {rel}"
    report["provenance"]["all_read_input_fingerprints_unchanged_at_completion"] = True
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT:
        raise SystemExit("Run this helper from the project root.")
    began = time.monotonic()
    inputs = get_inputs()
    windows = exposure_windows(inputs)
    rows, phases, data = exogenous()
    selected, attempt = select_split(rows, windows)
    if args.preview:
        print(json.dumps(data))
        invalid = {r["start_day"]: [e["series"] for e in r["coverage_errors"]]
                   for r in rows if r["coverage_errors"]}
        fresh = [r for r in rows if not r["coverage_errors"]
                 and not conflicts(r["start_day"], windows)]
        print("invalid_26h", json.dumps(invalid))
        print("fresh_counts", dict(Counter(r["load_stratum"] for r in fresh)))
        print("fresh_high_days", [
            (r["start_day"], r["phase"], r["load_peak_mw"],
             r["proxy_lmp_peak_usd_mwh"]) for r in fresh if r["load_stratum"] == "high"
        ])
        print("phase_means", [(r["phase"], r["nominal_it_mean_mw"]) for r in phases])
        print("selection_attempt", attempt)
        for role, picks in selected.items():
            print(role, [(r["start_day"], r["phase"], r["load_stratum"],
                          r["load_peak_mw"]) for r in picks])
        return
    report = create_report(inputs, windows, rows, phases, data, selected, attempt, began)
    OUTPUT.relative_to(PROJECT).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    saved = read_json(OUTPUT)
    assert saved["proposal_id"] == report["proposal_id"]
    assert saved["selected_window_replay_verification"]["all_selected_windows_pass"]
    print(json.dumps({
        "output": relative(OUTPUT), "proposal_id": saved["proposal_id"],
        "start_days": {role: spec["start_days"] for role, spec in saved["recommended_split"].items()},
        "verified_windows": saved["selected_window_replay_verification"]["selected_window_count"],
        "fresh_eligible_count": saved["exclusions_and_eligibility"]["fresh_eligible_count"],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "policy_actions_executed": 0,
    }))


if __name__ == "__main__":
    main()
