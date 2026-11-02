"""Preserve the completed read-only PID reproduction; do not rerun it."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SESSION_FILES = Path(
    "/root/.copilot/session-state/589eaa5d-3422-4cb3-97b1-1479f309d6e4/files"
)
DIAGNOSTIC = SESSION_FILES / "pid_relief_attribution_20261102"
EXPECTED_SUMMARY = "8e908a5818765d69fbe2139dcd953940a3349143aa3499e76c81f4f1f324ec0e"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    original = DIAGNOSTIC / "summary.json"
    if sha256(original) != EXPECTED_SUMMARY:
        raise ValueError("The completed diagnostic summary changed")
    report = json.loads(original.read_text())
    expected = {
        (mode, day, seed)
        for mode in ("legacy_v0", "physical_v1")
        for day in (207, 248, 249)
        for seed in range(100, 105)
    }
    observed = {(row["mode"], row["day"], row["seed"]) for row in report["cases"]}
    if observed != expected or len(report["cases"]) != len(expected):
        raise ValueError("The diagnostic is missing or duplicating a case pair")
    for row in report["cases"]:
        if row["agent_ticks"] != 17280 or row["baseline_ticks"] != 17280:
            raise ValueError("A diagnostic pair did not complete 24-hour service")
        np.testing.assert_allclose(
            row["stress_cooling_relief_mw"] + row["stress_other_facility_relief_mw"],
            row["reported_formula_relief_mw"],
            rtol=0,
            atol=1e-10,
        )
    early, corrected = report["summary"]["legacy_v0"], report["summary"]["physical_v1"]
    if round(early["reported_formula_relief_mw"], 2) != 6.74:
        raise ValueError("The legacy reproduction does not recover the archived result")
    np.testing.assert_allclose(
        corrected["reported_formula_relief_mw"],
        5.640798863930732,
        rtol=0,
        atol=1e-10,
    )
    inventory = []
    identities = set()
    for mode, day, seed in sorted(expected):
        for hardware in ("pid", "none"):
            directory = DIAGNOSTIC / mode / f"d{day}_s{seed}" / hardware
            context = json.loads((directory / "episode.json").read_text())
            if (
                context["hardware_steps"] != 17280
                or context["macro_steps"] != 96
                or context["terminated"]
                or context["simulation_mode"] != mode
                or context["hardware"] != hardware
            ):
                raise ValueError(
                    "Diagnostic trace context differs from its declared run"
                )
            identities.add(context["source_id"])
            for name in ("episode.json", "hardware.csv.gz", "macro.csv.gz"):
                path = directory / name
                inventory.append(
                    {
                        "mode": mode,
                        "day": day,
                        "seed": seed,
                        "hardware": hardware,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    if len(identities) != 1:
        raise ValueError("The reproduction used different source implementations")
    load = PROJECT / "data/processed/energy/NYC.csv"
    frame = pd.read_csv(load).dropna(subset=["Load"])
    frame["timestamp"] = pd.to_datetime(frame["Time Stamp"])
    frame = frame.sort_values("timestamp")
    calendar = []
    for day in (207, 248, 249):
        start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=day)
        rows = frame[
            (frame.timestamp >= start)
            & (frame.timestamp < start + pd.Timedelta(days=1))
        ]
        calendar.append(
            {
                "start_day": day,
                "date": str(start.date()),
                "valid_source_rows": len(rows),
                "legacy_last_source_timestamp": str(rows.iloc[287].timestamp),
                "legacy_workload_phase": 0,
                "physical_workload_phase": day % 30,
            }
        )
    destination = ROOT / "reports/pid_semantics_reproduction.json"
    if destination.exists():
        raise FileExistsError("Never overwrite the imported diagnostic record")
    probe = SESSION_FILES / "probe_pid_relief_semantics.py"
    recorded_probe = ROOT / "diagnostics/probe_pid_relief_semantics.py"
    recorded_probe.parent.mkdir(exist_ok=True)
    shutil.copyfile(probe, recorded_probe)
    result = {
        **report,
        "original_summary_sha256": EXPECTED_SUMMARY,
        "diagnostic_run_count": 60,
        "case_pair_count": 30,
        "per_mode_case_pairs": 15,
        "source_id": next(iter(identities)),
        "recorded_probe_source_sha256": sha256(recorded_probe),
        "calendar_replay_metadata": calendar,
        "load_source_sha256": sha256(load),
        "observed_difference": {
            key: corrected[key] - early[key]
            for key in (
                "reported_formula_relief_mw",
                "stress_cooling_relief_mw",
                "stress_other_facility_relief_mw",
                "stress_demanded_response_mw",
                "stress_observed_regd",
                "stress_actual_commitment_mw",
                "stress_nominal_facility_mw",
                "stress_actual_facility_mw",
            )
        },
        "same_date_and_seed_labels_not_identical_realized_inputs": True,
        "observed_power_balance_not_individual_change_causal_attribution": True,
        "no_PID_gain_change_or_SAC_training": True,
        "all_60_trajectories_complete_24h_service": True,
        "recovery_not_run_or_needed_for_this_service_only_metric_reproduction": True,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    inventory_path = ROOT / "reports/pid_semantics_trace_inventory.csv"
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(inventory[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(inventory)
    print("Preserved 30 complete case pairs / 60 trajectories; no new rollouts")
    print(
        "Observed facility-relief change:",
        result["observed_difference"]["reported_formula_relief_mw"],
    )


if __name__ == "__main__":
    main()
