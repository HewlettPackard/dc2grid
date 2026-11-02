"""Raw-audit every probe and compare complete, matched development cohorts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

# isort: off
from economy_context import (
    ROOT,
    case_key,
    file_hash,
    read_json,
    verify,
    write_json,
)
from economy_eval import decomposition
import numpy as np
import pandas as pd
# isort: on

METRICS = (
    "stress_facility_relief_mw",
    "stress_pcc_relief_mw",
    "cashflow_26h_usd",
    "tracking_rmse_kw",
    "battery_throughput_26h_mwh",
    "regulation_revenue_usd",
    "actual_pcc_boundary_peak_reduction_mw",
)


def episode_metrics(episode: dict) -> dict:
    if not episode["complete"] or not episode["recovery"]["complete"]:
        return {key: None for key in METRICS}
    return {
        "stress_facility_relief_mw": episode["grid"]["stress_facility_relief_mw"],
        "stress_pcc_relief_mw": episode["grid"]["stress_pcc_relief_mw"],
        "cashflow_26h_usd": episode["cashflow_26h_usd"],
        "tracking_rmse_kw": episode["service"]["tracking_rmse_kw"],
        "battery_throughput_26h_mwh": (
            episode["service"]["battery_throughput_mwh"]
            + episode["recovery"]["ledger"]["battery_throughput_mwh"]
        ),
        "regulation_revenue_usd": episode["regulation_revenue_usd"],
        "actual_pcc_boundary_peak_reduction_mw": episode["grid"][
            "actual_pcc_boundary_peak_reduction_mw"
        ],
    }


def aggregate(episodes: list[dict], cells: list[dict]) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    expected = {case_key(cell) for cell in cells}
    identities = set()
    for episode in episodes:
        key = case_key(episode["cell"])
        identity = (episode["policy"]["id"], key)
        if identity in identities:
            raise ValueError("Duplicate policy/case in an aggregate")
        identities.add(identity)
        grouped[episode["policy"]["family"]][key].append(episode)
    output = {}
    for family, cases in sorted(grouped.items()):
        if set(cases) != expected:
            raise ValueError(f"Incomplete case inventory for {family}")
        model_sets = [{row["policy"]["id"] for row in rows} for rows in cases.values()]
        if any(models != model_sets[0] for models in model_sets):
            raise ValueError(f"Changing model cohort across cases for {family}")
        rows = [row for case in cases.values() for row in case]
        complete = all(row["complete"] and row["recovery"]["complete"] for row in rows)
        qualified = sum(
            bool(row["qualified"] and row["delivered_service_gate"]) for row in rows
        )
        strata = {}
        for label in ("all", "0.0", "0.5"):
            selected = [
                case
                for case in cases.values()
                if label == "all" or case[0]["cell"]["regd_load_bias"] == float(label)
            ]
            if not selected:
                continue
            values = {}
            for metric in METRICS:
                per_case = [
                    [episode_metrics(row)[metric] for row in case] for case in selected
                ]
                if any(value is None for case in per_case for value in case):
                    values[metric] = {"mean": None, "sd": None}
                else:
                    means = np.array([np.mean(case) for case in per_case])
                    values[metric] = {
                        "mean": float(means.mean()),
                        "sd": float(means.std(ddof=1)) if len(means) > 1 else None,
                    }
            strata[label] = {"case_count": len(selected), "metrics": values}
        output[family] = {
            "model_count": len(model_sets[0]),
            "policy_case_count": len(rows),
            "qualified_policy_cases": qualified,
            "all_qualified": qualified == len(rows),
            "complete_cohort": complete,
            "minimum_service_ratio": min(
                row["delivered_service_ratio"] for row in rows
            ),
            "strata": strata,
            "no_survivor_only_mean": True,
        }
    return output


def load_frames(directory: Path, episode: dict) -> dict:
    frames = {}
    for name, digest in episode["trace_hashes"].items():
        path = directory / name
        if file_hash(path) != digest:
            raise ValueError(f"Raw trace identity changed: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def audit(episode: dict, frames: dict, reference: dict, protocol: dict) -> int:
    checks = 0

    def equal(actual, expected, *, atol=1e-7, rtol=1e-10):
        nonlocal checks
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        checks += 1

    frame, macros = frames["hardware.csv.gz"], frames["macro.csv.gz"]
    for name in ("hardware.csv.gz", "recovery.csv.gz"):
        if name not in frames:
            continue
        current = frames[name]
        baseline = reference[name].iloc[: len(current)]
        for column in (
            "lmp",
            "grid_load_mw",
            "T_amb",
            "observed_regd_signal",
            "p_reference_mw",
            "reference_temp_A",
            "reference_temp_B",
        ):
            equal(current[column], baseline[column], atol=0, rtol=0)
        if not np.array_equal(current.timestamp, baseline.timestamp):
            raise AssertionError("Calendar timestamps differ from the frozen control")
        checks += 1
        intervals = pd.to_datetime(current.timestamp, utc=True).diff()
        equal(intervals.iloc[1:].dt.total_seconds(), 5.0, atol=0, rtol=0)
        equal(current.duration_hours, 5 / 3600, atol=1e-15, rtol=0)
        equal(
            current.p_grid_import_mw,
            current.p_facility_mw - current.bess_actual_kw / 1000,
        )
        equal(
            current.delta_p_actual_kw,
            (current.p_reference_mw - current.p_facility_mw) * 1000
            + current.bess_actual_kw,
        )
        ledger = (
            episode["service"]
            if name == "hardware.csv.gz"
            else episode["recovery"]["ledger"]
        )
        equal(len(current), ledger["steps"], atol=0, rtol=0)
        equal(
            np.dot(current.p_grid_import_mw * current.lmp, current.duration_hours),
            ledger["electricity_usd"],
            atol=1e-6,
        )
        equal(
            np.dot(np.abs(current.bess_actual_kw) / 1000, current.duration_hours),
            ledger["battery_throughput_mwh"],
        )
        conservation = (
            ledger["initial_backlog_kwh"]
            + current.work_arrived_kwh.sum()
            - current.work_served_kwh.sum()
            - current.backlog_work_kwh.iloc[-1]
        )
        if abs(conservation) >= 1e-4:
            raise AssertionError("Raw work conservation failed")
        checks += 1
    for column in (
        "bid_mw",
        "bid_price",
        "committed_mw",
        "clearing_price",
        "rate",
        "residual_mw",
    ):
        equal(
            macros[column],
            reference["macro.csv.gz"][column].iloc[: len(macros)],
            atol=0,
            rtol=0,
        )
    if not np.array_equal(
        macros.bid_accepted, reference["macro.csv.gz"].bid_accepted.iloc[: len(macros)]
    ):
        raise AssertionError("Admission draws changed across fixed-macro policies")
    checks += 1
    start = 0
    for row in macros.itertuples():
        count = int(row.sub_steps_run)
        window = frame.iloc[start : start + count]
        error = float(np.abs(window.tracking_err_kw).mean())
        performance = max(0.0, 1.0 - error / (row.committed_mw * 1000))
        equal(performance, row.perf_score)
        equal(
            row.rate * row.committed_mw * performance * row.duration_hours,
            row.regulation_revenue,
        )
        start += count
    equal(start, len(frame), atol=0, rtol=0)
    delivered = float(
        np.dot(
            macros.committed_mw * macros.perf_score,
            macros.duration_hours,
        )
    )
    maximum = float(
        np.dot(
            np.where(macros.bid_accepted, macros.maximum_reference_offer_mw, 5.0),
            macros.duration_hours,
        )
    )
    equal(delivered / maximum, episode["delivered_service_ratio"])
    if not episode["complete"]:
        if episode["qualified"]:
            raise AssertionError("An incomplete episode was qualified")
        return checks
    recovered = frames["recovery.csv.gz"]
    equal(len(frame), 17280, atol=0, rtol=0)
    equal(len(recovered), 1440, atol=0, rtol=0)
    for phase in (frame, recovered):
        if phase.terminated.any() or not phase.truncated.iloc[-1]:
            raise AssertionError("A completed phase has inconsistent terminal flags")
        checks += 1
    equal(recovered.delta_p_demanded_kw, 0, atol=0, rtol=0)
    equal(recovered.committed_mw, 0, atol=0, rtol=0)
    revenue = float(macros.regulation_revenue.sum())
    cash = revenue - sum(
        float(
            np.dot(
                frames[name].p_grid_import_mw * frames[name].lmp,
                frames[name].duration_hours,
            )
        )
        for name in ("hardware.csv.gz", "recovery.csv.gz")
    )
    equal(cash, episode["cashflow_26h_usd"], atol=1e-6)
    equal(revenue, episode["regulation_revenue_usd"])
    equal(
        np.sqrt(np.mean(frame.tracking_err_kw.to_numpy() ** 2)),
        episode["service"]["tracking_rmse_kw"],
    )
    reference_load = frame.grid_load_mw.to_numpy()
    stress = reference_load >= np.quantile(reference_load, 0.9)
    equal(
        (frame.p_reference_mw - frame.p_facility_mw)[stress].mean(),
        episode["grid"]["stress_facility_relief_mw"],
    )
    equal(
        (frame.p_reference_mw - frame.p_grid_import_mw)[stress].mean(),
        episode["grid"]["stress_pcc_relief_mw"],
    )
    for key, value in decomposition(frame).items():
        if isinstance(value, bool):
            if episode["decomposition"][key] != value:
                raise AssertionError(f"Changed decomposition: {key}")
            checks += 1
        else:
            equal(value, episode["decomposition"][key], atol=1e-6)
    gate = protocol["service_gate"]
    physical = bool(
        episode["recovery"]["complete"]
        and frame.bess_soc.iloc[-1] >= gate["min_terminal_soc"]
        and frame.backlog_work_kwh.iloc[-1]
        <= gate["max_backlog_fraction"] * frame.work_arrived_kwh.sum() + 1e-6
        and frame.thermal_warning.mean() <= gate["max_thermal_warning_fraction"]
        and abs(recovered.bess_soc.iloc[-1] - episode["initial"]["soc"])
        <= protocol["recovery"]["soc_tolerance"]
        and abs(recovered.backlog_work_kwh.iloc[-1])
        <= protocol["recovery"]["backlog_tolerance_kwh"]
        and all(
            abs(
                recovered[f"temp_{zone}"].iloc[-1]
                - recovered[f"reference_temp_{zone}"].iloc[-1]
            )
            <= protocol["recovery"]["temperature_tolerance_C"]
            for zone in ("A", "B")
        )
    )
    if episode["qualified"] != (
        physical and delivered / maximum >= protocol["service_floor"]
    ):
        raise AssertionError("Raw-derived qualification differs from the saved result")
    return checks + 1


def main() -> None:
    protocol = read_json(ROOT / "headroom_registration.json")
    verify(protocol)
    directory = ROOT / "headroom_development"
    status = read_json(directory / "status.json")
    if (
        status["runtime_failures"]
        or status["completed"] != protocol["planned_episodes"]
    ):
        raise ValueError("The headroom run has runtime failures or missing cases")
    references, controls = {}, []
    for artifact in protocol["frozen_controls"]:
        episode = read_json(artifact["path"])
        controls.append(episode)
        if episode["policy"]["family"] == "cooling_first":
            references[case_key(episode["cell"])] = load_frames(
                Path(artifact["path"]).parent,
                episode,
            )
    episodes, check_count = [], 0
    for policy in protocol["policies"]:
        for cell in protocol["development_cells"]:
            path = directory / policy["id"] / case_key(cell)
            episode = read_json(path / "episode.json")
            if (
                episode["study_registration_id"] != protocol["registration_id"]
                or episode["policy"] != policy
                or episode["cell"] != cell
                or episode["source_hashes"] != protocol["source_hashes"]
            ):
                raise ValueError("Episode provenance or condition changed")
            check_count += audit(
                episode,
                load_frames(path, episode),
                references[case_key(cell)],
                protocol,
            )
            episodes.append(episode)
    frozen = aggregate(controls, protocol["development_cells"])
    candidates = aggregate(episodes, protocol["development_cells"])
    eligible = {name: row for name, row in frozen.items() if row["all_qualified"]}
    if not eligible:
        raise ValueError("No fully qualified frozen reference family")
    envelope = {}
    for stratum in ("all", "0.0", "0.5"):
        envelope[stratum] = {
            metric: max(
                row["strata"][stratum]["metrics"][metric]["mean"]
                for row in eligible.values()
            )
            for metric in (
                "stress_facility_relief_mw",
                "stress_pcc_relief_mw",
                "cashflow_26h_usd",
            )
        }
    decisions = {}
    for policy in protocol["policies"]:
        row = candidates[policy["family"]]
        contrasts = {}
        if row["complete_cohort"]:
            for stratum in ("all", "0.0", "0.5"):
                contrasts[stratum] = {
                    metric: row["strata"][stratum]["metrics"][metric]["mean"] - value
                    for metric, value in envelope[stratum].items()
                }
        promising = bool(
            row["all_qualified"]
            and policy["track"] == "economic_dispatch"
            and contrasts["all"]["cashflow_26h_usd"]
            >= protocol["comparison"]["cash_headroom_threshold_usd"]
            and all(contrasts[bias]["cashflow_26h_usd"] >= 0 for bias in ("0.0", "0.5"))
            and all(
                contrasts["0.5"][metric]
                >= protocol["comparison"]["relief_headroom_threshold_mw"]
                for metric in ("stress_facility_relief_mw", "stress_pcc_relief_mw")
            )
        )
        decisions[policy["id"]] = {
            "track": policy["track"],
            "contrasts_vs_frozen_envelope": contrasts,
            "physical_economic_headroom_demonstrated": promising,
            "not_a_learned_policy_promotion": True,
        }
    report = {
        "registration_id": protocol["registration_id"],
        "source_hashes": protocol["source_hashes"],
        "role": "consumed_development_only",
        "new_episodes": len(episodes),
        "frozen_reference_episodes": len(controls),
        "raw_scalar_or_array_checks": check_count,
        "qualified_new_episodes": sum(row["qualified"] for row in episodes),
        "frozen_families": frozen,
        "probe_families": candidates,
        "matched_reference_envelope": envelope,
        "decisions": decisions,
        "descriptive_sd_not_independent_standard_errors": True,
        "all_probes_are_nonlearned": True,
    }
    destination = ROOT / "headroom_report.json"
    if destination.exists():
        raise FileExistsError("Never overwrite an audited headroom report")
    write_json(destination, report)
    print(
        f"Audited {len(episodes)} new episodes and {len(controls)} frozen references; "
        f"{check_count} raw scalar/array checks."
    )
    for name, decision in decisions.items():
        print(name, decision)


if __name__ == "__main__":
    main()
