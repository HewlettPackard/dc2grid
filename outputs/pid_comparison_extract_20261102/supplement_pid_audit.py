"""Additional read-only metering/deferral checks; registered outcomes stay intact."""

from __future__ import annotations

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# isort: off
from pid_context import (
    CONFIRMATION,
    ROOT,
    cap_cpu,
    case_key,
    file_hash,
    initialize_worker,
    read_json,
    verify,
    write_json,
)
from audit_pid import frozen_inputs
from confirmation_audit import close
from c2g_env import C2GFastEnv
import numpy as np
import pandas as pd
# isort: on


def extra_case(artifact, floor_kw):
    path = Path(artifact["path"])
    if file_hash(path) != artifact["sha256"]:
        raise ValueError("A registered PID outcome changed")
    outcome = read_json(path)
    if not artifact["complete"]:
        raise ValueError(
            "Do not manufacture full-horizon diagnostics from partial cases"
        )
    for name in ("hardware.csv.gz", "macro.csv.gz"):
        if file_hash(path.parent / name) != outcome["trace_hashes"][name]:
            raise ValueError("A PID trace changed before supplemental audit")
    service = pd.read_csv(
        path.parent / "hardware.csv.gz",
        usecols=[
            "grid_load_mw",
            "throttle_batch",
            "flex_reduction_kw",
            "duration_hours",
            "backlog_work_kwh",
            "bess_soc",
            "tracking_err_kw",
        ],
    )
    market = pd.read_csv(path.parent / "macro.csv.gz")
    start = 0
    for row in market.itertuples():
        count = int(row.sub_steps_run)
        error = float(service.tracking_err_kw.iloc[start : start + count].abs().mean())
        denominator = max(row.committed_mw * 1000, floor_kw)
        if floor_kw > row.committed_mw * 1000:
            raise AssertionError("Inherited fee-audit denominator assumption fails")
        close(max(0, 1 - error / denominator), row.perf_score)
        start += count
    if start != 17280:
        raise AssertionError("Incomplete physical service trace")
    stress = service.grid_load_mw >= service.grid_load_mw.quantile(0.9)
    return {
        "panel": artifact["panel"],
        "policy_id": artifact["policy_id"],
        "cell": artifact["cell"],
        "qualified": artifact["qualified"],
        "stress_mean_throttle": float(service.loc[stress, "throttle_batch"].mean()),
        "minimum_throttle": float(service.throttle_batch.min()),
        "stress_mean_it_curtailment_mw": float(
            service.loc[stress, "flex_reduction_kw"].mean() / 1000,
        ),
        "positive_service_it_curtailment_mwh": float(
            np.dot(
                np.maximum(service.flex_reduction_kw, 0) / 1000,
                service.duration_hours,
            )
        ),
        "maximum_backlog_kwh": float(service.backlog_work_kwh.max()),
        "terminal_service_soc": float(service.bess_soc.iloc[-1]),
        "minimum_service_soc": float(service.bess_soc.min()),
        "recovery_soc_error": outcome["recovery"]["soc_error"],
        "fee_floor_checks": len(market),
    }


def main():
    cap_cpu(32)
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    cases_path = ROOT / "reports/pid_cases.json"
    report_path = ROOT / "reports/pid_report.json"
    before = {str(path): file_hash(path) for path in (cases_path, report_path)}
    cases = read_json(cases_path)
    references = frozen_inputs()
    for row in cases:
        expected = references[(row["panel"], case_key(row["cell"]))]
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise ValueError("Missing or invalid matched input fingerprint")
        if row["complete"] and row["inputs_sha256"] != expected:
            raise ValueError("PID input pairing was not actually verified")
    probe = C2GFastEnv(
        scenario="default",
        simulation_mode="physical_v1",
        config_path=protocol["config_path"],
    )
    try:
        probe.reset(seed=19000011)
        floor = float(probe._macro_tracking_floor_kw)
    finally:
        probe.close()
    original_floor = read_json(CONFIRMATION / "registration.json")[
        "macro_tracking_floor_kw"
    ]
    if floor != original_floor:
        raise ValueError("PID and confirmation fee floors differ")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    with ProcessPoolExecutor(
        max_workers=len(protocol["cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        details = list(pool.map(extra_case, cases, [floor] * len(cases)))
    queue.close()
    queue.join_thread()
    summary = {}
    for panel in protocol["panels"]:
        summary[panel] = {}
        for bias in (0.0, 0.5):
            summary[panel][str(bias)] = {}
            for policy in protocol["policies"]:
                selected = [
                    row
                    for row in details
                    if row["panel"] == panel
                    and row["cell"]["regd_load_bias"] == bias
                    and row["policy_id"] == policy["id"]
                ]
                summary[panel][str(bias)][policy["id"]] = {
                    "episodes": len(selected),
                    "qualified": sum(row["qualified"] for row in selected),
                    "stress_mean_throttle": float(
                        np.mean([row["stress_mean_throttle"] for row in selected])
                    ),
                    "stress_mean_it_curtailment_mw": float(
                        np.mean(
                            [row["stress_mean_it_curtailment_mw"] for row in selected]
                        )
                    ),
                    "positive_service_it_curtailment_mwh": float(
                        np.mean(
                            [
                                row["positive_service_it_curtailment_mwh"]
                                for row in selected
                            ]
                        )
                    ),
                    "worst_backlog_kwh": max(
                        row["maximum_backlog_kwh"] for row in selected
                    ),
                    "minimum_terminal_service_soc": min(
                        row["terminal_service_soc"] for row in selected
                    ),
                    "maximum_recovery_soc_error": max(
                        row["recovery_soc_error"] for row in selected
                    ),
                }
    if any(file_hash(path) != digest for path, digest in before.items()):
        raise AssertionError("Supplemental audit modified the registered results")
    result = {
        "registration_id": protocol["registration_id"],
        "role": "postprocessing_only_no_new_rollouts",
        "original_report_hashes": before,
        "source_sha256": file_hash(__file__),
        "macro_tracking_floor_kw_verified_from_runtime_config": floor,
        "all_complete_input_digests_nonnull_and_matched": True,
        "fee_floor_checks": sum(row["fee_floor_checks"] for row in details),
        "positive_it_curtailment_is_not_equated_with_deferred_work_energy": True,
        "summary": summary,
        "case_diagnostics": details,
    }
    target = ROOT / "reports/pid_audit_supplement.json"
    if target.exists():
        raise FileExistsError("Never overwrite the supplemental diagnostic report")
    write_json(target, result)
    print("Supplemental fee checks:", result["fee_floor_checks"], "floor:", floor)


if __name__ == "__main__":
    main()
