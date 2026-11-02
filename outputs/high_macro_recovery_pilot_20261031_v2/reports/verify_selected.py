"""Recompute selected policies and the strongest baseline directly from raw traces."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isort: off
from study_common import cell_key, file_hash, load_json, write_json  # noqa: E402
from register import verify_registration  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
# isort: on


def main():
    protocol = load_json(ROOT / "registration.json")
    verify_registration(protocol)
    selected = load_json(ROOT / "reports/selected_models.json")
    sources = [("full_fraction_baseline", ROOT / "baselines/fraction_1/summary.json")]
    for row in selected:
        if not row["available"]:
            continue
        choice = load_json(Path(row["model"]).parent / "selection.json")
        sources.append(
            (
                f"{row['variant']}_seed_{row['seed']}",
                Path(choice["source_validation"]),
            )
        )
    reference = {}
    records = []
    scalar_checks = 0

    def close(a, b):
        nonlocal scalar_checks
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6):
            raise AssertionError((a, b))
        scalar_checks += 1

    for policy_id, path in sources:
        summary = load_json(path)
        for episode in summary["cells"]:
            key = cell_key(episode["condition"])
            directory = path.parent / key
            for name, expected in episode["trace_hashes"].items():
                if file_hash(directory / name) != expected:
                    raise AssertionError(f"Trace changed: {directory / name}")
            service = pd.read_csv(
                directory / "hardware.csv.gz", float_precision="round_trip"
            )
            recovery = pd.read_csv(
                directory / "recovery.csv.gz", float_precision="round_trip"
            )
            macro = pd.read_csv(
                directory / "macro.csv.gz", float_precision="round_trip"
            )
            assert len(service) == 17280 and len(recovery) == 1440 and len(macro) == 96
            full = pd.concat([service, recovery], ignore_index=True)
            timestamps = np.asarray(full.timestamp, dtype="datetime64[ns]")
            assert (np.diff(timestamps) == np.timedelta64(5, "s")).all()
            assert not full.terminated.any()
            assert service.truncated.iloc[-1] and recovery.truncated.iloc[-1]
            assert (macro.bid_price == 0).all()
            assert (macro.bid_mw <= macro.residual_mw).all()
            assert (macro.bid_mw >= 5).all()
            close(math.fsum(service.duration_hours), 24)
            close(math.fsum(recovery.duration_hours), 2)
            close(math.fsum(macro.duration_hours), 24)
            cost = math.fsum(full.lmp * full.p_grid_import_mw * full.duration_hours)
            revenue = math.fsum(macro.regulation_revenue)
            close(revenue - cost, episode["cashflow_closed_recovery_usd"])
            close(revenue, episode["regulation_revenue_usd"])
            close(
                math.fsum(full.work_arrived_kwh - full.work_served_kwh),
                full.backlog_work_kwh.iloc[-1],
            )
            assert abs(recovery.bess_soc.iloc[-1] - episode["initial"]["soc"]) <= 1e-6
            for zone in ("A", "B"):
                assert (
                    abs(
                        recovery[f"temp_{zone}"].iloc[-1]
                        - recovery[f"reference_temp_{zone}"].iloc[-1]
                    )
                    <= 0.1
                )
            maximum = []
            for row in macro.itertuples():
                native = np.float32(min(30, row.residual_mw) / 30)
                if float(native) * 30 > min(30, row.residual_mw):
                    native = np.nextafter(native, np.float32(0))
                capacity = float(native) * 30 if row.bid_accepted else 5
                maximum.append(capacity * row.duration_hours)
            numerator = math.fsum(
                macro.committed_mw * macro.perf_score * macro.duration_hours
            )
            ratio = numerator / math.fsum(maximum)
            close(ratio, episode["delivered_service_ratio"])
            assert ratio >= 0.9
            assert service.bess_soc.iloc[-1] >= 0.2
            assert service.thermal_warning.mean() <= 0.01
            assert (
                service.backlog_work_kwh.iloc[-1]
                <= 0.01 * math.fsum(service.work_arrived_kwh) + 1e-6
            )
            signature = hashlib.sha256()
            for column in (
                "p_facility_mw",
                "p_grid_import_mw",
                "bess_soc",
                "temp_A",
                "temp_B",
                "backlog_work_kwh",
            ):
                signature.update(full[column].to_numpy(dtype=np.float64).tobytes())
            digest = signature.hexdigest()
            if policy_id == "full_fraction_baseline":
                reference[key] = digest
            records.append(
                {
                    "policy_id": policy_id,
                    "cell": key,
                    "cashflow_closed_recovery_usd": revenue - cost,
                    "delivered_service_ratio": ratio,
                    "quantity_fraction_mean": float(macro.quantity_fraction.mean()),
                    "quantity_fraction_min": float(macro.quantity_fraction.min()),
                    "quantity_fraction_max": float(macro.quantity_fraction.max()),
                    "physical_trajectory_matches_full_fraction_baseline": digest
                    == reference[key],
                }
            )
    summaries = []
    for policy_id, _ in sources:
        rows = [row for row in records if row["policy_id"] == policy_id]
        summaries.append(
            {
                "policy_id": policy_id,
                "verified_cases": len(rows),
                "physical_trajectories_identical_to_baseline": sum(
                    row["physical_trajectory_matches_full_fraction_baseline"]
                    for row in rows
                ),
                "mean_quantity_fraction": float(
                    np.mean([row["quantity_fraction_mean"] for row in rows])
                ),
            }
        )
    result = {
        "passed": True,
        "registered_study": protocol["registration_id"],
        "raw_26h_episodes": len(records),
        "independent_scalar_checks": scalar_checks,
        "all_selected_policy_cells_qualified": True,
        "policy_summaries": summaries,
        "scope": "Exact on these development cells; not global equivalence.",
        "sealed_policy_evaluations": 0,
        "auditor_sha256": file_hash(__file__),
    }
    verify_registration(protocol)
    write_json(ROOT / "reports/selected_raw_verification.json", result)
    pd.DataFrame(records).to_csv(
        ROOT / "reports/selected_raw_verification.csv", index=False
    )
    print(result)


if __name__ == "__main__":
    main()
