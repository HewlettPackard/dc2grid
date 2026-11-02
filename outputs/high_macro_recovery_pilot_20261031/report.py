"""All-attempt development learning readback; never evaluate the sealed panel."""

from __future__ import annotations

from pathlib import Path

# isort: off
from study_common import ROOT, cell_key, load_json, write_json
from register import verify_registration
import numpy as np
import pandas as pd
# isort: on


def full_mean(rows: list[dict], key: str, count: int):
    if len(rows) != count or any(row.get(key) is None for row in rows):
        return None
    values = [row[key] for row in rows]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite report metric: {key}")
    return float(np.mean(values))


def main():
    protocol = load_json(ROOT / "registration.json")
    verify_registration(protocol)
    baseline_summary = load_json(ROOT / "baselines/summary.json")
    baseline_rows = baseline_summary["policies"]
    passing = [row for row in baseline_rows if row["all_qualified"]]
    benchmark = max(passing, key=lambda row: row["cashflow"]) if passing else None
    reference = None
    if benchmark:
        reference = {
            cell_key(row["condition"]): row
            for row in load_json(Path(benchmark["summary"]))["cells"]
        }
    learning, selected, conditions = [], [], []
    attempted = []
    for variant in protocol["variants"]:
        for seed in protocol["macro_seeds"]:
            run = ROOT / "runs" / f"{variant}_seed_{seed}"
            if not (run / "status.json").exists():
                selected.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "available": False,
                        "reason": "No completed learner status",
                    }
                )
                continue
            status = load_json(run / "status.json")
            attempted.append(status)
            for step in [0, *protocol["milestones"]]:
                path = run / f"validation/step_{step}/summary.json"
                if not path.exists():
                    learning.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "macro_transitions": step,
                            "status": "unavailable",
                        }
                    )
                    continue
                summary = load_json(path)
                learning.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "macro_transitions": step,
                        "status": "recorded",
                        "kind": summary["kind"],
                        "qualified_cells": summary["qualified_cells"],
                        "planned_cells": summary["planned_cells"],
                        "mean_cashflow": summary["mean_cashflow_closed_recovery_usd"],
                        "mean_service_ratio": summary["mean_delivered_service_ratio"],
                        "cash_delta_vs_strongest_qualified_baseline": (
                            summary["mean_cashflow_closed_recovery_usd"]
                            - benchmark["cashflow"]
                            if benchmark
                            and summary["mean_cashflow_closed_recovery_usd"] is not None
                            else None
                        ),
                    }
                )
            choice_path = run / "best_model/selection.json"
            if not choice_path.exists():
                selected.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "available": False,
                        "reason": (
                            "No fully qualified RL checkpoint; "
                            "no imitation/final fallback"
                        ),
                    }
                )
                continue
            choice = load_json(choice_path)
            summary = load_json(Path(choice["source_validation"]))
            selected.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "available": True,
                    "macro_transitions": choice["macro_transitions"],
                    "cashflow": choice["score"],
                    "cash_delta_vs_strongest_qualified_baseline": (
                        choice["score"] - benchmark["cashflow"] if benchmark else None
                    ),
                    "model": str(run / "best_model/model.zip"),
                    "model_sha256": choice["checkpoint_sha256"],
                }
            )
            for episode in summary["cells"]:
                key = cell_key(episode["condition"])
                row = {
                    "variant": variant,
                    "seed": seed,
                    "cell": key,
                    "start_day": episode["condition"]["start_day"],
                    "bias": episode["condition"]["regd_load_bias"],
                    "qualified": episode["qualified"],
                    "cashflow": episode["cashflow_closed_recovery_usd"],
                    "service_ratio": episode["delivered_service_ratio"],
                    "tracking_rmse_kw": episode["service"]["tracking_rmse_kw"],
                    "terminal_service_soc": episode["service_terminal"]["soc"],
                    "cash_delta_vs_baseline": (
                        episode["cashflow_closed_recovery_usd"]
                        - reference[key]["cashflow_closed_recovery_usd"]
                        if reference
                        and reference[key]["cashflow_closed_recovery_usd"] is not None
                        else None
                    ),
                }
                directory = Path(choice["source_validation"]).parent / key
                frame = pd.read_csv(
                    directory / "hardware.csv.gz", float_precision="round_trip"
                )
                if episode["service_complete"]:
                    load = frame.grid_load_mw.to_numpy()
                    nominal = frame.p_reference_mw.to_numpy()
                    actual = frame.p_facility_mw.to_numpy()
                    stress = load >= np.quantile(load, 0.9)
                    row.update(
                        stress_facility_relief_mw=float(
                            (nominal - actual)[stress].mean()
                        ),
                        actual_modeled_facility_boundary_peak_reduction_mw=float(
                            np.max(load + nominal) - np.max(load + actual)
                        ),
                    )
                conditions.append(row)
    arms = []
    for variant in protocol["variants"]:
        rows = [row for row in selected if row["variant"] == variant]
        arms.append(
            {
                "variant": variant,
                "planned_training_seeds": len(protocol["macro_seeds"]),
                "qualified_selected_models": sum(row["available"] for row in rows),
                "mean_selected_cashflow": full_mean(
                    rows, "cashflow", len(protocol["macro_seeds"])
                ),
                "mean_cash_delta_vs_strongest_baseline": full_mean(
                    rows,
                    "cash_delta_vs_strongest_qualified_baseline",
                    len(protocol["macro_seeds"]),
                ),
            }
        )
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    for filename, rows in (
        ("learning_progress", learning),
        ("selected_models", selected),
        ("selected_conditions", conditions),
        ("arm_summary", arms),
    ):
        write_json(output / f"{filename}.json", rows)
        pd.DataFrame(rows).to_csv(output / f"{filename}.csv", index=False)
    readback = {
        "registration_id": protocol["registration_id"],
        "training_gate": load_json(ROOT / "training_gate.json"),
        "planned_training_runs": len(protocol["variants"])
        * len(protocol["macro_seeds"]),
        "recorded_learner_statuses": len(attempted),
        "completed_learners": sum(row["state"] == "completed" for row in attempted),
        "macro_transitions": sum(row["macro_transitions"] for row in attempted),
        "normal_sac_updates": sum(row["normal_sac_updates"] for row in attempted),
        "critic_only_updates": sum(row["calibration_updates"] for row in attempted),
        "baseline_family": baseline_rows,
        "strongest_qualified_baseline": benchmark,
        "arms": arms,
        "sealed_test_policy_rollouts": 0,
        "interpretation": [
            "Development pilot only; the six-date sealed panel is not executed.",
            "Initial imitation/warmup validate the pipeline, not RL superiority.",
            "Four independent macro seeds per arm share one frozen hardware policy.",
            "Missing selected seeds prevent a full-arm mean; no final-weight fallback.",
            "Both bias conditions and all initial/failed evaluations remain explicit.",
            "Stress-average relief is not actual maximum-load reduction.",
            "Closed cash excludes wear/SLA valuation and is not total business profit.",
            "Market/constant-quantity comparators are not robust physical MPC.",
        ],
    }
    write_json(output / "readback.json", readback)
    print("Saved all-attempt development readback:", output)


if __name__ == "__main__":
    main()
