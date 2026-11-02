"""Read back every registered final-budget outcome without survivor filtering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = str(PILOT / "reports/plot-cache")
os.sched_setaffinity(0, {28})
sys.path.insert(0, str(PILOT / "source"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines.online_stability import verify_registration

ARMS = ["sham", "optimizer_reset", "critic_restore"]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    registration = json.loads((PILOT / "registration.json").read_text())
    verify_registration(registration)
    status = json.loads((PILOT / "status.json").read_text())
    assert status["counts"] == {"pending": 0, "running": 0, "completed": 36, "failed": 0}
    budget = registration["pilot_protocol"]["online_transitions"]
    expected_cells = {
        (c["start_day"], c["regd_load_bias"], c["seed"])
        for c in registration["environment_protocol"]["validation"]
    }
    run_rows, cell_rows, curve_rows, identities, fingerprints = [], [], [], [], {}
    for job in registration["jobs"]:
        directory = PILOT / "runs" / job["id"]
        state = json.loads((directory / "status.json").read_text())
        init = json.loads((directory / "initialization.json").read_text())
        final_path = directory / "validation" / f"online_{budget}.json"
        final = json.loads(final_path.read_text())
        assert state["state"] == "completed"
        assert state["new_online_transitions"] == budget
        assert state["new_normal_sac_updates"] == budget
        assert state["actual_replay_position"] == state["expected_replay_position"]
        assert final["new_normal_sac_updates"] == budget
        assert final["all_cells_complete"] and final["attempts"] == 4
        assert len(final["cells"]) == 4
        prefix = {"id": job["id"], "base_seed": job["base_seed"], "stream": job["stream"], "arm": job["arm"]}
        run_rows.append({
            **prefix, "new_transitions": budget, "normal_sac_updates": budget,
            "all_qualified": final["all_cells_qualified"],
            "qualified_cells": final["qualified_count"],
            "rmse_kw": final["mean_rmse_kw"], "nrmse": final["mean_normalized_rmse"],
            "reward": final["selection_score"],
        })
        conditions = {
            (c["condition"]["start_day"], c["condition"]["regd_load_bias"], c["condition"]["seed"])
            for c in final["cells"]
        }
        assert conditions == expected_cells
        for cell in final["cells"]:
            assert cell["complete"] and cell["fast_steps"] == 17280
            assert abs(cell["work_conservation_error_kwh"]) < 1e-4
            condition = cell["condition"]
            reasons = []
            gate = registration["environment_protocol"]["selection"]
            if cell["terminal_soc"] < gate["min_terminal_soc"]:
                reasons.append("terminal_soc")
            if cell["thermal_warning_fraction"] > gate["max_thermal_warning_fraction"]:
                reasons.append("thermal_warning")
            if cell["terminal_backlog_kwh"] > gate["max_backlog_fraction"] * cell["work_arrived_kwh"] + 1e-6:
                reasons.append("backlog")
            assert bool(reasons) != cell["qualified"]
            cell_rows.append({
                **prefix, "day": condition["start_day"], "bias": condition["regd_load_bias"],
                "tape_seed": condition["seed"], "failure_reasons": "|".join(reasons),
                **{key: value for key, value in cell.items() if key != "condition"},
            })
        for path in sorted((directory / "validation").glob("online_*.json")):
            report = json.loads(path.read_text())
            curve_rows.append({
                **prefix, "new_transitions": report["new_online_transitions"],
                "normal_sac_updates": report["new_normal_sac_updates"],
                "all_complete": report["all_cells_complete"],
                "all_qualified": report["all_cells_qualified"],
                "qualified_cells": report["qualified_count"],
                "rmse_kw": report["mean_rmse_kw"], "nrmse": report["mean_normalized_rmse"],
                "reward": report["selection_score"],
            })
        identities.append((job["base_seed"], job["stream"], init["unchanged_actor_entropy"]))
        fingerprints[job["id"]] = {
            "final_model_sha256": digest(directory / "final_model.zip"),
            "final_validation_sha256": digest(final_path),
        }
    for seed in registration["pilot_protocol"]["base_seeds"]:
        for stream in registration["pilot_protocol"]["continuation_streams"]:
            group = [identity for s, r, identity in identities if (s, r) == (seed, stream)]
            assert len(group) == 3 and group[0] == group[1] == group[2]

    runs = pd.DataFrame(run_rows)
    cells = pd.DataFrame(cell_rows)
    curves = pd.DataFrame(curve_rows)
    assert len(runs) == 36 and len(cells) == 144
    assert runs.groupby(["base_seed", "stream"]).size().eq(3).all()
    for _, row in runs.iterrows():
        group = cells[cells.id == row.id]
        assert math.isclose(row.rmse_kw, group.tracking_rmse_kw.mean(), abs_tol=1e-8)
        assert math.isclose(row.nrmse, group.normalized_rmse.mean(), abs_tol=1e-12)
        assert int(row.qualified_cells) == int(group.qualified.sum())
    summary = runs.groupby("arm").agg(
        runs=("id", "size"), rmse_kw=("rmse_kw", "mean"), nrmse=("nrmse", "mean"),
        reward=("reward", "mean"), fully_qualified_models=("all_qualified", "sum"),
        qualified_cells=("qualified_cells", "sum"),
    ).reindex(ARMS)
    per_parent = runs.groupby(["base_seed", "arm"]).agg(
        streams=("id", "size"), rmse_kw=("rmse_kw", "mean"),
        nrmse=("nrmse", "mean"), reward=("reward", "mean"),
        fully_qualified_models=("all_qualified", "sum"), qualified_cells=("qualified_cells", "sum"),
    )
    biased = cells.groupby(["arm", "bias"]).agg(
        cells=("id", "size"), rmse_kw=("tracking_rmse_kw", "mean"),
        nrmse=("normalized_rmse", "mean"), qualified_cells=("qualified", "sum"),
        mean_soc=("terminal_soc", "mean"),
        maximum_warning_fraction=("thermal_warning_fraction", "max"),
    )
    paired = runs.pivot(index=["base_seed", "stream"], columns="arm", values=["rmse_kw", "nrmse", "reward"])
    paired_results = pd.DataFrame({
        "restore_vs_sham_rmse_ratio": paired["rmse_kw"]["critic_restore"] / paired["rmse_kw"]["sham"],
        "restore_vs_sham_nrmse_ratio": paired["nrmse"]["critic_restore"] / paired["nrmse"]["sham"],
        "restore_vs_sham_reward_change": paired["reward"]["critic_restore"] - paired["reward"]["sham"],
        "optimizer_vs_sham_rmse_ratio": paired["rmse_kw"]["optimizer_reset"] / paired["rmse_kw"]["sham"],
    })
    output = PILOT / "reports"
    runs.to_csv(output / "final_runs.csv", index=False)
    cells.to_csv(output / "final_cells.csv", index=False)
    curves.to_csv(output / "validation_curves.csv", index=False)
    summary.to_csv(output / "final_summary.csv")
    per_parent.to_csv(output / "final_by_parent.csv")
    biased.to_csv(output / "final_by_bias.csv")
    paired_results.to_csv(output / "paired_continuations.csv")

    print("ALL REGISTERED FINAL-BUDGET RUNS:")
    print(summary.to_string())
    print("\nPER PARENT CHECKPOINT:")
    print(per_parent.to_string())
    print("\nBY BIAS:")
    print(biased.to_string())
    print("\nFAILED QUALIFICATION REASONS:")
    print(cells[~cells.qualified].groupby(["base_seed", "arm", "failure_reasons"]).size().to_string())
    print("\nPAIRED CONTINUATIONS (not independent parent seeds):")
    print(paired_results.to_string())

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = ["#8c8c8c", "#fdbb84", "#2c7fb8"]
    labels = ["Sham", "Optimizer\nreset", "Critic/target\nrestoration"]
    for axis, seed in zip(axes.flat, registration["pilot_protocol"]["base_seeds"]):
        values = runs[runs.base_seed == seed]
        means = [values[values.arm == arm].rmse_kw.mean()/1000 for arm in ARMS]
        bars = axis.bar(np.arange(3), means, color=colors, alpha=.8)
        for index, arm in enumerate(ARMS):
            group = values[values.arm == arm].sort_values("stream")
            axis.scatter(index + np.array([-.06, 0, .06]), group.rmse_kw/1000, color="black", s=23, zorder=3)
            axis.annotate(
                f"{int(group.all_qualified.sum())}/3 fully qualified",
                (index, max(group.rmse_kw)/1000),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
            )
        axis.set_xticks(np.arange(3), labels)
        axis.set(title=f"Parent checkpoint {seed}", ylabel="Mean episode tracking RMSE (MW)",
                 ylim=(0, values.rmse_kw.max()/1000*1.25))
    fig.suptitle(
        "Completed ONLINE stability pilot: final-budget outcomes\n"
        "Bars = stream means; dots = three nested continuations; all cells retained",
        fontsize=13,
    )
    fig.savefig(output / "online_pilot_outcomes.png", dpi=170)
    plt.close(fig)

    receipt = {
        "schema": "online-stability-pilot-final-readback-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registration_id": registration["registration_id"],
        "all_source_and_protocol_hashes_verified": True,
        "registered_runs": len(runs), "runtime_failures": 0,
        "new_online_transitions": int(runs.new_transitions.sum()),
        "normal_sac_updates": int(runs.normal_sac_updates.sum()),
        "additional_critic_calibration_updates": len(runs) * registration["pilot_protocol"]["critic_calibration_steps"],
        "full_final_validation_cells": len(cells),
        "paired_initial_actor_entropy_identities_verified": True,
        "final_update_and_replay_insertion_accounting_verified": True,
        "arm_summary": summary.reset_index().to_dict("records"),
        "parent_summary": per_parent.reset_index().to_dict("records"),
        "paired_raw_rmse_improvements": int((paired_results.restore_vs_sham_rmse_ratio < 1).sum()),
        "paired_normalized_rmse_improvements": int((paired_results.restore_vs_sham_nrmse_ratio < 1).sum()),
        "pair_count": len(paired_results),
        "final_artifact_hashes": fingerprints,
        "limitations": [
            "Four selected parent checkpoints, with three nested continuation streams each; not12independent from-scratch seeds.",
            "Same fixed validation cells used for monitoring/selection; no heldout peak-day test.",
            "Primary outcomes are final-budget, including every registered run and unqualified cell.",
            "Recovery treatment restores critics and targets together and includes fresh Adam; not a random critic reset.",
            "All arms receive common500critic-onlycalibration, so sham is a matched-calibration control.",
            "Tracking improvement does not imply constraint dominance or Table2facility peak-relief improvement.",
            "Exploratory selected-case online continuation, not a population-wide safety guarantee.",
        ],
    }
    (output / "readback.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print("\nVerified completed pilot; outputs:", output)


if __name__ == "__main__":
    main()
