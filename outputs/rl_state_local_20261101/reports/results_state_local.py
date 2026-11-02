"""Read-only derivation of learning and raw physical evidence for the new study."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# isort: off
from study import ROOT, BASELINE, SOURCE, cap_cpu, file_hash, read_json, write_json
from common import case_key

sys.path.insert(0, str(BASELINE / "reports"))
from recompute_comparators import read_frame, verify_and_derive
from analyze_final import capacity_floor, verify_prefix
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
# isort: on


def audit_episode(arguments):
    directory, episode, median, protocol = arguments
    directory = Path(directory)
    if not episode["complete"] or not episode["recovery"]["complete"]:
        return {
            "policy": episode["policy"],
            "cell": episode["cell"],
            **verify_prefix(directory, episode),
        }
    result = verify_and_derive(directory, episode, median)
    hardware = read_frame(directory / "hardware.csv.gz")
    recovery = read_frame(directory / "recovery.csv.gz")
    macros = read_frame(directory / "macro.csv.gz")
    gate = protocol["service_gate"]
    checks = {
        "complete": len(hardware) == 17280,
        "terminal_soc": hardware.bess_soc.iloc[-1] >= gate["min_terminal_soc"],
        "terminal_work": hardware.backlog_work_kwh.iloc[-1]
        <= (gate["max_backlog_fraction"] * hardware.work_arrived_kwh.sum() + 1e-6),
        "thermal_warnings": hardware.thermal_warning.mean()
        <= gate["max_thermal_warning_fraction"],
    }
    if checks != episode["service_checks"]:
        raise AssertionError("Raw service gates disagree with episode results")
    recovery_config = protocol["recovery"]
    final = recovery.iloc[-1]
    closed = bool(
        abs(final.bess_soc - episode["initial"]["soc"])
        <= recovery_config["soc_tolerance"]
        and abs(final.backlog_work_kwh) <= recovery_config["backlog_tolerance_kwh"]
        and abs(final.temp_A - final.reference_temp_A)
        <= recovery_config["temperature_tolerance_C"]
        and abs(final.temp_B - final.reference_temp_B)
        <= recovery_config["temperature_tolerance_C"]
    )
    if closed != episode["recovery"]["closed"]:
        raise AssertionError("Raw recovery closure disagrees with the episode")
    combined = pd.concat([hardware, recovery])
    conservation = (
        combined.work_arrived_kwh.sum()
        - combined.work_served_kwh.sum()
        - final.backlog_work_kwh
        + episode["initial"]["backlog_kwh"]
    )
    if abs(conservation) >= 1e-4:
        raise AssertionError("Raw combined work conservation failed")
    if episode["qualified"] != bool(all(checks.values()) and closed):
        raise AssertionError("Raw qualification differs from the reported result")
    observed_means = (
        hardware.observed_regd_signal.to_numpy().reshape(96, 180).mean(axis=1)
    )
    np.testing.assert_allclose(
        macros.signed_regd_mean, observed_means, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        capacity_floor(macros), episode["delivered_service_ratio"], atol=1e-12, rtol=0
    )
    if episode["policy"]["macro"] != "rule":
        if not (
            (macros.bid_price == 0)
            & (macros.bid_mw >= 5)
            & (macros.bid_mw <= macros.residual_mw)
        ).all():
            raise AssertionError("Price-taking capacity band violated in raw data")
    result.update(
        policy=episode["policy"],
        cell=episode["cell"],
        complete_26h=True,
        delivered_service_ratio=episode["delivered_service_ratio"],
        delivered_service_gate=episode["delivered_service_gate"],
        raw_work_conservation_kwh=float(conservation),
        checks=result["checks"] + 7,
    )
    if episode["policy"]["macro"] == "forecast_quantity":
        threshold = episode["policy"]["margin_threshold_usd_mw_h"]
        negative = macros.estimated_marginal_cash_usd_mw_h < -threshold
        result["negative_forecast_windows"] = int(negative.sum())
        result["negative_windows_blocked_by_credit"] = int(
            (negative & ~macros.sufficient_worst_interval_credit).sum()
        )
        result["quantity_reduced_windows"] = int(macros.quantity_reduced.sum())
    return result


def init_worker(queue):
    cap_cpu(queue.get())


def audit_panel(panel, output, cpus):
    if output.exists():
        raise FileExistsError("Raw audit records are immutable")
    summary = read_json(panel / "summary.json")
    manifest = read_json(panel / "manifest.json")
    protocol = read_json(ROOT / "registration.json")
    if summary["runtime_failures"]:
        raise ValueError("Runtime failures require adjudication, not silent filtering")
    expected = {
        (policy["id"], case_key(cell))
        for policy in manifest["policies"]
        for cell in manifest["cells"]
    }
    observed = [
        (row["policy"]["id"], case_key(row["cell"])) for row in summary["episodes"]
    ]
    if set(observed) != expected or len(observed) != len(expected):
        raise AssertionError("Audit panel has missing or duplicated policy cases")
    load_path = SOURCE / "data/processed/energy/NYC.csv"
    median = float(pd.read_csv(load_path)["Load"].median())
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    arguments = [
        (
            panel / row["policy"]["id"] / case_key(row["cell"]),
            row,
            median,
            protocol,
        )
        for row in summary["episodes"]
    ]
    with ProcessPoolExecutor(
        max_workers=len(cpus),
        mp_context=context,
        initializer=init_worker,
        initargs=(queue,),
    ) as pool:
        rows = list(pool.map(audit_episode, arguments))
    queue.close()
    queue.join_thread()
    result = {
        "registration_id": protocol["registration_id"],
        "panel": str(panel),
        "episodes": len(rows),
        "scalar_or_array_checks": sum(row["checks"] for row in rows),
        "source_summary_sha256": file_hash(panel / "summary.json"),
        "auditor_sha256": file_hash(__file__),
        "reused_verifier_sha256": file_hash(
            BASELINE / "reports/recompute_comparators.py"
        ),
        "new_rollouts": 0,
        "rows": rows,
    }
    write_json(output, result)
    print({key: value for key, value in result.items() if key != "rows"}, flush=True)


def learning(stage, output, study_root=ROOT):
    protocol = read_json(study_root / "registration.json")
    reference_path = (
        Path(protocol["frozen_reference_path"])
        if "frozen_reference_path" in protocol
        else study_root / "development_references.json"
    )
    references = read_json(reference_path)["per_initializer"]
    colors = dict(zip(protocol["variants"], ("#cc5b33", "#2479ad"), strict=True))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    trajectories, groups, hashes = [], [], {}
    for variant in protocol["variants"]:
        by_seed = {}
        for seed in protocol["seeds"]:
            path = study_root / f"runs/{stage}/{variant}_{seed}/validation_summary.json"
            if not path.exists():
                by_seed[seed] = []
                continue
            rows = read_json(path)
            hashes[str(path)] = file_hash(path)
            by_seed[seed] = rows
            x = [0] + [row["timesteps"] / 1000 for row in rows]
            y = [references[str(seed)]["bc"]["rmse_kw"]] + [
                row["rmse_kw"] if row["rmse_kw"] is not None else np.nan for row in rows
            ]
            axes[0].plot(x, y, color=colors[variant], alpha=0.25, linewidth=1)
            axes[1].plot(
                x,
                np.array(y) / references[str(seed)]["bc"]["rmse_kw"],
                color=colors[variant],
                alpha=0.5,
                linewidth=1,
            )
            axes[2].plot(
                x[1:],
                [
                    row["diagnostics"]["fixed_demonstration_states"][
                        "native_local_std_mean"
                    ][1]
                    * 50000
                    for row in rows
                ],
                color=colors[variant],
                alpha=0.5,
                linewidth=1,
            )
            for row in rows:
                trajectories.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        **{
                            key: row[key]
                            for key in (
                                "timesteps",
                                "rmse_kw",
                                "all_qualified",
                                "qualified_cells",
                                "rmse_ratio_to_bc",
                                "rmse_ratio_to_guided",
                                "increment_vs_bc",
                                "extension_qualified",
                            )
                        },
                    }
                )
        completed_steps, means = [], []
        for step in protocol["stage_milestones"][stage]:
            current = [
                row
                for seed_rows in by_seed.values()
                for row in seed_rows
                if row["timesteps"] == step
            ]
            full = len(current) == 4 and all(
                row["rmse_kw"] is not None for row in current
            )
            mean = float(np.mean([row["rmse_kw"] for row in current])) if full else None
            groups.append(
                {
                    "variant": variant,
                    "timesteps": step,
                    "observed_fits": len(current),
                    "all_four_mean_rmse_kw": mean,
                    "qualified_fits": sum(row["all_qualified"] for row in current),
                    "extension_qualified_fits": sum(
                        row["extension_qualified"] for row in current
                    ),
                    "no_survivor_only_mean": True,
                }
            )
            if full:
                completed_steps.append(step / 1000)
                means.append(mean)
        axes[0].plot(
            [0] + completed_steps,
            [np.mean([value["bc"]["rmse_kw"] for value in references.values()])]
            + means,
            color=colors[variant],
            marker="o",
            label=variant.replace("_", " "),
        )
    for family, label, linestyle in (
        ("bc", "Exact BC initializer (four-seed mean)", ":"),
        ("guided", "Published Guided SAC (frozen selection)", "--"),
    ):
        axes[0].axhline(
            np.mean([value[family]["rmse_kw"] for value in references.values()]),
            color="black",
            linestyle=linestyle,
            label=label,
        )
    axes[1].axhline(1.0, color="black", linestyle=":")
    axes[1].axhline(1.25, color="gray", linestyle="--")
    for axis, title, ylabel in zip(
        axes,
        (
            "All 24 matched Dev-B cases",
            "Each fit versus its exact BC initializer",
            "HVAC exploration retained",
        ),
        (
            "Mean case tracking RMSE (kW)",
            "RMSE / BC RMSE",
            "Local action SD (nominal kW)",
        ),
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xlabel("Executed online transitions (thousands)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=170)
    plt.close(figure)
    write_json(
        output.with_suffix(".json"),
        {
            "registration_id": protocol["registration_id"],
            "stage": stage,
            "scope": "Development, both biases; not the beta0.5-only paper table",
            "all_attempts_remain_in_inventory": True,
            "trajectory_hashes": hashes,
            "trajectories": trajectories,
            "aggregates": groups,
            "analysis_sha256": file_hash(__file__),
        },
    )
    print(groups, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--panel", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--cpus", type=int, nargs="+", required=True)
    curve = sub.add_parser("learning")
    curve.add_argument("--stage", choices=("pilot", "extended"), required=True)
    curve.add_argument("--output", type=Path, required=True)
    curve.add_argument("--study-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "audit":
        audit_panel(args.panel, args.output, args.cpus)
    else:
        learning(args.stage, args.output, args.study_root)
