"""Plot all registered milestones, keeping failures and BC initialization distinct."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isort: off
from common import file_hash, read_json, write_json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
# isort: on


def main():
    protocol = read_json(ROOT / "registration.json")
    bc_rows = read_json(ROOT / "bc_development_v1/summary.json")["episodes"]
    bc = {
        seed: float(
            np.mean(
                [
                    r["service"]["tracking_rmse_kw"]
                    for r in bc_rows
                    if r["policy"]["id"] == f"history_{seed}"
                ]
            )
        )
        for seed in protocol["seeds"]
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.5), constrained_layout=True)
    colors = {"warm_sac": "#c65335", "guided_sac": "#2878a5"}
    names = {"warm_sac": "Warm SAC", "guided_sac": "Guided SAC"}
    records, aggregates, hashes = [], [], {}
    for variant in protocol["variants"]:
        trajectories = {}
        for seed in protocol["seeds"]:
            directory = ROOT / "runs" / f"{variant}_{seed}"
            path = directory / "validation_summary.json"
            trajectory = read_json(path)
            hashes[f"{variant}_{seed}"] = file_hash(path)
            trajectories[seed] = trajectory
            x = [0] + [row["timesteps"] / 1000 for row in trajectory]
            y = [bc[seed]] + [row.get("rmse_kw", np.nan) for row in trajectory]
            axes[0].plot(x, y, color=colors[variant], alpha=0.22, linewidth=1)
            selected = read_json(directory / "selection.json")["selected"]
            if selected is not None:
                axes[0].scatter(
                    [selected["timesteps"] / 1000],
                    [selected["rmse_kw"]],
                    marker="D",
                    s=38,
                    color=colors[variant],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=5,
                )
            for row in trajectory:
                records.append({"variant": variant, "seed": seed, **row})
            telemetry = pd.read_csv(directory / "metrics/progress.csv")
            telemetry = telemetry.dropna(
                subset=["train/n_updates", "train/native_local_std_hvac"]
            )
            online = (
                telemetry["train/n_updates"]
                - protocol["critic_calibration_updates"]
                + protocol["learning_starts"]
            ) / 1000
            axes[2].plot(
                online,
                telemetry["train/native_local_std_hvac"] * 50000,
                color=colors[variant],
                alpha=0.6,
                linewidth=1,
            )
        means, eligible = [], []
        for index, step in enumerate(protocol["milestones"]):
            current = [trajectories[seed][index] for seed in protocol["seeds"]]
            all_defined = all(row.get("rmse_kw") is not None for row in current)
            value = (
                float(np.mean([row["rmse_kw"] for row in current]))
                if all_defined
                else None
            )
            count = sum(row["eligible"] for row in current)
            means.append(np.nan if value is None else value)
            eligible.append(count)
            aggregates.append(
                {
                    "variant": variant,
                    "timesteps": step,
                    "mean_rmse_kw_all_four_seeds": value,
                    "eligible_seeds": count,
                    "all_four_seed_mean_defined": all_defined,
                }
            )
        axes[0].plot(
            [0] + [step / 1000 for step in protocol["milestones"]],
            [float(np.mean(list(bc.values())))] + means,
            color=colors[variant],
            marker="o",
            markersize=4,
            linewidth=2,
            label=names[variant],
        )
        axes[1].plot(
            np.array(protocol["milestones"]) / 1000,
            eligible,
            color=colors[variant],
            marker="o",
            linewidth=2,
            label=names[variant],
        )
    axes[0].axhline(
        np.mean(list(bc.values())),
        color="#4d7852",
        linestyle="--",
        label="BC-only initialization",
    )
    axes[0].axhline(
        448.3298678461515, color="#777777", linestyle=":", label="Frozen old SAC"
    )
    axes[0].axhline(637.5247489436841, color="#a08050", linestyle=":", label="PID")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean development RMSE (kW, log scale)")
    axes[0].set_title("All seeds and all milestones")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].annotate(
        "Diamonds: selected checkpoints\n"
        "Final guided mean undefined:\n1 incomplete evaluation case",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
    )
    axes[1].set_ylim(-0.15, 4.25)
    axes[1].set_yticks(range(5))
    axes[1].set_ylabel("Eligible checkpoints / 4 training seeds")
    axes[1].set_title("Registered physical and improvement gates")
    axes[1].legend(fontsize=8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Local HVAC exploration SD (kW)")
    axes[2].set_title("Exploration decreases in both arms")
    for axis in axes:
        axis.set_xlabel("New online transitions (thousands)")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "BC-initialized SAC: early retention, not convergent superiority over BC",
        fontsize=12,
    )
    figure.savefig(ROOT / "reports/learning_progress.png", dpi=170)
    plt.close(figure)
    write_json(
        ROOT / "reports/learning_progress.json",
        {
            "registration_id": protocol["registration_id"],
            "bc_only_rmse_kw_by_seed": bc,
            "step_zero_is_supervised_not_RL": True,
            "all_milestones": records,
            "aggregate": aggregates,
            "validation_hashes": hashes,
            "plotter_sha256": file_hash(__file__),
            "interpretation": (
                "No survivor-only mean; later incomplete guided checkpoint has no "
                "four-seed RMSE mean. Exploration SD is a local linearization, "
                "not measured tracking error or a confidence interval."
            ),
        },
    )


if __name__ == "__main__":
    main()
