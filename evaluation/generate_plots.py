"""
evaluation/generate_plots.py  —  Publication-Ready Benchmark Plots
===================================================================
Reads evaluation/results.csv and produces a set of figures suitable for
a NeurIPS paper submission.

Figures produced
----------------
  fig_reward_heatmap.png       — mean_reward per (agent × scenario)
  fig_tracking_rmse.png        — tracking RMSE bar chart
  fig_thermal_violation.png    — thermal violation rate bar chart
  fig_throughput.png           — throughput ratio bar chart
  fig_survival_rate.png        — episode survival rate
  fig_radar_default.png        — radar chart, default scenario

Usage
-----
  cd .../C2G-Macro
  python evaluation/generate_plots.py                       # reads evaluation/results.csv
  python evaluation/generate_plots.py --csv path/to/results.csv --outdir figs/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering (cluster / CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from c2g_env.experiments.identity import configuration_id, group_conditions

# ── aesthetics ─────────────────────────────────────────────────────────────
SCENARIO_LABELS = {
    "default"    : "Default",
    "scenario_a" : "GenAI Crisis",
    "scenario_b" : "Thermal Squeeze",
    "scenario_c" : "Battery Drain",
}
AGENT_COLORS = {
    "random"     : "#aaaaaa",
    "rule_based" : "#4c72b0",
    "ppo"        : "#dd8452",
    "sac"        : "#55a868",
}
FIGSIZE_WIDE = (10, 4)
FIGSIZE_SQ   = (6, 5)
DPI          = 150


def _palette(agents: list[str]) -> list[str]:
    return [AGENT_COLORS.get(a, "#888888") for a in agents]


def _require_single_condition(df: pd.DataFrame) -> None:
    if len(group_conditions(df.to_dict(orient="records"))) > 1:
        raise ValueError("Multiple capacity/thermal conditions; select one condition or use generate_all().")


# ── individual plot helpers ─────────────────────────────────────────────────

def plot_metric_grouped_bar(
    df       : pd.DataFrame,
    metric   : str,
    ylabel   : str,
    title    : str,
    outpath  : Path,
    lower_is_better: bool = False,
    ylim     : tuple | None = None,
) -> None:
    _require_single_condition(df)
    scenarios = list(SCENARIO_LABELS.keys())
    scenarios = [s for s in scenarios if s in df["scenario"].unique()]
    agents    = sorted(df["agent"].unique(), key=lambda a: list(AGENT_COLORS).index(a)
                       if a in AGENT_COLORS else 99)

    x      = np.arange(len(scenarios))
    width  = 0.7 / max(len(agents), 1)
    offsets= np.linspace(-(len(agents)-1)*width/2, (len(agents)-1)*width/2, len(agents))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, agent in enumerate(agents):
        sub = df[df["agent"] == agent]
        vals = [
            float(sub[sub["scenario"] == sc][metric].mean())
            if sc in sub["scenario"].values else np.nan
            for sc in scenarios
        ]
        ax.bar(x + offsets[i], vals, width * 0.9,
               label=agent, color=AGENT_COLORS.get(agent, "#888888"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def plot_reward_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    _require_single_condition(df)
    scenarios = [s for s in SCENARIO_LABELS if s in df["scenario"].unique()]
    agents    = sorted(df["agent"].unique(), key=lambda a: list(AGENT_COLORS).index(a)
                       if a in AGENT_COLORS else 99)

    matrix = np.full((len(agents), len(scenarios)), np.nan)
    for i, agent in enumerate(agents):
        for j, sc in enumerate(scenarios):
            sub = df[(df["agent"] == agent) & (df["scenario"] == sc)]
            if not sub.empty:
                matrix[i, j] = float(sub["mean_reward"].mean())

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Mean step reward")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=9)
    ax.set_title("Mean Step Reward (agent × scenario)", fontsize=11, fontweight="bold")

    for i in range(len(agents)):
        for j in range(len(scenarios)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def plot_radar(df: pd.DataFrame, scenario: str, outpath: Path) -> None:
    """Spider / radar chart for a single scenario."""
    metrics = [
        ("mean_reward",       "Reward\n(norm)",       True),
        ("tracking_rmse",     "Tracking\nRMSE",       False),
        ("thermal_viol_rate", "Thermal Viol\nRate",   False),
        ("throughput_ratio",  "Throughput\nRatio",    True),
        ("survival_rate",     "Survival\nRate",       True),
    ]
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        return
    _require_single_condition(sub)

    agents = sorted(sub["agent"].unique(), key=lambda a: list(AGENT_COLORS).index(a)
                    if a in AGENT_COLORS else 99)
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    # Normalise each metric to [0, 1] across agents for this scenario
    raw = {m: [] for m, _, _ in metrics}
    for m, _, _ in metrics:
        raw[m] = [float(sub[sub["agent"] == a][m].mean())
                  if a in sub["agent"].values else 0.0 for a in agents]

    for m, _, higher_better in metrics:
        vals = raw[m]
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            raw[m] = [0.5] * len(agents)
        else:
            normed = [(v - vmin) / (vmax - vmin) for v in vals]
            raw[m] = normed if higher_better else [1 - v for v in normed]

    for i, agent in enumerate(agents):
        values = [raw[m][i] for m, _, _ in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=agent,
                color=AGENT_COLORS.get(agent, "#888888"))
        ax.fill(angles, values, alpha=0.08,
                color=AGENT_COLORS.get(agent, "#888888"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for _, lbl, _ in metrics], fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(f"Agent Comparison — {SCENARIO_LABELS.get(scenario, scenario)}",
                 fontsize=10, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


# ── main ────────────────────────────────────────────────────────────────────

def _generate_condition_plots(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plot_reward_heatmap(
        df, outdir / "fig_reward_heatmap.png"
    )
    plot_metric_grouped_bar(
        df, "mean_reward", "Mean Step Reward", "Mean Step Reward per Scenario",
        outdir / "fig_reward_bars.png",
    )
    plot_metric_grouped_bar(
        df, "tracking_rmse", "Tracking RMSE (kW)", "Grid-Following Tracking RMSE",
        outdir / "fig_tracking_rmse.png", lower_is_better=True,
    )
    plot_metric_grouped_bar(
        df, "thermal_viol_rate", "Thermal Violation Rate", "Thermal SLA Violation Rate",
        outdir / "fig_thermal_violation.png", lower_is_better=True, ylim=(0, 1),
    )
    plot_metric_grouped_bar(
        df, "throughput_ratio", "Throughput Ratio", "Batch Throughput Ratio",
        outdir / "fig_throughput.png", ylim=(0, 1),
    )
    if "survival_rate" in df.columns:
        plot_metric_grouped_bar(
            df, "survival_rate", "Survival Rate", "Episode Survival Rate (no thermal termination)",
            outdir / "fig_survival_rate.png", ylim=(0, 1),
        )

    # Radar for each scenario
    for sc in df["scenario"].unique():
        plot_radar(df, sc, outdir / f"fig_radar_{sc}.png")

    print("\nAll figures saved.")


def generate_all(csv_path: Path, outdir: Path) -> None:
    if not csv_path.exists():
        print(f"ERROR: Results CSV not found at {csv_path}")
        print("Run:  python evaluation/run_benchmark.py  first.")
        return

    df = pd.read_csv(csv_path)
    if "survival_rate" not in df.columns and "survived" in df.columns:
        df["survival_rate"] = df["survived"]

    for condition, rows in group_conditions(df.to_dict(orient="records")):
        target = outdir
        if condition:
            target = outdir / f"condition_{configuration_id(condition)[:12]}"
            target.mkdir(parents=True, exist_ok=True)
            (target / "condition.json").write_text(json.dumps(condition, indent=2), encoding="utf-8")
        print(f"Generating plots from {csv_path}: {target}/")
        _generate_condition_plots(pd.DataFrame(rows), target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate C2G-Bench benchmark plots")
    parser.add_argument("--csv",    default="evaluation/results.csv")
    parser.add_argument("--outdir", default="evaluation/figures")
    args = parser.parse_args()
    generate_all(Path(args.csv), Path(args.outdir))
