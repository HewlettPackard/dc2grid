"""Publish the complete adaptive factorial, including failures and null results."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def draw_frontier(rows, result):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = next(row for row in rows if row["family"] == "cooling_first")
    references = [
        row for row in result["reference_families"].values() if row["all_qualified"]
    ]
    best_pcc = max(
        row["strata"]["0.5"]["metrics"]["stress_pcc_relief_mw"]["mean"]
        for row in references
    )
    best_cash = max(
        row["strata"]["0.5"]["metrics"]["cashflow_26h_usd"]["mean"]
        for row in references
    )
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, row in enumerate(rows):
        if row["cashflow_26h_usd"] is None or row["stress_pcc_relief_mw"] is None:
            continue
        candidate = row["family"].startswith(("fraction_", "potential_"))
        color = "#1565c0" if candidate else "#666666"
        axis.scatter(
            row["stress_pcc_relief_mw"],
            row["cashflow_26h_usd"] - baseline["cashflow_26h_usd"],
            s=85 if candidate else 55,
            marker="o" if row["all_qualified"] else "x",
            color=color,
            label=row["family"],
        )
        axis.annotate(
            str(index + 1),
            (
                row["stress_pcc_relief_mw"],
                row["cashflow_26h_usd"] - baseline["cashflow_26h_usd"],
            ),
            xytext=(5, 5 + (index % 2) * 8),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(best_pcc + 0.001, color="#999999", linestyle="--", linewidth=1)
    axis.axhline(
        best_cash + 5 - baseline["cashflow_26h_usd"],
        color="#999999",
        linestyle="--",
        linewidth=1,
    )
    axis.set(
        xlabel="Stress-time PCC relief (MW; higher is better)",
        ylabel="26-hour cash improvement over analytic control (USD)",
        title="Economic dispatch: consumed development cases, bias 0.5",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    destination = ROOT / "reports/economic_frontier.png"
    if destination.exists():
        raise FileExistsError("Do not replace a generated frontier figure")
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def main():
    study = ROOT / "track_b_fraction"
    result = read(study / "development_report.json")
    protocol = read(study / "registration.json")
    if (
        result["registration_id"] != protocol["registration_id"]
        or result["role"] != "adaptive_consumed_development_only"
        or not result["final_data_not_opened"]
    ):
        raise ValueError("Unexpected factorial report scope")
    completed_models = sum(
        row["model_count"] for row in result["learned_families"].values()
    )
    names = (
        "cooling_first",
        "constant_one",
        "greedy_budget",
        "margin_ten",
        "economic_ppo",
        "economic_sac",
        "fraction_ppo",
        "fraction_sac",
        "potential_ppo",
        "potential_sac",
    )
    families = {**result["reference_families"], **result["learned_families"]}
    metrics = (
        "stress_facility_relief_mw",
        "stress_pcc_relief_mw",
        "cashflow_26h_usd",
        "tracking_rmse_kw",
        "battery_throughput_26h_mwh",
    )
    rows = []
    for name in names:
        family = families[name]
        values = family["strata"].get("0.5", {}).get("metrics", {})
        rows.append(
            {
                "family": name,
                "model_count": family["model_count"],
                "all_qualified": family["all_qualified"],
                "minimum_service_ratio": family.get("minimum_service_ratio"),
                **{
                    key: values[key]["mean"] if key in values else None
                    for key in metrics
                },
            }
        )
    csv_path = ROOT / "reports/fraction_summary.csv"
    if csv_path.exists():
        raise FileExistsError("Do not overwrite a factorial summary")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Fractional economic-budget PPO/SAC follow-up",
        "",
        "**Adaptive development results, not the original Table 2 and not a",
        "fresh final test.** This follow-up was registered after observing V1",
        "on the same consumed Dev-B cells. All 16 fits and all four model seeds",
        "per family are retained; only mandatory final checkpoints are used.",
        "",
        "## What changed",
        "",
        "- The action chooses a fraction of an allowance frozen at the first",
        "  actual hardware tick after the contract handshake. The requested MW",
        "  stays fixed within that window; the original per-tick guard remains.",
        "- The second arm adds training-only observed-credit potential",
        "  `Phi = 10 USD/MWh * credit_MWh / 100 USD`, with gamma 1 and true-terminal",
        "  next-potential zero. It preserves complete and invalid episode returns.",
        "- The value 10 came from the previously tested heuristic, not discovery.",
        "- No physics, bidding, observation, work, SOC target or recovery change",
        "  is introduced. Tracking deviations remain an explicit economic trade-off.",
        "",
        "Four native smoke fits passed. For each algorithm, both reward arms had",
        "identical physical warmup states/actions/terminals; their reward",
        "difference matched the potential formula. The task/integration tests",
        "also verified 26-hour behavior and true-terminal telescoping.",
        "",
        "## Matched biased development results",
        "",
        "All rows use the existing full-residual, zero-ask macro. Cash is net",
        "modeled 26-hour cash after physical recovery; less negative is better.",
        "Qualification includes both bias strata and the actual 90% service floor.",
        "",
        "| Family | Models | All gates | Facility MW | PCC MW | RMSE kW | "
        "Net cash USD | Minimum service |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]

    def number(value, precision=3):
        return "--" if value is None else f"{value:,.{precision}f}"

    for row in rows:
        minimum = row["minimum_service_ratio"]
        lines.append(
            f"| {row['family']} | {row['model_count']} | "
            f"{'Yes' if row['all_qualified'] else 'No'} | "
            f"{number(row['stress_facility_relief_mw'])} | "
            f"{number(row['stress_pcc_relief_mw'])} | "
            f"{number(row['tracking_rmse_kw'])} | "
            f"{number(row['cashflow_26h_usd'], 2)} | "
            f"{number(100 * minimum, 2) + '%' if minimum is not None else '--'} |"
        )
    lines.extend(
        [
            "",
            "## Predetermined promotion gate",
            "",
            "The comparator is a **composite per-metric envelope**, not a single",
            "achievable incumbent: strongest qualified family for cash, facility",
            "relief and PCC relief separately. It includes the new scripted greedy",
            "and price-margin controllers as well as earlier learned/frozen controls.",
            "A family must pass all four seeds' physical/service/recovery cases,",
            "improve both relief metrics, and improve cash without a negative mean",
            "cash contrast in either bias stratum. No threshold was relaxed.",
            "",
        ]
    )
    eligible = []
    for name, decision in result["decisions"].items():
        lines.append(
            f"- **{name}: joint improvement = {decision['joint_improvement']}.**"
        )
        if decision["joint_improvement"]:
            eligible.append(name)
    lines.extend(
        [
            "",
            "## Training accounting",
            "",
            f"Completed final models: **{completed_models}/16** at the registered",
            "24,576 macro transitions, two environments and 256 critic-only updates.",
            "PPO/SAC are matched on transitions, not optimizer counts or computation.",
            "Individual completion receipts record actual optimizer operations,",
            "episode qualification, conditions and retained transition arrays.",
            "",
            f"Raw evaluation primary checks: **{result['raw_primary_checks']:,}**.",
            "The underlying physical and recovery ledgers, true contract target,",
            "credit limits, raw cash and deployment without Phi were retained.",
            "",
            "## Scope and persistence",
            "",
            "- V1 was snapshotted before this follow-up at",
            "  `5e2bf2855a3da4ac8cfea1763df8e9d8b28e9f82`.",
            "- The adaptive-attempt ledger and one-further-campaign cap were frozen",
            "  before training. Reused Dev-B is not relabeled as fresh validation.",
            "- The shared six-date reservation has no new workload phases or p90-tail",
            "  coverage. Native 26-hour staleness checks passed for all six starts.",
            "- Constant offsets and battery-boundary effects are not called load",
            "  shifting or free regulation accuracy.",
            "",
            "## Evidence",
            "",
            "- [Audited outcomes](../track_b_fraction/development_report.json)",
            "- [Factorial registration](../track_b_fraction/registration.json)",
            "- [Exact warmup parity](../track_b_fraction/smoke_parity_receipt.json)",
            "- [Machine-readable comparison](fraction_summary.csv)",
            "- [Previous full-budget results](REPORT.md)",
            "- [Native calendar coverage](calendar_coverage_verification.json)",
            "- [Development economic frontier](economic_frontier.png)",
            "",
        ]
    )
    if eligible:
        lines.append(
            "Development-eligible families: "
            + ", ".join(eligible)
            + ". A separately locked one-shot final evaluation is still required."
        )
    else:
        lines.extend(
            [
                "**No family earned promotion under the registered gate.** The fresh",
                "calendar reservation remains unopened; do not substitute a different",
                "seed, earlier checkpoint, relaxed gate or repeated final opening.",
            ]
        )
    path = ROOT / "reports/FRACTION_REPORT.md"
    if path.exists():
        raise FileExistsError("Do not overwrite a factorial report")
    path.write_text("\n".join(lines) + "\n")
    draw_frontier(rows, result)
    print(path)


if __name__ == "__main__":
    main()
