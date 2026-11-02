"""Derived presentation after verify_results.py; registered metrics stay unchanged."""

from __future__ import annotations

import json

# isort: off
from run_evaluation import ROOT, load_json, mean_sd, verify_registration
from recovery import matched
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
# isort: on


def label(identifier: str) -> str:
    classical = {
        "nominal": "Nominal",
        "pid": "PID",
        "rule_hardware": "Rule",
        "bang_bang": "Bang-bang",
    }
    if identifier in classical:
        return classical[identifier]
    if identifier.startswith("old_low_"):
        return f"Rule + SAC {identifier[-4:]}"
    return f"Macro + SAC {identifier[-4:]}"


def main():
    registration = load_json(ROOT / "heldout_registration.json")
    verify_registration(registration)
    selection = load_json(ROOT / "selection.json")
    reports = ROOT / "heldout/reports"
    episodes = load_json(reports / "episodes.json")
    comparisons = load_json(reports / "comparisons.json")
    summaries = load_json(reports / "policy_summary.json")
    conditions = load_json(reports / "macro_condition_summary.json")
    readback = load_json(reports / "readback.json")
    verification = load_json(reports / "verification.json")
    primary_low = selection["primary_hardware"]
    primary_macro = selection["primary_macro_pair"]
    n = len(registration["cells"])
    nb = n // 2
    core = ["rule_hardware", "pid", "bang_bang", primary_low]
    colors = ["#64748b", "#16a34a", "#d97706", "#2563eb"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), layout="constrained")
    for axis, metric, title, units in (
        (
            axes[0],
            "stress_facility_relief_mw",
            "Facility relief under biased RegD",
            "MW; battery excluded",
        ),
        (axes[1], "tracking_rmse_kw", "Tracking under biased RegD", "RMSE (kW)"),
        (
            axes[2],
            "actual_system_facility_peak_reduction_mw",
            "Actual modeled peak reduction",
            "MW; max(load + facility), battery excluded",
        ),
    ):
        for index, policy_id in enumerate(core):
            if metric == "tracking_rmse_kw":
                group = [
                    row
                    for row in episodes
                    if row["policy_id"] == policy_id
                    and row["cell"]["regd_load_bias"] == 0.5
                ]
            else:
                group = [
                    row
                    for row in comparisons
                    if row["a"] == policy_id
                    and row["contrast"] == "vs_nominal"
                    and row["cell"]["regd_load_bias"] == 0.5
                ]
            mean, sd = mean_sd(group, metric, nb)
            if mean is not None:
                axis.bar(
                    index, mean, yerr=sd, capsize=4, color=colors[index], width=0.65
                )
            else:
                axis.text(
                    index,
                    0.05,
                    "No full-cohort\nestimate",
                    ha="center",
                    va="bottom",
                    transform=axis.get_xaxis_transform(),
                    fontsize=8,
                )
        axis.set_xticks(
            range(len(core)), [label(p) for p in core], rotation=15, ha="right"
        )
        axis.set_title(title)
        axis.set_ylabel(units)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Locked hardware comparison: rule-based macro for every column\n"
        "15 biased peak-day/seed cases; error bars are sample SD, "
        "not confidence intervals",
        fontsize=11,
    )
    fig.savefig(reports / "heldout_hardware.png", dpi=170)
    plt.close(fig)

    selected_conditions = [
        row for row in conditions if row["policy_id"] == primary_macro
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for index, row in enumerate(selected_conditions):
        for shift, key, color, legend in (
            (-0.18, "cashflow_delta_before_recovery_usd", "#94a3b8", "Before recovery"),
            (
                0.18,
                "cashflow_delta_after_closed_recovery_usd",
                "#2563eb",
                "After closed recovery",
            ),
        ):
            mean = row[key]
            if mean is not None:
                ax.bar(
                    index + shift,
                    mean,
                    width=0.34,
                    yerr=row[key + "_sd"],
                    capsize=3,
                    color=color,
                    label=legend if index == 0 else None,
                )
            else:
                ax.text(
                    index + shift,
                    0,
                    "N/A",
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=8,
                )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(
        range(len(selected_conditions)),
        [
            f"Day {row['start_day']}, bias {row['bias']:g}\n"
            f"closed pairs {row['closed_recovery_pairs']}/{row['planned_pairs']}"
            for row in selected_conditions
        ],
        fontsize=8,
    )
    ax.set_ylabel("Incremental benchmark cashflow (USD)")
    ax.set_title(
        f"{label(primary_macro)} versus rule macro with the exact same hardware\n"
        "Five seeds per condition; sample SD. Wear/SLA valuation excluded."
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(reports / "heldout_macro_cashflow.png", dpi=170)
    plt.close(fig)

    palette = ["#cbd5e1", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#2563eb"]
    status_names = [
        "Unavailable / not admitted",
        "Service qualified + recovery closed",
        "Service gate violation",
        "Service termination / runtime failure",
        "Recovery incomplete",
        "Recovery inventory unclosed",
    ]
    policy_ids = [policy["id"] for policy in registration["policies"]]
    cells = registration["cells"]
    lookup = {(row["policy_id"], row["cell"]["id"]): row for row in episodes}
    statuses = np.zeros((len(policy_ids), len(cells)), dtype=int)
    for i, policy_id in enumerate(policy_ids):
        for j, cell in enumerate(cells):
            row = lookup[policy_id, cell["id"]]
            if row["status"] == "unavailable":
                status = 0
            elif not row.get("complete"):
                status = 3
            elif not row["recovery"]["complete"]:
                status = 4
            elif not row["recovery"]["closed"]:
                status = 5
            elif not row["constraint_qualified"]:
                status = 2
            else:
                status = 1
            statuses[i, j] = status
    fig, ax = plt.subplots(figsize=(14, 9), layout="constrained")
    ax.imshow(
        statuses, cmap=ListedColormap(palette), vmin=-0.5, vmax=5.5, aspect="auto"
    )
    ax.set_yticks(range(len(policy_ids)), [label(p) for p in policy_ids], fontsize=8)
    ax.set_xticks(
        range(len(cells)),
        [f"{c['start_day']}\n{c['seed']}\n{c['regd_load_bias']:g}" for c in cells],
        fontsize=7,
    )
    for boundary in (9.5, 19.5):
        ax.axvline(boundary, color="white", linewidth=2)
    ax.set_xlabel("Day / evaluation seed / bias")
    ax.set_title(
        "All 600 registered policy-cells retained\n"
        "Color is diagnostic priority; service and recovery flags "
        "remain separate in the records"
    )
    ax.legend(
        handles=[
            Patch(facecolor=color, label=name)
            for color, name in zip(palette, status_names)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        fontsize=8,
    )
    fig.savefig(reports / "heldout_all_attempts.png", dpi=170)
    plt.close(fig)

    wear_rows = []
    for contrast in ("vs_nominal", "low_vs_pid", "macro_vs_matched_rule"):
        for policy_id in policy_ids:
            for scope, expected in (("all", n), ("biased", nb)):
                group = [
                    row
                    for row in comparisons
                    if row["contrast"] == contrast
                    and row["a"] == policy_id
                    and (scope == "all" or row["cell"]["regd_load_bias"] == 0.5)
                ]
                if not group:
                    continue
                for price in registration["recovery"]["wear_prices_usd_per_mwh"]:
                    values = [
                        {
                            "cash": row.get("wear_adjusted_cashflow_usd", {}).get(
                                str(price)
                            )
                        }
                        for row in group
                    ]
                    mean, sd = mean_sd(values, "cash", expected)
                    wear_rows.append(
                        {
                            "contrast": contrast,
                            "policy_id": policy_id,
                            "scope": scope,
                            "planned_pairs": expected,
                            "closed_pairs": sum(
                                row["recovery_pair_closed"] for row in group
                            ),
                            "assumed_wear_usd_per_mwh_ac_throughput": price,
                            "mean_adjusted_cashflow_usd": mean,
                            "sample_sd_usd": sd,
                            "positive_pairs": sum(row["cash"] > 0 for row in values)
                            if mean is not None
                            else None,
                        }
                    )
    pd.DataFrame(wear_rows).to_csv(reports / "wear_sensitivity.csv", index=False)

    components = []
    for policy_id in policy_ids:
        group = [
            row
            for row in comparisons
            if row["contrast"] == "vs_nominal"
            and row["a"] == policy_id
            and row["cell"]["regd_load_bias"] == 0.5
        ]
        for component in matched.COMPONENTS:
            values = [
                {"value": row.get("component_stress_relief_mw", {}).get(component)}
                for row in group
            ]
            mean, sd = mean_sd(values, "value", nb)
            components.append(
                {
                    "policy_id": policy_id,
                    "component": component,
                    "mean_biased_stress_relief_mw": mean,
                    "sample_sd_mw": sd,
                    "planned_cases": nb,
                }
            )
    pd.DataFrame(components).to_csv(
        reports / "biased_component_relief.csv", index=False
    )

    primary_comparisons = [
        row
        for row in comparisons
        if (row["a"] == primary_low and row["contrast"] == "low_vs_pid")
        or (row["a"] == primary_macro and row["contrast"] == "macro_vs_matched_rule")
    ]
    paired_summary = []
    for policy_id in (primary_low, primary_macro):
        group = [row for row in primary_comparisons if row["a"] == policy_id]
        result = {
            "policy_id": policy_id,
            "planned_pairs": n,
            "complete_service_pairs": sum(row["pair_complete"] for row in group),
            "closed_recovery_pairs": sum(row["recovery_pair_closed"] for row in group),
            "positive_closed_cashflow_cases": sum(
                row["cashflow_delta_after_closed_recovery_usd"] > 0
                for row in group
                if row.get("cashflow_delta_after_closed_recovery_usd") is not None
            ),
        }
        for metric in (
            "cashflow_delta_before_recovery_usd",
            "cashflow_delta_after_closed_recovery_usd",
            "total_battery_throughput_delta_mwh",
            "total_backlog_exposure_delta_kwh_hours",
        ):
            result[metric], result[metric + "_sd"] = mean_sd(group, metric, n)
        cash = result["cashflow_delta_after_closed_recovery_usd"]
        throughput = result["total_battery_throughput_delta_mwh"]
        result["mean_cashflow_wear_break_even_usd_per_mwh"] = (
            cash / throughput
            if cash is not None
            and throughput is not None
            and abs(throughput) > 1e-12
            and cash / throughput >= 0
            else None
        )
        result["mean_cashflow_wear_slope_mwh"] = (
            -throughput if cash is not None else None
        )
        paired_summary.append(result)
    summary_by_id = {row["policy_id"]: row for row in summaries}
    low_summary, pid_summary = summary_by_id[primary_low], summary_by_id["pid"]
    final = {
        "heldout_registration_id": registration["registration_id"],
        "selection_locked_before_test": selection["heldout_results_seen"] is False,
        "primary_hardware": primary_low,
        "primary_macro_pair": primary_macro,
        "readback": readback,
        "independent_verification": verification,
        "primary_hardware_tradeoff_vs_pid": {
            "biased_stress_window_relief_delta_mw": (
                low_summary["biased_facility_relief_mw"]
                - pid_summary["biased_facility_relief_mw"]
            ),
            "all_case_mean_rmse_delta_kw": (
                low_summary["rmse_kw"] - pid_summary["rmse_kw"]
            ),
            "biased_actual_modeled_system_peak_relief_delta_mw": (
                low_summary["biased_actual_system_peak_relief_mw"]
                - pid_summary["biased_actual_system_peak_relief_mw"]
            ),
            "interpretation": (
                "Stress-average relief, tracking, and maximum-load reduction "
                "are distinct objectives; improvement in one is not dominance in all."
            ),
        },
        "primary_and_reference_summary": [
            row
            for row in summaries
            if row["policy_id"]
            in {*core, primary_macro, selection["macro_exact_hardware_reference"]}
        ],
        "primary_pairwise_summary": paired_summary,
        "primary_macro_conditions": selected_conditions,
        "interpretation_limits": list(
            dict.fromkeys(
                [
                    *readback["interpretation"],
                    *verification["interpretation_limits"],
                    "Measured mean relief is mainly HVAC modulation, "
                    "not net IT deferral.",
                ]
            )
        ),
        "input_sha256": {
            name: matched.file_hash(reports / name)
            for name in ("episodes.json", "comparisons.json", "policy_summary.json")
        },
    }
    matched.write_json(reports / "final_report.json", final)
    print(
        json.dumps(
            {"primary": [primary_low, primary_macro], "readback": readback}, indent=2
        )
    )


if __name__ == "__main__":
    main()
