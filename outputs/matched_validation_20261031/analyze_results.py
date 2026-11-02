"""Full-denominator readback of the registered validation matrix."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "plot-cache")
os.sched_setaffinity(0, {28})

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_complete(rows, key, expected):
    if len(rows) != expected or any(row.get(key) is None for row in rows):
        return None
    return float(np.mean([row[key] for row in rows]))


def main():
    status = json.loads((ROOT / "status.json").read_text())
    registration = json.loads((ROOT / "registration.json").read_text())
    inventory = registration["inventory"]
    episodes = json.loads((ROOT / "reports/episodes.json").read_text())
    pairs = json.loads((ROOT / "reports/comparisons.json").read_text())
    assert status["state"] == "completed" and status["runtime_failures"] == 0
    assert len(episodes) == registration["planned_policy_cells"] == 256
    assert len(pairs) == 492
    episode_map = {
        (e["policy_id"], e["cell"]["id"] if "cell" in e else e["cell_id"]): e
        for e in episodes
    }
    assert len(episode_map) == 256
    summary_rows, component_rows, safety_rows, failures = [], [], [], []
    for policy in inventory["policies"]:
        group = [episode_map[policy["id"], cell["id"]] for cell in inventory["cells"]]
        row = {
            "policy_id": policy["id"], "group": policy["group"],
            "base_seed": policy["base_seed"], "stream": policy["stream"],
            "available": policy["available"],
            "prior_selection_qualified": policy["prior_selection_qualified"],
            "diagnostic_only": policy["diagnostic_only"], "planned_cells": 8,
        }
        if not policy["available"]:
            row.update(complete_cells=0, qualified_cells=0, unavailable_reason=policy["unavailable_reason"])
            summary_rows.append(row)
            continue
        row.update(
            complete_cells=sum(e["complete"] for e in group),
            qualified_cells=sum(e["constraint_qualified"] for e in group),
            rmse_kw=mean_complete(group, "tracking_rmse_kw", 8),
            normalized_rmse=mean_complete(group, "normalized_rmse", 8),
            regulation_revenue_usd=mean_complete(group, "revenue_usd", 8),
            electricity_cost_usd=mean_complete(group, "electricity_cost_usd", 8),
            mean_committed_mw=float(np.mean([e["mean_committed_mw"] for e in group])),
            terminal_soc_mean=float(np.mean([e["terminal"]["soc"] for e in group])),
            terminal_soc_min=float(min(e["terminal"]["soc"] for e in group)),
            terminal_battery_drawdown_mwh=float(np.mean([e["terminal_battery_drawdown_mwh"] for e in group])),
        )
        nominal = [c for c in pairs if c["contrast"] == "vs_nominal" and c["a"] == policy["id"]]
        biased = [c for c in nominal if c["cell"]["regd_load_bias"] == .5]
        bias_effect = [c for c in pairs if c["contrast"] == "matched_bias_effect" and c["a"] == policy["id"]]
        row.update(
            biased_facility_relief_mw=mean_complete(biased, "stress_facility_relief_mw", 4),
            biased_pcc_relief_mw=mean_complete(biased, "stress_pcc_relief_mw", 4),
            biased_actual_system_peak_relief_mw=mean_complete(biased, "actual_system_facility_peak_reduction_mw", 4),
            biased_LFR_pct=mean_complete(biased, "LFR_pct", 4),
            biased_SPDR_proxy_pct=mean_complete(biased, "SPDR_proxy_pct", 4),
            matched_bias_revenue_change_pct=mean_complete(bias_effect, "regulation_revenue_change_pct", 4),
            bias_induced_stress_relief_mw=mean_complete(bias_effect, "stress_facility_relief_mw", 4),
            cashflow_vs_nominal_before_recovery_usd=mean_complete(nominal, "cashflow_delta_before_recovery_usd", 8),
        )
        summary_rows.append(row)
        if all(c["pair_complete"] for c in biased):
            components = {
                name: float(np.mean([c["component_stress_relief_mw"][name] for c in biased]))
                for name in biased[0]["component_stress_relief_mw"]
            }
            assert math.isclose(sum(components.values()), row["biased_facility_relief_mw"], abs_tol=1e-8)
            component_rows.append({
                "policy_id": policy["id"], "complete_biased_pairs": 4,
                "prior_selection_qualified": policy["prior_selection_qualified"],
                **components,
                "battery": mean_complete(biased, "stress_battery_support_mw", 4),
                "facility_total": row["biased_facility_relief_mw"],
                "pcc_total": row["biased_pcc_relief_mw"],
                "stress_work_shift_kwh": mean_complete(biased, "stress_work_shift_kwh", 4),
            })
        for episode in group:
            metrics = episode["observed_hardware_metrics"]
            reasons = []
            gate = inventory["selection_gate"]
            if not episode["complete"]:
                reasons.append("incomplete_or_terminated")
            if episode["terminal"]["soc"] < gate["min_terminal_soc"]:
                reasons.append("terminal_soc")
            if metrics["thermal_warning_fraction"] > gate["max_thermal_warning_fraction"]:
                reasons.append("thermal_warning")
            if episode["terminal"]["backlog_kwh"] > gate["max_backlog_fraction"] * metrics["work_arrived_kwh"] + 1e-6:
                reasons.append("backlog")
            if abs(metrics["work_conservation_error_kwh"]) >= 1e-4:
                reasons.append("work_conservation")
            assert bool(reasons) != episode["constraint_qualified"]
            safety_rows.append({
                "policy_id": policy["id"], "cell_id": episode["cell"]["id"],
                "original_validation_cell": episode["cell"]["original_validation_cell"],
                "complete": episode["complete"],
                "qualified": episode["constraint_qualified"],
                "failure_reasons": "|".join(reasons),
                "terminal_soc": episode["terminal"]["soc"],
                "terminal_backlog_kwh": episode["terminal"]["backlog_kwh"],
                "thermal_warning_fraction": metrics["thermal_warning_fraction"],
            })
            if not episode["complete"]:
                failures.append({
                    "policy_id": policy["id"], "cell": episode["cell"],
                    "observed_ticks": episode["observed_ticks"],
                    "terminal": episode["terminal"],
                })

    macro_rows, restored_rows = [], []
    for contrast, target in [
        ("macro_vs_matched_rule", macro_rows),
        ("restored_vs_selected_parent", restored_rows),
    ]:
        for a in sorted({c["a"] for c in pairs if c["contrast"] == contrast}):
            group = [c for c in pairs if c["contrast"] == contrast and c["a"] == a]
            b = group[0]["b"]
            row = {"a": a, "b": b, "planned_pairs": 8,
                   "complete_pairs": sum(c["pair_complete"] for c in group)}
            for key in (
                "cashflow_delta_before_recovery_usd", "regulation_revenue_delta_usd",
                "electricity_cost_delta_usd", "mean_committed_mw_delta",
                "terminal_stored_energy_delta_mwh", "terminal_temp_A_delta_C",
                "terminal_temp_B_delta_C", "terminal_backlog_delta_kwh",
                "battery_throughput_delta_mwh", "battery_age_fraction_delta",
                "stress_facility_relief_mw",
            ):
                row[key] = mean_complete(group, key, 8)
            for bias in (0., .5):
                subset = [c for c in group if c["cell"]["regd_load_bias"] == bias]
                tag = "unbiased" if bias == 0 else "biased"
                row[f"{tag}_cashflow_delta_before_recovery_usd"] = mean_complete(subset, "cashflow_delta_before_recovery_usd", 4)
                row[f"{tag}_terminal_energy_delta_mwh"] = mean_complete(subset, "terminal_stored_energy_delta_mwh", 4)
                row[f"{tag}_stress_facility_relief_delta_mw"] = mean_complete(subset, "stress_facility_relief_mw", 4)
            full = all(c["pair_complete"] for c in group)
            row["cashflow_positive_cells"] = sum(c["cashflow_delta_before_recovery_usd"] > 0 for c in group) if full else None
            row["nonnegative_terminal_energy_cells"] = sum(c["terminal_stored_energy_delta_mwh"] >= 0 for c in group) if full else None
            if full:
                ae = [episode_map[a, c["cell"]["id"]] for c in group]
                be = [episode_map[b, c["cell"]["id"]] for c in group]
                row.update(
                    a_rmse_kw=float(np.mean([e["tracking_rmse_kw"] for e in ae])),
                    b_rmse_kw=float(np.mean([e["tracking_rmse_kw"] for e in be])),
                    a_qualified_cells=sum(e["constraint_qualified"] for e in ae),
                    b_qualified_cells=sum(e["constraint_qualified"] for e in be),
                )
            target.append(row)

    reports = ROOT / "reports"
    summary = pd.DataFrame(summary_rows)
    components = pd.DataFrame(component_rows)
    macros = pd.DataFrame(macro_rows)
    restored = pd.DataFrame(restored_rows)
    summary.to_csv(reports / "policy_summary.csv", index=False)
    components.to_csv(reports / "biased_component_relief.csv", index=False)
    macros.to_csv(reports / "macro_incremental_summary.csv", index=False)
    restored.to_csv(reports / "restored_vs_parent_summary.csv", index=False)
    condition_rows = []
    for policy_id in sorted({c["a"] for c in pairs if c["contrast"] == "macro_vs_matched_rule"}):
        for day in (196, 256):
            for bias in (0., .5):
                group = [
                    c for c in pairs if c["contrast"] == "macro_vs_matched_rule"
                    and c["a"] == policy_id and c["cell"]["start_day"] == day
                    and c["cell"]["regd_load_bias"] == bias
                ]
                row = {
                    "policy_id": policy_id, "day": day, "bias": bias,
                    "planned_pairs": 2, "complete_pairs": sum(c["pair_complete"] for c in group),
                }
                for key in ("cashflow_delta_before_recovery_usd", "terminal_stored_energy_delta_mwh", "stress_facility_relief_mw"):
                    row[key] = mean_complete(group, key, 2)
                condition_rows.append(row)
    pd.DataFrame(condition_rows).to_csv(reports / "macro_condition_summary.csv", index=False)
    pd.DataFrame(safety_rows).to_csv(reports / "qualification_by_cell.csv", index=False)
    (reports / "incomplete_episodes.json").write_text(json.dumps(failures, indent=2))

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2})
    available = [p for p in inventory["policies"] if p["available"]]
    fig, ax = plt.subplots(figsize=(13, 12), constrained_layout=True)
    y = np.arange(len(available))
    lookup = components.set_index("policy_id")
    groups = [
        ("IT", ["it_A", "it_B"], "#3182bd"),
        ("HVAC", ["hvac"], "#31a354"),
        ("Pump + chiller", ["pump", "chiller"], "#74c476"),
        ("Electrical losses + aux", ["ups_loss", "pdu_loss", "transformer_loss", "auxiliary"], "#9e9ac8"),
        ("BESS contribution to PCC", ["battery"], "#fd8d3c"),
    ]
    positive, negative = np.zeros(len(y)), np.zeros(len(y))
    for label, keys, color in groups:
        values = np.array([
            sum(lookup.loc[p["id"], key] for key in keys) if p["id"] in lookup.index else 0.
            for p in available
        ])
        ax.barh(y, np.maximum(values, 0), left=positive, color=color, label=label)
        ax.barh(y, np.minimum(values, 0), left=negative, color=color)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    totals = [lookup.loc[p["id"], "facility_total"] if p["id"] in lookup.index else np.nan for p in available]
    ax.scatter(totals, y, color="black", marker="|", s=130, label="Facility-only total")
    labels = []
    for p in available:
        q = summary[summary.policy_id == p["id"]].iloc[0].qualified_cells
        labels.append(f"{p['id']}{'*' if p['diagnostic_only'] else ''} [{int(q)}/8 qualified]")
        if p["id"] not in lookup.index:
            ax.text(0.05, available.index(p), "N/A: incomplete biased pair", va="center", color="darkred")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean stress-window relief vs nominal (MW); beta=0.5, four matched cells")
    ax.set_title("Validation power decomposition: all available candidates retained\n* Previously unqualified restored final; ratios and qualification are not a deployment claim")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(reports / "biased_power_decomposition.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    valid = macros[macros.complete_pairs == 8]
    positions = np.arange(len(valid))
    labels = valid.a.str.replace("old_pair_", "", regex=False)
    axes[0].bar(positions-.18, valid.regulation_revenue_delta_usd, width=.36, label="Revenue change")
    axes[0].bar(positions+.18, -valid.electricity_cost_delta_usd, width=.36, label="Electricity saving")
    axes[0].scatter(positions, valid.cashflow_delta_before_recovery_usd, color="black", marker="D", label="Cashflow before recovery")
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_xticks(positions, labels)
    axes[0].set(title="Learned macro vs rule macro: exact same hardware", xlabel="Parent seed", ylabel="Mean USD per 24-hour validation episode")
    axes[0].legend(fontsize=8)
    axes[1].scatter(valid.cashflow_delta_before_recovery_usd, valid.terminal_stored_energy_delta_mwh, s=55)
    for _, row in valid.iterrows():
        axes[1].annotate(row.a.replace("old_pair_", ""), (row.cashflow_delta_before_recovery_usd, row.terminal_stored_energy_delta_mwh), xytext=(4,4), textcoords="offset points")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set(title="Terminal inventory remains a separate obligation", xlabel="Cashflow before recovery (USD)", ylabel="Stored battery energy change vs rule (MWh)")
    fig.suptitle("Matched macro validation; seed1002 omitted from aggregate chart because 2/8 episodes terminated")
    fig.savefig(reports / "macro_cashflow_and_inventory.png", dpi=170)
    plt.close(fig)

    selected = [c for c in pairs if c["contrast"] == "macro_vs_matched_rule" and c["a"] == "old_pair_1006"]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for day, color in [(196, "#3182bd"), (256, "#e6550d")]:
        for bias, marker in [(0., "o"), (.5, "^")]:
            group = [c for c in selected if c["cell"]["start_day"] == day and c["cell"]["regd_load_bias"] == bias]
            ax.scatter(
                [c["cashflow_delta_before_recovery_usd"] for c in group],
                [c["terminal_stored_energy_delta_mwh"] for c in group],
                color=color, marker=marker, s=70, label=f"day {day}, bias {bias:g}",
            )
            for c in group:
                right_edge = c["cashflow_delta_before_recovery_usd"] > 3000
                ax.annotate(
                    str(c["cell"]["seed"]),
                    (c["cashflow_delta_before_recovery_usd"], c["terminal_stored_energy_delta_mwh"]),
                    xytext=(-5 if right_edge else 5, 5),
                    ha="right" if right_edge else "left",
                    textcoords="offset points", fontsize=8,
                )
    ax.axhline(0, color="black", linewidth=.8)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(
        title="Macro1006 vs rule macro with identical hardware: eight individual cells\n"
              "Cashflow positive in6/8; stored battery energy nonnegative in3/8",
        xlabel="Cashflow change BEFORE recovery/degradation valuation (USD)",
        ylabel="Terminal stored battery energy change (MWh)",
    )
    ax.legend()
    fig.savefig(reports / "macro1006_cell_tradeoffs.png", dpi=170)
    plt.close(fig)

    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registration_id": registration["registration_id"],
        "planned_policy_cells": 256, "runtime_completed_rollouts": 232,
        "full_horizon_rollouts": sum(e.get("complete", False) for e in episodes),
        "unavailable_policy_cells": 24, "runtime_errors": 0,
        "pairwise_records": len(pairs), "incomplete_episodes": failures,
        "interpretation": [
            "All policy cells retained. Any incomplete member suppresses that policy/contrast aggregate rather than using a shorter prefix.",
            "Old selected and restored final candidates have different selection histories; no algorithm-level superiority claim.",
            "Four original cells plus four same-seed bias counterparts; these are validation data, not held-out peak-day tests.",
            "Cashflow is before recovery/degradation/service valuation; mean terminal inventories do not establish per-cell dominance.",
            "Mean zero IT/work contribution does not prove no intra-day deferral; the full traces retain workload history.",
            "Hardware and macro reward scales are separate; primary evidence is physical service, economics and qualification.",
        ],
    }
    receipt_path = reports / "readback.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt_path.write_text(json.dumps({**previous, **receipt}, indent=2))
    print("Saved verified summaries and figures:", reports)
    print(macros.to_string(index=False))


if __name__ == "__main__":
    main()
