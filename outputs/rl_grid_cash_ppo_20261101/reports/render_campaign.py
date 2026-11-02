"""Render verified development results without inventing a final-test claim."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def display(value, digits=3):
    return "--" if value is None else f"{value:,.{digits}f}"


def training_rows(track):
    protocol = read(ROOT / track / "registration.json")
    rows = []
    for algorithm in protocol["algorithms"]:
        records, failed = [], []
        for seed in protocol["seeds"]:
            directory = ROOT / track / f"runs/full/{algorithm}_{seed}"
            path = directory / "completion.json"
            if not path.exists():
                failure = directory / "failure.json"
                if not failure.exists():
                    raise RuntimeError("Training is still active; no completed report")
                failed.append(seed)
                continue
            record = read(path)
            key = "num_timesteps" if track == "track_a" else "macro_transitions"
            if (
                record["status"] != "completed"
                or record["registration_id"] != protocol["registration_id"]
                or record[key] != protocol["budgets"]["full"]
            ):
                raise ValueError("Training receipt differs from the registered budget")
            records.append(record)
        actors = ", ".join(str(row["optimizer_updates"]["actor"]) for row in records)
        critics = ", ".join(str(row["optimizer_updates"]["critic"]) for row in records)
        rows.append(
            f"| {track} / {algorithm.upper()} | {len(records)}/4 | "
            f"{protocol['budgets']['full']:,} | {actors or '--'} | {critics or '--'} |"
        )
        if failed:
            rows.append(
                f"\nFailed training slots retained for {track}/{algorithm}: {failed}.\n"
            )
    return rows


def main():
    low = read(ROOT / "track_a/development_report.json")
    high = read(ROOT / "track_b/development_report.json")
    if (
        low["role"] != "consumed_development_only"
        or high["role"] != "consumed_development_only"
        or not low["final_data_not_opened"]
        or not high["final_data_not_opened"]
    ):
        raise ValueError("This renderer is scoped to the registered development panels")
    lines = [
        "# Matched PPO/SAC grid-relief and economics results",
        "",
        "These results use the 24 previously consumed Dev-B conditions. They are",
        "**development evidence**, not the original three-date Table 2, a fresh",
        "final test, or a demonstrated replacement of the paper's benchmark.",
        "The two tracks below have different objectives and must not be pooled.",
        "",
        "## Track A: original low-level tracking task",
        "",
        "The original reward, causal observations, full-work actuator space and",
        "rule macro are unchanged. Means below use bias 0.5 and include the same",
        "two-hour physical recovery in cash. Cash is modeled net cash, not profit.",
        "",
        "| Controller family | Models | All physical gates | RMSE, kW | "
        "Facility relief, MW | PCC relief, MW | Net cash, USD/26h |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    low_families = {**low["frozen_families"], **low["learned_families"]}
    for name in (
        "pid",
        "old_sac",
        "cooling_first",
        "bc_only",
        "guided_sac",
        "union_original",
        "union_conditioned",
        "native_ppo",
        "native_sac",
    ):
        row = low_families.get(name)
        if row is None:
            lines.append(f"| {name} | 0 | Not admitted | -- | -- | -- | -- |")
            continue
        values = row["strata"].get("0.5", {})
        lines.append(
            f"| {name} | {row['model_count']} | "
            f"{'Yes' if row['all_qualified'] else 'No'} | "
            f"{display(values.get('rmse_kw'))} | "
            f"{display(values.get('facility_relief_mw'))} | "
            f"{display(values.get('pcc_relief_mw'))} | "
            f"{display(values.get('cash_usd'), 2)} |"
        )
    lines.extend(
        [
            "",
            "Final-step-only decisions (mean tracking compared with each seed's own",
            "BC in both bias strata; individual-case regressions are in the JSON):",
            "",
        ]
    )
    for algorithm, decision in low["decisions"].items():
        lines.append(
            f"- **{algorithm.upper()}:** four-seed tracking non-inferiority = "
            f"**{decision['all_four_noninferior_to_own_bc']}**; joint relief/cash "
            f"improvement = **{decision['joint_improvement']}**."
        )
    lines.extend(
        [
            "",
            "## Track B: disclosed economic-dispatch variant",
            "",
            "All rows retain the existing full-residual, zero-ask bidding policy.",
            "The learned policy allocates additional requested relief using a causal",
            "performance-credit guard. The actual 90% capacity-service floor and",
            "physical/recovery gates are enforced. More tracking error is permitted",
            "in this separate task; it is not a free regulation-accuracy gain.",
            "Means below use bias 0.5; qualification includes both biases.",
            "",
            "| Controller family | Models | All gates | RMSE, kW | "
            "Facility relief, MW | PCC relief, MW | Net cash, USD/26h |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    families = {
        **high["scripted_and_frozen_families"],
        **high["learned_families"],
    }
    for name in (
        "cooling_first",
        "bc_only",
        "union_conditioned",
        "constant_one",
        "greedy_budget",
        "margin_ten",
        "economic_ppo",
        "economic_sac",
    ):
        row = families[name]
        values = row["strata"].get("0.5", {}).get("metrics", {})

        def mean(key):
            return values[key]["mean"] if key in values else None

        lines.append(
            f"| {name} | {row['model_count']} | "
            f"{'Yes' if row['all_qualified'] else 'No'} | "
            f"{display(mean('tracking_rmse_kw'))} | "
            f"{display(mean('stress_facility_relief_mw'))} | "
            f"{display(mean('stress_pcc_relief_mw'))} | "
            f"{display(mean('cashflow_26h_usd'), 2)} |"
        )
    lines.extend(["", "Predetermined four-seed joint-improvement decisions:", ""])
    for family, decision in high["decisions"].items():
        lines.append(f"- **{family}: {decision['joint_improvement']}**.")
    lines.extend(
        [
            "",
            "The economic comparison includes the newly measured scripted budget",
            "and price-margin controls, not just the weaker zero-overlay controller.",
            "Facility/PCC differences and inventory effects remain explicit.",
            "A constant offset is not temporal load shifting or learned forecasting.",
            "",
            "## Exact training budgets",
            "",
            "Track A counts five-second physical training transitions; Track B counts",
            "15-minute transitions with physical service and recovery.",
            "Budgets match **within** a track, not between task cadences.",
            "PPO counts below are actual optimizer operations, not SB3 epoch counters.",
            "",
            "| Track / algorithm | Completed models | Transitions per model | "
            "Actor updates by seed | Critic updates by seed |",
            "|---|---:|---:|---|---|",
            *training_rows("track_a"),
            *training_rows("track_b"),
            "",
            "All final checkpoints are the registered mandatory last step. No earlier",
            "'best' checkpoint substitutes for an unsuccessful final model. The actors",
            "were held fixed during 256 critic-only updates in each fit. PPO and SAC",
            "differ in value/Q critics, entropy, exploration after initialization and",
            "sample reuse; equal transitions do not mean equal computation.",
            "",
            "## Verification and scope",
            "",
            f"- Track A raw primary checks: **{low['raw_primary_checks']:,}**.",
            f"- Track B raw primary checks: **{high['raw_primary_checks']:,}**.",
            "- Earlier headroom experiment: 144/144 qualified, 36,720 raw checks.",
            "- Failed training slots and incomplete evaluation cases are retained.",
            "- No survivor-only four-seed family average is used.",
            "- Development conditions are correlated; no significance claim is made.",
            "- Five independent adversarial reviews informed the design; tests and",
            "  raw data, not reviewer consensus, determine the claims.",
            "- Frozen physics, prior studies, original manuscript and read-only",
            "  external study materials were not changed.",
            "",
            "## Reopening the evidence",
            "",
            "- [Track A audited results](../track_a/development_report.json)",
            "- [Track B audited results](../track_b/development_report.json)",
            "- [Track A protocol](../track_a/registration.json)",
            "- [Track B protocol](../track_b/registration.json)",
            "- [Pre-learning headroom](HEADROOM.md)",
            "- [Reserved calendar combinations](../final_reservation.json)",
            "",
        ]
    )
    opening = ROOT / "final_opening_grid_cash.json"
    if not opening.exists():
        lines.extend(
            [
                "**The six reserved calendar combinations remain unopened.** They",
                "are neither an unseen-workload-phase test nor a p90-tail holdout.",
                "No final-test or paper-replacement claim follows from this report.",
            ]
        )
    else:
        lines.append(
            "A final-opening receipt exists; this report remains development-only."
        )
    output = ROOT / "reports/REPORT.md"
    if output.exists():
        raise FileExistsError("Do not replace an existing campaign report")
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
