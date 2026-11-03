"""Render the frozen Table 1 results without selecting policies or running cases."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_table1 import METRICS
from prepare_table1 import PROJECT, ROOT, verify
from table1_eval import file_hash

PUBLICATION = ROOT / "publication"
LABELS = {
    "rule_macro+none": "Rule macro + no agent",
    "rule_macro+random": "Rule macro + Random",
    "rule_macro+bang_bang": "Rule macro + Bang-bang",
    "rule_macro+rule_based": "Rule macro + Rule",
    "rule_macro+pid": "Rule macro + PID",
    "rule_macro+sac": "Rule macro + SAC",
    "sac_macro+sac": "SAC macro + SAC",
    "ppo_macro+sac": "PPO macro + SAC",
    "llm_macro+sac": "LLM macro + SAC",
}
PRECISION = {
    "corrected": dict(zip(METRICS, (1, 1, 3, 3, 3, 2), strict=True)),
    "archived": dict(zip(METRICS, (0, 0, 2, 2, 1, 2), strict=True)),
}
CONTRACT_COLUMNS = (
    "bid_mw",
    "bid_price",
    "bid_accepted",
    "committed_mw",
    "rate",
    "rmcp_usd",
    "residual_mw",
    "settlement_hours",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
        f"Displayed aggregate differs from all-case recomputation: {label}",
    )


def validate_report(report: dict, protocol: dict) -> None:
    require(
        report["registration_id"] == protocol["registration_id"],
        "Report registration differs from the frozen protocol",
    )
    expected_ids = [row["id"] for row in protocol["policies"]]
    require(
        [row["id"] for row in report["rows"]] == expected_ids,
        "Corrected rows differ from the registered panel",
    )
    expected_cases = {
        (policy, seed) for policy in expected_ids for seed in protocol["panel"]["seeds"]
    }
    actual_cases = [(case["policy"], case["seed"]) for case in report["cases"]]
    require(
        set(actual_cases) == expected_cases
        and len(actual_cases) == len(expected_cases),
        "Missing, duplicate or unregistered result cases",
    )
    require(
        report["planned"] == report["recorded"] == len(expected_cases),
        "Reported cohort count differs",
    )
    for row in report["rows"]:
        cases = [case for case in report["cases"] if case["policy"] == row["id"]]
        complete = sum(case["complete_service"] for case in cases)
        qualified = sum(case["qualified"] for case in cases)
        require(
            row["planned"] == len(cases)
            and row["complete_service"] == complete
            and row["qualified"] == qualified,
            f"Wrong row counts: {row['id']}",
        )
        require(
            row["diagnostic_not_admissible"] == (qualified != len(cases)),
            f"Qualification marker differs: {row['id']}",
        )
        for key in METRICS:
            metric = row["metrics"][key]
            if complete != len(cases):
                require(
                    metric == {"mean": None, "sd": None},
                    f"Incomplete cohort must not have a primary estimate: {row['id']}",
                )
                continue
            values = [case["metrics"][key] for case in cases]
            close(metric["mean"], statistics.fmean(values), f"{row['id']}/{key}/mean")
            close(metric["sd"], statistics.stdev(values), f"{row['id']}/{key}/SD")
        for key, source in (
            ("mean_committed_mw", "mean_committed_mw"),
            ("mean_cash_26h_usd", "closed_26h_cash_usd"),
            ("mean_observed_battery_throughput_mwh", "battery_throughput_mwh"),
        ):
            values = [case[source] for case in cases]
            if any(value is None for value in values):
                require(row[key] is None, f"Missing supplemental case: {key}")
            else:
                close(row[key], statistics.fmean(values), f"{row['id']}/{key}")
        close(
            row["service_end_soc_min"],
            min(case["service_end_soc"] for case in cases),
            f"{row['id']}/minimum service-end SOC",
        )
    require(
        report["complete_service"]
        == sum(c["complete_service"] for c in report["cases"])
        and report["qualified_supplemental"]
        == sum(c["qualified"] for c in report["cases"]),
        "Total completion or qualification count differs",
    )


def displayed_cell(metric: dict, decimals: int) -> dict:
    mean, sd = metric["mean"], metric["sd"]
    if mean is None or sd is None:
        require(mean is None and sd is None, "A metric has only one missing statistic")
        return {"mean": None, "sd": None, "decimals": decimals, "display": "---"}
    require(
        math.isfinite(mean) and math.isfinite(sd) and sd >= 0,
        "A displayed metric is non-finite or has negative SD",
    )
    mean_text, sd_text = f"{mean:,.{decimals}f}", f"{sd:,.{decimals}f}"
    return {
        "mean": mean,
        "sd": sd,
        "decimals": decimals,
        "mean_display": mean_text,
        "sd_display": sd_text,
        "display": f"{mean_text} +/- {sd_text}",
    }


def display_panel(rows: list[dict], panel: str) -> list[dict]:
    result = []
    for row in rows:
        entry = {
            "policy": row["id"],
            "label": LABELS[row["id"]],
            "metrics": {
                key: displayed_cell(row["metrics"][key], PRECISION[panel][key])
                for key in METRICS
            },
        }
        if panel == "corrected":
            entry.update(
                planned=row["planned"],
                complete_service=row["complete_service"],
                qualified=row["qualified"],
                diagnostic_not_admissible=row["diagnostic_not_admissible"],
            )
        result.append(entry)
    return result


def require_identical_contracts(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in CONTRACT_COLUMNS:
        require(
            np.array_equal(left[column].to_numpy(), right[column].to_numpy()),
            f"PID/SAC actual macro contracts differ: {column}",
        )


def collect_tradeoffs(report: dict, protocol: dict) -> dict:
    rows = {row["id"]: row for row in report["rows"]}
    cases = {(case["policy"], case["seed"]): case for case in report["cases"]}

    def macro_frame(policy: str, seed: int) -> pd.DataFrame:
        case = cases[policy, seed]
        episode_path = ROOT / case["episode_path"]
        require(
            file_hash(episode_path) == case["episode_sha256"],
            f"Episode receipt changed: {episode_path}",
        )
        episode = read_json(episode_path)
        path = episode_path.parent / "service_macro.csv.gz"
        require(
            file_hash(path) == episode["trace_hashes"][path.name],
            f"Market trace changed: {path}",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The 'float_precision' argument is deprecated."
            )
            return pd.read_csv(path, float_precision="round_trip")

    paired, high = [], []
    for seed in protocol["panel"]["seeds"]:
        require_identical_contracts(
            macro_frame("rule_macro+pid", seed), macro_frame("rule_macro+sac", seed)
        )
        a, b = cases["rule_macro+pid", seed], cases["rule_macro+sac", seed]
        paired.append(
            {
                "seed": seed,
                "identical_actual_macro_contracts": True,
                "rmse_delta_sac_minus_pid_kw": (
                    b["metrics"]["tracking_rmse_kw"] - a["metrics"]["tracking_rmse_kw"]
                ),
                "revenue_delta_sac_minus_pid_usd": (
                    b["metrics"]["regulation_revenue_usd"]
                    - a["metrics"]["regulation_revenue_usd"]
                ),
                "aging_delta_sac_minus_pid_x1e4": (
                    b["metrics"]["bess_aging_x1e4"] - a["metrics"]["bess_aging_x1e4"]
                ),
            }
        )
        frame = macro_frame("sac_macro+sac", seed)
        high.append(
            {
                "seed": seed,
                "windows": len(frame),
                "accepted": int(frame.bid_accepted.sum()),
                "mean_bid_price_usd_mwh": float(frame.bid_price.mean()),
                "mean_posted_rmcp_usd_mwh": float(frame.rmcp_usd.mean()),
                "bids_above_posted_price": int(
                    (frame.bid_price > frame.rmcp_usd).sum()
                ),
                "all_actual_commitments_mw": sorted(set(frame.committed_mw.tolist())),
            }
        )
    a, b = rows["rule_macro+pid"], rows["rule_macro+sac"]
    return {
        "registration_id": report["registration_id"],
        "paired_low_sac_vs_pid": paired,
        "rmse_reduction_percent": 100
        * (
            1
            - b["metrics"]["tracking_rmse_kw"]["mean"]
            / a["metrics"]["tracking_rmse_kw"]["mean"]
        ),
        "revenue_delta_usd": b["metrics"]["regulation_revenue_usd"]["mean"]
        - a["metrics"]["regulation_revenue_usd"]["mean"],
        "aging_increase_percent": 100
        * (
            b["metrics"]["bess_aging_x1e4"]["mean"]
            / a["metrics"]["bess_aging_x1e4"]["mean"]
            - 1
        ),
        "cash26h_delta_usd": b["mean_cash_26h_usd"] - a["mean_cash_26h_usd"],
        "high_macro_bidding": high,
        "high_total_accepted": sum(row["accepted"] for row in high),
        "high_total_windows": sum(row["windows"] for row in high),
        "statistical_significance_claimed": False,
        "original_vs_corrected_algorithm_only_gain_claimed": False,
        "producer_sha256": file_hash(Path(__file__)),
    }


def tex_number(value: str) -> str:
    return value.replace(",", "{,}")


def tex_table(rows: list[dict], *, corrected: bool) -> str:
    columns = r"lrrrrrrrr" if corrected else r"lrrrrrr"
    header = (
        r"Macro + hardware & \shortstack{RMSE\\(kW)} & "
        r"\shortstack{Reg. rev.\\(USD)} & \shortstack{Macro\\reward} & "
        r"\shortstack{Low\\reward} & \shortstack{Warning\\fraction} & "
        r"\shortstack{Age\\($10^{-4}$)}"
    )
    if corrected:
        header += r" & \shortstack{Full\\24h} & \shortstack{All\\gates}"
    lines = [rf"\begin{{tabular}}{{{columns}}}\toprule", header + r" \\\midrule"]
    for row in rows:
        label = row["label"]
        if corrected and row["diagnostic_not_admissible"]:
            label += r"$^\dagger$"
        cells = [label]
        for key in METRICS:
            metric = row["metrics"][key]
            cells.append(
                "---"
                if metric["mean"] is None
                else rf"\numcell{{{tex_number(metric['mean_display'])}}}"
                rf"{{{tex_number(metric['sd_display'])}}}"
            )
        if corrected:
            cells += [
                f"{row['complete_service']}/{row['planned']}",
                f"{row['qualified']}/{row['planned']}",
            ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule\end{tabular}"])


def supplemental_rows(report: dict) -> list[dict]:
    result = []
    for row in report["rows"]:
        cases = [case for case in report["cases"] if case["policy"] == row["id"]]
        result.append(
            {
                "policy": row["id"],
                "label": LABELS[row["id"]],
                "mean_committed_mw": row["mean_committed_mw"],
                "mean_cash_26h_usd": row["mean_cash_26h_usd"],
                "mean_battery_throughput_26h_mwh": row[
                    "mean_observed_battery_throughput_mwh"
                ],
                "minimum_service_end_soc": row["service_end_soc_min"],
                "recovery_closed": sum(case["recovery_closed"] for case in cases),
                "qualified": row["qualified"],
                "planned": len(cases),
            }
        )
    return result


def supplemental_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}\toprule",
        r"Macro + hardware & \shortstack{Mean\\commitment\\(MW)} & "
        r"\shortstack{Mean grid cash\\26h (USD)} & "
        r"\shortstack{Mean BESS\\throughput\\26h (MWh)} & "
        r"\shortstack{Minimum\\SOC at 24h} & \shortstack{Recovery\\closed} & "
        r"\shortstack{All\\gates} \\\midrule",
    ]
    for row in rows:
        cells = [row["label"]]
        for key, precision in (
            ("mean_committed_mw", 3),
            ("mean_cash_26h_usd", 2),
            ("mean_battery_throughput_26h_mwh", 3),
            ("minimum_service_end_soc", 4),
        ):
            cells.append("$" + tex_number(f"{row[key]:,.{precision}f}") + "$")
        cells += [
            f"{row['recovery_closed']}/{row['planned']}",
            f"{row['qualified']}/{row['planned']}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule\end{tabular}"])


def main() -> None:
    protocol = verify(require_committed=True)
    report = read_json(ROOT / "reports/table1_report.json")
    original = read_json(ROOT / "original_table1.json")
    validate_report(report, protocol)
    require(
        report["raw_audit"]["implementation_sha256"]
        == file_hash(ROOT / "analyze_table1.py"),
        "The raw analysis implementation changed after producing this report",
    )
    for case in report["cases"]:
        require(
            file_hash(ROOT / case["episode_path"]) == case["episode_sha256"],
            f"Episode receipt changed: {case['episode_path']}",
        )
    require(
        report["complete_service"] == report["planned"]
        and all(case["recovery_closed"] for case in report["cases"]),
        "This completed-study narrative requires full service "
        "and recovery for every case",
    )
    failures = [case for case in report["cases"] if not case["qualified"]]
    require(
        len(failures) == 4
        and all(
            case["policy"] == "sac_macro+sac"
            and [key for key, passed in case["checks"].items() if not passed]
            == ["service_end_soc"]
            for case in failures
        ),
        "The frozen narrative's four service-SOC failures "
        "no longer match the audited data",
    )
    tradeoffs = collect_tradeoffs(report, protocol)
    require(
        tradeoffs["high_total_accepted"] == 0,
        "The frozen all-rejected-bids finding no longer matches the traces",
    )
    write_json(ROOT / "reports/observed_tradeoffs.json", tradeoffs)
    corrected = display_panel(report["rows"], "corrected")
    archived = display_panel(original["rows"], "archived")
    supplemental = supplemental_rows(report)
    registration_commit = subprocess.check_output(
        [
            "git",
            "--no-pager",
            "log",
            "-1",
            "--format=%H",
            "--",
            str((ROOT / "registration.json").relative_to(PROJECT)),
        ],
        cwd=PROJECT,
        text=True,
    ).strip()
    commands = {
        "RegisteredCases": str(report["planned"]),
        "CompletedCases": str(report["complete_service"]),
        "QualifiedCases": str(report["qualified_supplemental"]),
        "RawChecks": f"{report['raw_audit']['scalar_checks']:,}",
        "RawTraceFiles": str(report["raw_audit"]["trace_files"]),
        "SACPIDRMSEGainPct": f"{tradeoffs['rmse_reduction_percent']:.1f}",
        "SACPIDRevenueDelta": f"{tradeoffs['revenue_delta_usd']:.2f}",
        "SACPIDAgeIncreasePct": f"{tradeoffs['aging_increase_percent']:.1f}",
        "SACPIDCashDelta": f"{tradeoffs['cash26h_delta_usd']:.2f}",
        "SACPIDRMSEWins": str(
            sum(
                row["rmse_delta_sac_minus_pid_kw"] < 0
                for row in tradeoffs["paired_low_sac_vs_pid"]
            )
        ),
        "HighBidWindows": str(tradeoffs["high_total_windows"]),
        "HighAcceptedBids": str(tradeoffs["high_total_accepted"]),
        "HighMeanBidPrice": format(
            statistics.fmean(
                row["mean_bid_price_usd_mwh"] for row in tradeoffs["high_macro_bidding"]
            ),
            ".2f",
        ),
        "HighMeanPostedPrice": format(
            statistics.fmean(
                row["mean_posted_rmcp_usd_mwh"]
                for row in tradeoffs["high_macro_bidding"]
            ),
            ".2f",
        ),
        "RegistrationID": protocol["registration_id"],
        "RegistrationCommit": registration_commit,
        "BaselineCommit": protocol["source_baseline_commit"],
        "LowModelHash": protocol["models"]["low"]["sha256"],
        "HighModelHash": protocol["models"]["high"]["sha256"],
        "ArchiveHash": original["source_archive_sha256"],
        "ArchiveMemberHash": original["camera_ready_member_sha256"],
        "LowSelectedSteps": f"{protocol['models']['low']['selected_timesteps']:,}",
        "HighSelectedSteps": f"{protocol['models']['high']['selected_timesteps']:,}",
    }
    tex = [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in commands.items()]
    for name, table in (
        ("CorrectedServiceTable", tex_table(corrected, corrected=True)),
        ("ArchivedServiceTable", tex_table(archived, corrected=False)),
        ("SupplementalTable", supplemental_tex(supplemental)),
    ):
        tex.append(rf"\newcommand{{\{name}}}{{%" + "\n" + table + "\n}")
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    (PUBLICATION / "table1_values.tex").write_text("\n".join(tex) + "\n")
    source_paths = [
        "registration.json",
        "original_table1.json",
        "analyze_table1.py",
        "render_table1.py",
        "ORIGINAL_SAC_AUDIT.md",
        "reports/table1_report.json",
        "reports/raw_trace_inventory.csv",
        "reports/observed_tradeoffs.json",
        "reports/training_configuration_comparison.json",
        "publication/table1_corrected_comparison.tex",
        "publication/table1_values.tex",
    ]
    manifest = {
        "schema": "table1-comparison-display-v1",
        "registration_id": protocol["registration_id"],
        "registration_commit": registration_commit,
        "source_baseline_commit": protocol["source_baseline_commit"],
        "source_sha256": {path: file_hash(ROOT / path) for path in source_paths},
        "models": protocol["models"],
        "panel": protocol["panel"],
        "statistic": report["statistic"],
        "precision": PRECISION,
        "corrected": corrected,
        "archived": archived,
        "supplemental": supplemental,
        "not_rerun": protocol["missing_rows"],
        "numeric_commands": commands,
        "training_performed": False,
        "llm_calls_performed": False,
        "post_result_checkpoint_selection": False,
        "legacy_corrected_reward_meanings_are_equal": False,
        "exact_archived_producing_cases_recovered": False,
    }
    write_json(PUBLICATION / "table1_values.json", manifest)
    with (PUBLICATION / "table1_cells.csv").open("w", newline="") as stream:
        fields = (
            "panel",
            "policy",
            "metric",
            "mean",
            "sd",
            "display",
            "complete_service",
            "qualified",
            "diagnostic",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for panel in ("corrected", "archived"):
            for row in manifest[panel]:
                for key in METRICS:
                    metric = row["metrics"][key]
                    writer.writerow(
                        {
                            "panel": panel,
                            "policy": row["policy"],
                            "metric": key,
                            "mean": metric["mean"],
                            "sd": metric["sd"],
                            "display": metric["display"],
                            "complete_service": row.get("complete_service"),
                            "qualified": row.get("qualified"),
                            "diagnostic": row.get("diagnostic_not_admissible"),
                        }
                    )
    print(
        json.dumps(
            {
                "corrected_rows": len(corrected),
                "archived_rows": len(archived),
                "numeric_metric_cells": sum(
                    metric["mean"] is not None
                    for row in corrected + archived
                    for metric in row["metrics"].values()
                ),
                "output": str(PUBLICATION),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
