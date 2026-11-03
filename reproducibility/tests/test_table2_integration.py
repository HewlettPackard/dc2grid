"""Table 2 scope, aggregation and native-source binding regression tests."""

import csv
import io
import json
import re
import shutil
import subprocess

import pytest

from reproducibility import table2_snapshot as tables
from reproducibility.runtime import CORE_COMMIT, CORE_SOURCE_AMENDMENTS
from reproducibility.tracking import tracking_release as replay

PDF = (
    tables.ROOT
    / "outputs/pid_comparison_extract_20261102/publication"
    / "pid_progress_comparison.pdf"
)


def test_closed_two_panel_and_twenty_one_policy_inventory():
    index = replay.inventory()
    assert set(index["panels"]) == {"table2_regression", "value_peak_regression"}
    policies = [
        policy for panel in index["panels"].values() for policy in panel["policy_ids"]
    ]
    assert len(policies) == len(set(policies)) == 21
    assert all(
        policy.startswith("rule__") and "nominal" not in policy for policy in policies
    )
    assert len(index["models"]) == 17


def test_replay_uses_native_core_not_a_second_simulator_copy():
    index = replay.inventory()
    assert index["integration_core_commit"] == CORE_COMMIT
    modules = index["runtime"]["module_files"]
    assert modules["c2g_env.env_high_level"] == "c2g_env/env_high_level.py"
    assert modules["c2g_env.env_low_level"] == "c2g_env/env_low_level.py"
    assert modules["baselines.pid_controller"] == "baselines/pid_controller.py"
    assert all("online_stability_pilot" not in name for name in modules.values())
    for path, amendment in CORE_SOURCE_AMENDMENTS.items():
        change = index["source_binding_changes"][path]
        assert change["producer_sha256"] == amendment["producer_sha256"]
        assert change["integration_sha256"] == amendment["integration_sha256"]


@pytest.mark.parametrize(
    "policy", ["rule__nominal", "full_residual__pid", "rule__warm_sac_1100"]
)
def test_non_table2_policies_are_rejected_before_loading_assets(policy):
    with pytest.raises(replay.ReleaseError, match="Table 2 policy roster"):
        replay.prepare(
            tables.ROOT,
            {"snapshot": "table2_regression", "policy_id": policy},
            {"start_day": 207, "seed": 100, "regd_load_bias": 0.5},
        )


def test_other_consumed_panels_are_not_silently_exposed():
    with pytest.raises(replay.ReleaseError, match="Unknown consumed snapshot"):
        replay.prepare(
            tables.ROOT,
            {"snapshot": "value_final_B", "policy_id": "rule__union_conditioned_1100"},
            {"start_day": 12, "seed": 94001, "regd_load_bias": 0.5},
        )


@pytest.mark.parametrize(
    "cell",
    [
        {"start_day": True, "seed": 100, "regd_load_bias": 0.5},
        {"start_day": 207, "seed": -1, "regd_load_bias": 0.5},
        {"start_day": 207, "seed": 100, "regd_load_bias": float("nan")},
    ],
)
def test_invalid_cells_fail_explicitly(cell):
    with pytest.raises(replay.ReleaseError):
        replay.normalize_cell(cell)


def test_family_action_restrictions_remain_distinct():
    assert not ({"old_sac", "pid", "bang_bang", "rule_hardware"} & replay.FULL_WORK)
    assert {
        "cooling_first",
        "bc_only",
        "guided_sac",
        "union_original",
        "union_conditioned",
    } <= replay.FULL_WORK


def test_gate_failures_and_discrete_types_cannot_be_smoothed_away():
    assert not replay.compare_metrics(
        {"publication_qualified": False}, {"publication_qualified": False}
    )
    assert replay.compare_metrics(
        {"publication_qualified": False}, {"publication_qualified": True}
    )
    assert replay.compare_metrics({"steps": 17280}, {"steps": 17280.0})


@pytest.mark.skipif(
    not (
        tables.ROOT
        / "outputs/rl_value_conditioning_20261101/reports/final_episode_metrics.json"
    ).is_file(),
    reason="Fetch Table 2 assets before running frozen evidence tests",
)
def test_all_234_numbers_and_630_selectors_from_complete_cohorts():
    output = tables.build()
    result = json.loads(output["table2_reproduction.json"])
    rows = list(csv.DictReader(io.StringIO(output["table2_cells.csv"])))
    assert len(rows) == 18
    assert result["numeric_entries_per_view"] == 234
    assert result["checkpoint_count"] == 17
    assert result["policy_case_count"] == 630
    assert result["table2_numeric_validation"]["low_aggregate_rows"] == 9
    assert result["table2_numeric_validation"]["low_metric_mean_SD_pairs"] == 162
    requests = [json.loads(line) for line in output["replay_cases.jsonl"].splitlines()]
    assert len(requests) == 630
    assert (
        len(
            {
                (
                    row["snapshot"],
                    row["policy"],
                    row["start_day"],
                    row["seed"],
                    row["regd_load_bias"],
                )
                for row in requests
            }
        )
        == 630
    )
    assert {row["start_day"] for row in requests} == {207, 248, 249}
    assert {row["seed"] for row in requests} == set(range(100, 105))
    assert {row["regd_load_bias"] for row in requests} == {0.0, 0.5}
    pid = next(row for row in rows if row["row_id"] == "T2a.pid")
    assert (pid["qualified"], pid["planned"]) == ("26", "30")


@pytest.mark.skipif(
    shutil.which("pdftotext") is None or not PDF.is_file(),
    reason="Fetch Table 2 assets and supply Poppler for the printed-PDF check",
)
def test_all_234_numbers_are_printed_in_the_correct_pdf_rows():
    text = subprocess.check_output(
        ["pdftotext", "-f", "1", "-l", "1", "-layout", str(PDF), "-"], text=True
    ).replace("\N{MINUS SIGN}", "-")
    with (tables.ROOT / "reproducibility/table2/table2_cells.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    count = 0
    for row in rows:
        label = re.compile(
            "^" + re.escape(row["label_1"]) + r"[\s\N{DAGGER}]*(?=[-+]?\d)"
        )
        matches = [line for line in text.splitlines() if label.match(line)]
        assert len(matches) == 2, row["label_1"]
        suffix = label.sub("", matches[0 if row["table_id"] == "T2a" else 1])
        observed = re.findall(r"[-+]?\d+(?:\.\d+)?", suffix.replace(",", ""))
        expected = [
            row[f"c{index}_{kind}"]
            for index in range(1, 8)
            for kind in ("value", "sd")
            if row[f"c{index}_{kind}"]
        ] + [row["qualified"], row["planned"]]
        assert observed == expected, row["row_id"]
        count += len(expected)
    assert count == 234
