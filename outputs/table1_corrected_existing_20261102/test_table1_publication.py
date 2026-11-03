import copy
import math

import pandas as pd
import pytest
from render_table1 import (
    CONTRACT_COLUMNS,
    METRICS,
    ROOT,
    display_panel,
    displayed_cell,
    read_json,
    require_identical_contracts,
    tex_table,
    validate_report,
)
from verify_table1_publication import numeric_tokens, verify_panel


@pytest.fixture
def reports():
    return (
        read_json(ROOT / "reports/table1_report.json"),
        read_json(ROOT / "registration.json"),
        read_json(ROOT / "original_table1.json"),
    )


def test_all_35_cases_reproduce_means_and_sample_sd(reports):
    report, protocol, _ = reports
    validate_report(report, protocol)
    assert report["complete_service"] == 35
    assert report["qualified_supplemental"] == 31


@pytest.mark.parametrize("statistic", ["mean", "sd"])
def test_changed_aggregate_is_rejected(reports, statistic):
    report, protocol, _ = reports
    tampered = copy.deepcopy(report)
    tampered["rows"][0]["metrics"]["tracking_rmse_kw"][statistic] += 1
    with pytest.raises(ValueError, match="all-case recomputation"):
        validate_report(tampered, protocol)


def test_duplicate_case_cannot_replace_a_missing_seed(reports):
    report, protocol, _ = reports
    report["cases"][-1] = report["cases"][-2]
    with pytest.raises(ValueError, match="Missing, duplicate"):
        validate_report(report, protocol)


def test_corrected_and_archived_panels_are_kept_separate(reports):
    report, _, original = reports
    corrected = display_panel(report["rows"], "corrected")
    archived = display_panel(original["rows"], "archived")
    assert len(corrected) == 7
    assert len(archived) == 9
    assert not {"ppo_macro+sac", "llm_macro+sac"} & {row["policy"] for row in corrected}
    assert {row["policy"] for row in archived} - {
        row["policy"] for row in corrected
    } == {
        "ppo_macro+sac",
        "llm_macro+sac",
    }
    assert (
        sum(
            cell["mean"] is not None
            for row in corrected + archived
            for cell in row["metrics"].values()
        )
        == 93
    )


def test_displayed_precision_and_historical_sd_are_exact(reports):
    report, _, original = reports
    corrected = {
        row["policy"]: row for row in display_panel(report["rows"], "corrected")
    }
    archived = {
        row["policy"]: row for row in display_panel(original["rows"], "archived")
    }
    assert (
        corrected["rule_macro+sac"]["metrics"]["tracking_rmse_kw"]["display"]
        == "272.6 +/- 32.9"
    )
    assert (
        corrected["rule_macro+pid"]["metrics"]["tracking_rmse_kw"]["display"]
        == "394.7 +/- 230.8"
    )
    assert (
        archived["sac_macro+sac"]["metrics"]["tracking_rmse_kw"]["display"]
        == "780 +/- 626"
    )
    assert archived["rule_macro+none"]["metrics"]["bess_aging_x1e4"]["display"] == "---"


def test_incomplete_qualification_is_visible_without_filtering(reports):
    report, _, _ = reports
    rows = display_panel(report["rows"], "corrected")
    high = next(row for row in rows if row["policy"] == "sac_macro+sac")
    assert high["complete_service"] == 5
    assert high["qualified"] == 1
    assert high["diagnostic_not_admissible"]
    assert all(high["metrics"][key]["mean"] is not None for key in METRICS)
    tex = tex_table(rows, corrected=True)
    assert "SAC macro + SAC$^\\dagger$" in tex
    assert "5/5 & 1/5" in tex
    assert "\\numcell{272.6}{32.9}" in tex


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_values_are_not_formatted(value):
    with pytest.raises(ValueError, match="non-finite"):
        displayed_cell({"mean": value, "sd": 0.0}, 2)


def test_half_missing_metric_is_rejected():
    with pytest.raises(ValueError, match="one missing"):
        displayed_cell({"mean": None, "sd": 0.0}, 2)


def test_actual_contract_comparison_detects_a_changed_rate():
    left = pd.DataFrame({key: [1.0, 2.0] for key in CONTRACT_COLUMNS})
    right = left.copy()
    require_identical_contracts(left, right)
    right.loc[1, "rate"] = 3.0
    with pytest.raises(ValueError, match="contracts differ: rate"):
        require_identical_contracts(left, right)


def test_pdf_tokens_handle_minus_signs_counts_and_missing_values():
    assert numeric_tokens("\N{MINUS SIGN}53,657.43 5/5 \N{EM DASH}") == [
        "-53,657.43",
        "5/5",
        "---",
    ]


def pdf_pid_row(report):
    row = next(
        row
        for row in display_panel(report["rows"], "corrected")
        if row["policy"] == "rule_macro+pid"
    )
    means = [row["metrics"][key]["mean_display"] for key in METRICS]
    sd = [f"\N{PLUS-MINUS SIGN}{row['metrics'][key]['sd_display']}" for key in METRICS]
    page = row["label"] + "  " + "  ".join(means + ["5/5", "5/5"]) + "\n"
    return row, page + "  ".join(sd)


def test_pdf_parser_checks_all_means_and_sds(reports):
    row, page = pdf_pid_row(reports[0])
    assert verify_panel(page, [row], corrected=True) == 6


def test_pdf_wrong_sd_is_rejected(reports):
    row, page = pdf_pid_row(reports[0])
    with pytest.raises(ValueError, match="sample SDs differ"):
        verify_panel(page.replace("230.8", "230.9"), [row], corrected=True)


def test_pdf_wrong_completion_count_is_rejected(reports):
    row, page = pdf_pid_row(reports[0])
    with pytest.raises(ValueError, match="means/counts differ"):
        verify_panel(page.replace("5/5", "4/5", 1), [row], corrected=True)


def test_supplemental_cash_must_also_use_all_planned_cases(reports):
    report, protocol, _ = reports
    report["rows"][-1]["mean_cash_26h_usd"] += 1.0
    with pytest.raises(ValueError, match="all-case recomputation"):
        validate_report(report, protocol)
