"""Verify PDF/CSV values, source identities and preserved earlier artifacts."""

from __future__ import annotations

import csv
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_table1 import PROJECT, verify
from render_table1 import (
    METRICS,
    PUBLICATION,
    ROOT,
    read_json,
    require,
    validate_report,
    write_json,
)
from table1_eval import file_hash


def normalize(text: str) -> str:
    return text.replace("\N{MINUS SIGN}", "-").replace("\N{EM DASH}", "---")


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"---|-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", normalize(text))


def row_position(lines: list[str], label: str) -> int:
    candidates = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*" + re.escape(label) + r"(?:\s|\N{DAGGER})", line)
    ]
    require(len(candidates) == 1, f"PDF row is missing or ambiguous: {label}")
    return candidates[0]


def verify_panel(page: str, rows: list[dict], *, corrected: bool) -> int:
    lines = page.splitlines()
    checked = 0
    for row in rows:
        index = row_position(lines, row["label"])
        line = lines[index].strip()
        suffix = line[len(row["label"]) :]
        expected_means = [
            row["metrics"][key].get("mean_display", "---") for key in METRICS
        ]
        if corrected:
            expected_means += [
                f"{row['complete_service']}/{row['planned']}",
                f"{row['qualified']}/{row['planned']}",
            ]
            require(
                ("\N{DAGGER}" in suffix) == row["diagnostic_not_admissible"],
                f"PDF qualification marker differs: {row['policy']}",
            )
        require(
            numeric_tokens(suffix) == expected_means,
            f"PDF means/counts differ: {row['policy']}",
        )
        expected_sd = [
            row["metrics"][key]["sd_display"]
            for key in METRICS
            if row["metrics"][key]["sd"] is not None
        ]
        require(
            index + 1 < len(lines) and numeric_tokens(lines[index + 1]) == expected_sd,
            f"PDF sample SDs differ: {row['policy']}",
        )
        checked += len(expected_sd)
    return checked


def verify_supplemental(page: str, rows: list[dict]) -> None:
    lines = page.splitlines()
    for row in rows:
        line = lines[row_position(lines, row["label"])].strip()
        expected = [
            f"{row['mean_committed_mw']:,.3f}",
            f"{row['mean_cash_26h_usd']:,.2f}",
            f"{row['mean_battery_throughput_26h_mwh']:,.3f}",
            f"{row['minimum_service_end_soc']:,.4f}",
            f"{row['recovery_closed']}/{row['planned']}",
            f"{row['qualified']}/{row['planned']}",
        ]
        require(
            numeric_tokens(line[len(row["label"]) :]) == expected,
            f"PDF supplemental values differ: {row['policy']}",
        )


def verify_csv(manifest: dict) -> int:
    with (PUBLICATION / "table1_cells.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    expected = [
        (panel, row, key)
        for panel in ("corrected", "archived")
        for row in manifest[panel]
        for key in METRICS
    ]
    require(len(actual) == len(expected), "CSV metric-cell count differs")
    for entry, (panel, row, key) in zip(actual, expected, strict=True):
        require(
            (entry["panel"], entry["policy"], entry["metric"])
            == (panel, row["policy"], key),
            "CSV cell identity or order differs",
        )
        metric = row["metrics"][key]
        for statistic in ("mean", "sd"):
            observed = None if entry[statistic] == "" else float(entry[statistic])
            require(observed == metric[statistic], "CSV full-precision value differs")
        require(entry["display"] == metric["display"], "CSV display text differs")
        for column, source in (
            ("complete_service", "complete_service"),
            ("qualified", "qualified"),
            ("diagnostic", "diagnostic_not_admissible"),
        ):
            expected_value = "" if source not in row else str(row[source])
            require(entry[column] == expected_value, f"CSV {column} differs")
    return len(actual)


def verify_pdf_bounds(pdf: Path) -> int:
    document = ET.fromstring(
        subprocess.check_output(["pdftotext", "-bbox", str(pdf), "-"], text=True)
    )
    ns = "{http://www.w3.org/1999/xhtml}"
    pages = document.findall(f".//{ns}page")
    require(len(pages) == 4, "Expected the four-page standalone report")
    checked = 0
    for page in pages:
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(f".//{ns}word"):
            left, right = float(word.attrib["xMin"]), float(word.attrib["xMax"])
            top, bottom = float(word.attrib["yMin"]), float(word.attrib["yMax"])
            require(
                44.8 <= left < right <= width - 44.8 and 0 <= top < bottom <= height,
                f"PDF text exceeds the page/margin bounds: {word.text}",
            )
            checked += 1
    require(checked > 1000, "Too little PDF text was inspected")
    return checked


def main() -> None:
    protocol = verify(require_committed=True)
    report = read_json(ROOT / "reports/table1_report.json")
    manifest = read_json(PUBLICATION / "table1_values.json")
    validate_report(report, protocol)
    require(
        manifest["registration_id"] == protocol["registration_id"]
        and manifest["models"] == protocol["models"]
        and manifest["panel"] == protocol["panel"],
        "Displayed model or panel identity differs from registration",
    )
    for relative, expected in manifest["source_sha256"].items():
        require(
            file_hash(ROOT / relative) == expected, f"Report input changed: {relative}"
        )
    pdf = PUBLICATION / "table1_corrected_comparison.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    pages = [page for page in text.split("\f") if page.strip()]
    require(len(pages) == 4, "Expected exactly four PDF pages")
    cells = verify_panel(pages[0], manifest["corrected"], corrected=True)
    cells += verify_panel(pages[1], manifest["archived"], corrected=False)
    verify_supplemental(pages[2], manifest["supplemental"])
    require(cells == 93, "Not every nonmissing Table 1 metric cell was verified")
    for value in (
        protocol["registration_id"],
        manifest["registration_commit"],
        *[model["sha256"] for model in protocol["models"].values()],
    ):
        require(value in "".join(pages[3].split()), "A PDF provenance identity differs")
    require("not rerun" in text, "Missing disclosure of the omitted corrected rows")
    csv_cells = verify_csv(manifest)
    words = verify_pdf_bounds(pdf)
    preservation = read_json(ROOT / "reports/preservation_receipt.json")
    for record in preservation["files"]:
        require(
            file_hash(PROJECT / record["path"]) == record["sha256"],
            f"Earlier artifact changed: {record['path']}",
        )
    outputs = [
        "publication/table1_corrected_comparison.pdf",
        "publication/table1_corrected_comparison.tex",
        "publication/table1_values.tex",
        "publication/table1_values.json",
        "publication/table1_cells.csv",
        "ORIGINAL_SAC_AUDIT.md",
        "reports/table1_report.json",
        "reports/observed_tradeoffs.json",
        "reports/preservation_receipt.json",
    ]
    receipt = {
        "status": "passed",
        "registration_id": protocol["registration_id"],
        "pdf_pages": len(pages),
        "nonmissing_mean_and_sample_sd_cells_verified": cells,
        "numeric_statistics_verified": 2 * cells,
        "missing_archival_cells_preserved": 3,
        "csv_metric_cells_verified": csv_cells,
        "supplemental_rows_verified": len(manifest["supplemental"]),
        "pdf_word_bounding_boxes_checked": words,
        "corrected_primary_counts_and_diagnostic_markers_verified": True,
        "models_panel_and_registration_verified": True,
        "prior_archive_and_pdfs_unchanged": True,
        "output_sha256": {path: file_hash(ROOT / path) for path in outputs},
        "verifier_sha256": file_hash(Path(__file__)),
        "training_performed": False,
        "remote_publication_performed": False,
    }
    write_json(ROOT / "reports/publication_validation.json", receipt)
    print(
        f"Verified {cells} PDF mean/SD cells, {csv_cells} CSV cells, "
        f"{len(pages)} pages and {words} word bounding boxes."
    )


if __name__ == "__main__":
    main()
