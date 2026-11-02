#!/usr/bin/env python3
"""Rebuild numeric CSV/HTML/TeX views, never the frozen manuscript/PDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from validate_table_provenance import (
    APPROVED_INPUTS,
    ARCHIVE_NUMERIC_LABELS,
    NUMBER,
    REPRODUCTION_LABELS,
    Sources,
    ValidationError,
    Validator,
    contained_path,
    decode_json,
    metadata_document,
    relative_locator,
    require,
)

FILENAMES = {
    "csv": "pid_numeric_tables.csv",
    "html": "pid_numeric_tables.html",
    "tex": "pid_numeric_tables.tex",
    "metadata": "pid_numeric_tables.metadata.json",
}


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def tex_escape(text):
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(char, char) for char in str(text))


def gate_metadata(table, row):
    def gate(name, counts):
        floor = name in {"historical_full_residual", "economic", "new_pid"}
        return {
            "source_gate_id": name,
            "physical_recovery_enforced": name is not None,
            "paid_service_floor_enforced": floor,
            "paid_service_floor": 0.9 if floor else None,
            "qualified_planned": counts,
            "biases": table.get(
                "gate_biases",
                [0.0, 0.5]
                if table["id"] == "Q5"
                else [row.get("bias", table.get("bias"))],
            ),
            "new_admission_test": False,
        }

    if table["id"] == "Q5":
        return {
            "rule": gate("historical_rule", row["rule_pass"]),
            "full_residual": gate(
                "historical_full_residual", row["full_residual_pass"]
            ),
        }
    return gate(row.get("gate", table.get("gate")), row.get("pass"))


def context(table):
    if table["id"] == "Q5":
        return (
            "rule: physical/recovery; full-residual: physical/recovery "
            "+ paid-service floor 90%"
        )
    if table["id"] == "Q6":
        return (
            "Source-study gates: physical/recovery + paid-service floor 90%; "
            "both biases; no new admission test"
        )
    if table["id"] == "Q1":
        return (
            "Reported legacy values; separate reproduction exception: "
            "PID facility relief only"
        )
    return "; ".join(
        f"{key}={table[key]}"
        for key in ("panel", "macro", "bias", "aggregation")
        if key in table
    )


def build_parts(validator):
    mapping = validator.mapping
    families = validator.contracts["controller_families"]
    main_values = {
        (value["scope"], value["family"], value["metric"]): value
        for value in validator.sources.read("comparison_values")
    }
    parts = []
    for table_index, table in enumerate(mapping["tables"]):
        title = f"Table {table['number']}" + (
            f"({table['part']})" if "part" in table else ""
        )
        part = {
            "id": table["id"],
            "title": title,
            "context": context(table),
            "rows": [],
        }
        for row_index, row in enumerate(table["rows"]):
            gate_column = "value_scope" in table or table["id"] == "Q11"
            numbers = validator.expected_tex[row["id"]]
            cells = numbers[:-1] if gate_column else numbers
            family = row.get("family", table.get("family", ""))
            if table["id"] == "Q1":
                labels = [ARCHIVE_NUMERIC_LABELS[row["id"]]]
                columns = [
                    {"name": name, "unit": row["unit"]}
                    for name in ("Rule", "PID", "Bang-bang", "SAC")
                ]
                refs = [
                    f"provenance/table_row_provenance.json#/tables/{table_index}/rows/{row_index}"
                ]
            elif table["id"] in {"Q2", "Q3"}:
                labels = [REPRODUCTION_LABELS[row["metric"]]]
                columns = [
                    {"name": name, "unit": row["unit"]}
                    for name in ("Earlier mode", "Corrected mode", "Change")
                ]
                refs = [
                    f"pid_reproduction#/summary/{mode}/{row['metric']}"
                    for mode in ("legacy_v0", "physical_v1")
                ]
            else:
                if table["id"] == "Q11":
                    panel = (
                        "New calendar" if row["panel"] == "calendar" else "Known peak"
                    )
                    labels = [
                        f"{panel}/{row['bias']}",
                        "Native" if family == "pid_native" else "Full work",
                    ]
                else:
                    labels = [families[family]["label"]]
                specs = (
                    table["columns"]
                    if table["id"] == "Q5"
                    else mapping["metric_sets"][table["metric_set"]]
                )
                columns = [
                    {"name": spec.get("column", spec.get("name")), "unit": spec["unit"]}
                    for spec in specs
                ]
                refs = [
                    row[key]
                    for key in (
                        "source_ref",
                        "rule_source_ref",
                        "full_residual_source_ref",
                    )
                    if key in row
                ]
                if "cash_reference_ref" in table:
                    refs.append(table["cash_reference_ref"])
            bold = (
                [
                    main_values[(table["value_scope"], family, spec["metric"])]["bold"]
                    for spec in mapping["metric_sets"][table["metric_set"]]
                ]
                if "value_scope" in table
                else [False] * len(cells)
            )
            part["rows"].append(
                {
                    "id": row["id"],
                    "family": family,
                    "labels": labels,
                    "panel": row.get("panel", table.get("panel")),
                    "macro": table.get("macro", "|".join(table.get("macros", []))),
                    "bias": row.get("bias", table.get("bias")),
                    "mean_cases_per_policy": row.get(
                        "cases_per_policy_in_mean",
                        table.get(
                            "cases_per_policy_in_mean",
                            table.get(
                                "cases_per_mode_in_mean",
                                table.get("reported_case_count"),
                            ),
                        ),
                    ),
                    "fixed_policy_count": (len(families[family]["model_ids"]) or 1)
                    if family
                    else None,
                    "columns": columns,
                    "cells": cells,
                    "bold": bold,
                    "pass": numbers[-1] if gate_column else None,
                    "diagnostic_marker": bool(row.get("diagnostic", False)),
                    "gate_metadata": gate_metadata(table, row),
                    "source_refs": refs,
                    "evidence_status": "reported_legacy"
                    if table["id"] == "Q1"
                    else "complete_cases_including_qualification_failures",
                }
            )
        part["label_headers"] = (
            ["Panel / bias", "PID variant"]
            if table["id"] == "Q11"
            else ["Quantity" if table["id"] in {"Q1", "Q2", "Q3"} else "Controller"]
        )
        parts.append(part)
    return parts


def render_csv(parts):
    fields = [
        "table_id",
        "row_id",
        "family",
        "label_1",
        "label_2",
        "panel",
        "macro",
        "bias",
        "mean_cases_per_policy",
        "fixed_policy_count",
        "evidence_status",
        "gate_metadata",
        "source_refs",
    ]
    for index in range(1, 8):
        fields.extend(
            f"c{index}_{suffix}" for suffix in ("name", "unit", "value", "sd")
        )
    fields.extend(("qualified", "planned"))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for part in parts:
        for row in part["rows"]:
            record = {
                "table_id": part["id"],
                "row_id": row["id"],
                "family": row["family"],
                "label_1": row["labels"][0],
                "label_2": row["labels"][1] if len(row["labels"]) == 2 else "",
                **{
                    key: row[key]
                    for key in (
                        "panel",
                        "macro",
                        "bias",
                        "mean_cases_per_policy",
                        "fixed_policy_count",
                        "evidence_status",
                    )
                },
                "gate_metadata": compact(row["gate_metadata"]),
                "source_refs": compact(row["source_refs"]),
                "qualified": row["pass"][0] if row["pass"] else "",
                "planned": row["pass"][1] if row["pass"] else "",
            }
            for index, (column, values) in enumerate(
                zip(row["columns"], row["cells"]), 1
            ):
                record.update(
                    {
                        f"c{index}_name": column["name"],
                        f"c{index}_unit": column["unit"],
                        f"c{index}_value": values[0],
                        f"c{index}_sd": values[1] if len(values) == 2 else "",
                    }
                )
            writer.writerow(record)
    return output.getvalue()


def headers(part):
    first = part["rows"][0]
    return (
        part["label_headers"]
        + [
            column["name"]
            + (
                f" [{column['unit']}]"
                if len({row["columns"][index]["unit"] for row in part["rows"]}) == 1
                else ""
            )
            + ("; mean +/- SD" if len(values) == 2 else "")
            for index, (column, values) in enumerate(
                zip(first["columns"], first["cells"])
            )
        ]
        + (["Pass / planned"] if first["pass"] else [])
    )


def render_tex(parts):
    lines = [
        "% New numeric table fragment; requires booktabs and graphicx.",
        "% Not manuscript source, original layout, or a nine-page PDF regeneration.",
    ]
    for part in parts:
        lines += [
            "",
            r"\par\medskip\noindent\textbf{" + tex_escape(part["title"]) + "}",
            r"\par\noindent{\small " + tex_escape(part["context"]) + "}",
            r"\par\noindent\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{"
            + "l" * len(part["label_headers"])
            + "r" * (len(headers(part)) - len(part["label_headers"]))
            + r"}\toprule",
            " & ".join(tex_escape(value) for value in headers(part)) + r" \\\midrule",
        ]
        for row in part["rows"]:
            columns = [tex_escape(label) for label in row["labels"]]
            if row["diagnostic_marker"]:
                columns[0] += r"$^\dagger$"
            for values, bold in zip(row["cells"], row["bold"]):
                value = r"\mathbf{" + values[0] + "}" if bold else values[0]
                columns.append(
                    "$"
                    + value
                    + (r" \pm " + values[1] if len(values) == 2 else "")
                    + "$"
                )
            if row["pass"]:
                columns.append("/".join(row["pass"]))
            lines.append(" & ".join(columns) + r" \\")
        lines += [r"\bottomrule\end{tabular}%", "}"]
    return "\n".join(lines) + "\n"


def render_html(parts):
    escape = html.escape
    lines = [
        (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>PID numeric tables</title>"
        ),
        (
            "<style>body{font:14px sans-serif;margin:2em}"
            "section{overflow:auto;margin-bottom:2em}"
            "table{border-collapse:collapse}th,td{padding:.35em;"
            "border-bottom:1px solid #ccc}"
            "td.numeric{text-align:right;white-space:nowrap}"
            "caption{text-align:left;font-weight:bold}"
            "code{font-size:.9em}</style><body><h1>PID numeric tables</h1>"
        ),
        (
            "<p>View: numeric only. Frozen nine-page PDF: unchanged; "
            "no layout regeneration.</p>"
        ),
    ]
    for part in parts:
        lines += [
            "<section><p><code>" + escape(part["context"]) + "</code></p>",
            "<table><caption>"
            + escape(part["title"])
            + "</caption><thead><tr>"
            + "".join("<th>" + escape(value) + "</th>" for value in headers(part))
            + "</tr></thead><tbody>",
        ]
        for row in part["rows"]:
            lines.append(
                '<tr data-row-id="'
                + escape(row["id"], quote=True)
                + '" data-gates="'
                + escape(compact(row["gate_metadata"]), quote=True)
                + '">'
            )
            lines.extend(
                "<td>"
                + escape(label)
                + (" †" if index == 0 and row["diagnostic_marker"] else "")
                + "</td>"
                for index, label in enumerate(row["labels"])
            )
            for index, (values, bold) in enumerate(zip(row["cells"], row["bold"])):
                mean = (
                    "<strong>" + escape(values[0]) + "</strong>"
                    if bold
                    else escape(values[0])
                )
                text = mean + (" ± " + escape(values[1]) if len(values) == 2 else "")
                lines.append(
                    f'<td class="numeric" data-cell-index="{index}">{text}</td>'
                )
            if row["pass"]:
                lines.append(
                    f'<td class="numeric" data-cell-index="{len(row["cells"])}">'
                    + "/".join(row["pass"])
                    + "</td>"
                )
            lines.append("</tr>")
        lines.append("</tbody></table></section>")
    return "\n".join(lines) + "</body></html>\n"


class NumericHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = {}, None, None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr" and "data-row-id" in attributes:
            self.row = attributes["data-row-id"]
            require(self.row not in self.rows, "Duplicate rendered HTML row")
            self.rows[self.row] = []
        elif tag == "td" and self.row is not None and "data-cell-index" in attributes:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.rows[self.row].append(NUMBER.findall("".join(self.cell)))
            self.cell = None
        elif tag == "tr":
            self.row = None


def verify_views(parts, csv_text, html_text, expected):
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    row_ids = [row["id"] for part in parts for row in part["rows"]]
    require(
        len(row_ids) == 118 and [row["row_id"] for row in csv_rows] == row_ids,
        "CSV row inventory mismatch",
    )
    count = 0
    for row in csv_rows:
        cells = [
            [row[f"c{index}_value"]]
            + ([row[f"c{index}_sd"]] if row[f"c{index}_sd"] else [])
            for index in range(1, 8)
            if row[f"c{index}_value"]
        ]
        if row["qualified"]:
            cells.append([row["qualified"], row["planned"]])
        require(
            cells == expected[row["row_id"]], f"{row['row_id']}: CSV numeric mismatch"
        )
        count += sum(len(values) for values in cells)
    parser = NumericHTML()
    parser.feed(html_text)
    require(list(parser.rows) == row_ids, "HTML row inventory mismatch")
    require(
        all(parser.rows[key] == expected[key] for key in row_ids),
        "HTML numeric mismatch",
    )
    require(count == 1026, "Expected 1,026 numeric entries in each numeric view")


def rebuild(mapping, contracts, acceptance, sources, public_dir):
    relative_locator(public_dir)
    require(
        "tex" not in sources.cache and "tex" not in sources.overrides,
        "Numeric rebuild must not consume any preexisting TeX",
    )
    prepare = Validator(mapping, contracts, acceptance, sources)
    prepare.metadata()
    prepare.main_tables()
    prepare.other_tables()
    parts = build_parts(prepare)
    outputs = {
        "csv": render_csv(parts),
        "html": render_html(parts),
        "tex": render_tex(parts),
    }
    verify_views(parts, outputs["csv"], outputs["html"], prepare.expected_tex)
    tex_locator = str(PurePosixPath(public_dir) / FILENAMES["tex"])
    sources.set_generated_tex(tex_locator, outputs["tex"])
    validation = Validator(mapping, contracts, acceptance, sources).run()
    require(set(sources.cache) == APPROVED_INPUTS, "Unexpected renderer dependency")
    metadata = {
        "schema": "dc2grid-numeric-table-rebuild-v1",
        "entrypoint": "provenance/rebuild_numeric_tables.py",
        "source_repository_head": mapping["source_repository_head"],
        "numbered_tables": 12,
        "table_parts": 13,
        "rows": 118,
        "numeric_entries_per_view": 1026,
        "main_comparison_cells": 528,
        "csv_numeric_fields": (
            "c1_value/c1_sd through c7_value/c7_sd, then qualified/planned; "
            "metadata fields are not extra table cells"
        ),
        "source_inputs": [
            receipt for receipt in validation["inputs"] if receipt["alias"] != "tex"
        ],
        "outputs": {
            kind: {
                "path": str(PurePosixPath(public_dir) / FILENAMES[kind]),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            for kind, text in outputs.items()
        },
        "validation": {
            "status": validation["status"],
            "counts": validation["counts"],
            "numeric_source_byte_hashes_enforced": sources.check_hashes,
        },
        "frozen_pdf_view": {
            "sha256": mapping["pdf"]["sha256"],
            "pages": 9,
            "read_or_written": False,
            "regenerated": False,
            "layout_or_byte_regeneration_claimed": False,
        },
        "input_datasets_required_or_read": False,
        "original_TeX_or_comparison_data_blob_read": False,
        "paper_prose_in_outputs": False,
        "scope": {
            "public_Git_input_datasets": "excluded",
            "export_ancestry": "strict_orphan_required",
            "raw_evidence": "local_compressed_parts_for_later_user_OneDrive_upload",
            "public_artifact_url_status": "pending",
            "public_artifact_urls": [],
        },
        "gate_erratum": {
            "id": "Q5_Q6_HISTORICAL_FULL_RESIDUAL_FLOOR",
            "Q5": (
                "Rule side excludes the paid-service floor; full-residual side "
                "includes the 90% floor. Paired source gate counts are retained below."
            ),
            "Q6": (
                "The first five historical low-level full-residual rows already "
                "include the 90% floor, as do the economic source rows."
            ),
            "new_admission_test": False,
            "scientific_value_changes": 0,
        },
        "row_metadata": [
            {
                "id": row["id"],
                "family": row["family"],
                "panel": row["panel"],
                "macro": row["macro"],
                "bias": row["bias"],
                "gates": row["gate_metadata"],
                "source_refs": row["source_refs"],
                "mean_cases_per_policy": row["mean_cases_per_policy"],
                "fixed_policy_count": row["fixed_policy_count"],
                "evidence_status": row["evidence_status"],
            }
            for part in parts
            for row in part["rows"]
        ],
        "limits": validation["limits"],
    }
    outputs["metadata"] = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    return outputs, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="New numeric-only PID table views; not a manuscript or PDF rebuild."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root containing the 13 approved result JSON inputs",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--input-map",
        help=(
            "Repository-relative JSON alias map; TeX and excluded aliases are forbidden"
        ),
    )
    parser.add_argument(
        "--public-output-dir",
        default="provenance/numeric_tables",
        help="Repository-relative public output locator prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Physical output directory; defaults to root/public-output-dir",
    )
    parser.add_argument("--check-source-hashes", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this renderer's four named outputs",
    )
    args = parser.parse_args(argv)
    try:
        relative_locator(args.public_output_dir)
        mapping = metadata_document(args.metadata_dir, "table_row_provenance.json")
        contracts = metadata_document(
            args.metadata_dir, "controller_protocol_provenance.json"
        )
        acceptance = metadata_document(args.metadata_dir, "provenance_acceptance.json")
        overrides = {}
        if args.input_map:
            overrides = decode_json(
                contained_path(args.root, args.input_map).read_bytes(), "input-map"
            )
            require(
                isinstance(overrides, dict) and "tex" not in overrides,
                "Numeric input-map must not supply TeX",
            )
        sources = Sources(args.root, mapping, overrides, args.check_source_hashes)
        outputs, metadata = rebuild(
            mapping, contracts, acceptance, sources, args.public_output_dir
        )
        destination = args.output_dir or args.root / args.public_output_dir
        for kind, text in outputs.items():
            path = destination / FILENAMES[kind]
            require(not path.is_symlink(), "Refusing an output symlink")
            if path.exists() and not args.overwrite:
                require(
                    path.read_bytes() == text.encode("utf-8"),
                    "Changed output exists; use --overwrite for generated outputs only",
                )
        destination.mkdir(parents=True, exist_ok=True)
        for kind, text in outputs.items():
            (destination / FILENAMES[kind]).write_bytes(text.encode("utf-8"))
        print(
            json.dumps(
                {
                    "status": "passed",
                    "outputs": metadata["outputs"],
                    "metadata": str(
                        PurePosixPath(args.public_output_dir) / FILENAMES["metadata"]
                    ),
                    "rows": 118,
                    "numeric_entries_per_view": 1026,
                    "main_cells": 528,
                    "PDF_regenerated": False,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        message = (
            str(exc)
            if isinstance(exc, ValidationError)
            else (
                "Invalid or unavailable approved evidence/output "
                f"({type(exc).__name__})"
            )
        )
        print(json.dumps({"status": "failed", "error": message}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
