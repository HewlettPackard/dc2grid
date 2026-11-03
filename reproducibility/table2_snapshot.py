"""Rebuild the updated Table 2 subset and its exact checkpoint/replay inventory."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import sys
from pathlib import Path

from .runtime import (
    file_hash,
    inside,
    limit_threads,
    new_output,
    read_json,
    verify_files,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BASE = "246d60e49e2478ededba81e39e60c0141fec41fa"
TABLE_IDS = ("T2a", "T2b")
METADATA_FILES = (
    "provenance/rebuild_numeric_tables.py",
    "provenance/validate_table_provenance.py",
    "provenance/table_row_provenance.json",
    "provenance/controller_protocol_provenance.json",
    "provenance/provenance_acceptance.json",
    "reproducibility/tracking/inventory.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numeric_module():
    directory = ROOT / "provenance"
    sys.path.insert(0, str(directory))
    module = importlib.import_module("rebuild_numeric_tables")
    require(
        Path(module.__file__).resolve() == directory / "rebuild_numeric_tables.py",
        "Numeric renderer came from a different checkout",
    )
    return module


def verify_selected_views(canonical, parts, csv_text, html_text, expected) -> int:
    ids = [row["id"] for part in parts for row in part["rows"]]
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    require(
        len(ids) == 18 and [row["row_id"] for row in csv_rows] == ids,
        "Table 2 must contain exactly nine rows in each of two parts",
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
        require(cells == expected[row["row_id"]], "Table 2 numeric cell differs")
        count += sum(map(len, cells))
    parser = canonical.NumericHTML()
    parser.feed(html_text)
    require(list(parser.rows) == ids, "Table 2 HTML row inventory differs")
    require(
        all(parser.rows[key] == expected[key] for key in ids),
        "Table 2 HTML numeric values differ",
    )
    return count


def build() -> dict[str, str]:
    verify_files(ROOT, list(METADATA_FILES))
    canonical = numeric_module()
    mapping = read_json(ROOT / METADATA_FILES[2])
    contracts = read_json(ROOT / METADATA_FILES[3])
    acceptance = read_json(ROOT / METADATA_FILES[4])
    sources = canonical.Sources(ROOT, mapping, {}, True)
    validator = canonical.Validator(mapping, contracts, acceptance, sources)
    validator.metadata()
    validator.main_tables()
    validator.other_tables()
    parts = [
        part for part in canonical.build_parts(validator) if part["id"] in TABLE_IDS
    ]
    require([part["id"] for part in parts] == list(TABLE_IDS), "Wrong table subset")
    _, parent_metadata = canonical.rebuild(
        mapping, contracts, acceptance, sources, "provenance/numeric_tables"
    )
    tables = [table for table in mapping["tables"] if table["id"] in TABLE_IDS]
    family_ids = [row["family"] for row in tables[0]["rows"]]
    require(len(family_ids) == len(set(family_ids)) == 9, "Wrong Table 2 families")
    require(
        [row["family"] for row in tables[1]["rows"]] == family_ids,
        "Table 2 parts use different families",
    )
    families = {key: contracts["controller_families"][key] for key in family_ids}
    model_ids = {model for family in families.values() for model in family["model_ids"]}
    models = [
        dict(model) for model in contracts["model_catalog"] if model["id"] in model_ids
    ]
    require(len(models) == len(model_ids) == 17, "Expected 17 selected checkpoints")
    for model in models:
        path = inside(ROOT, model["path"])
        require(file_hash(path) == model["sha256"], f"Wrong model: {model['id']}")
        model["bytes"] = path.stat().st_size
        model["download_url"] = (
            "https://raw.githubusercontent.com/HewlettPackard/dc2grid/"
            f"{PUBLIC_BASE}/{model['path']}"
        )
    inventory = read_json(ROOT / METADATA_FILES[5])
    panel = contracts["panels"]["original_peaks"]
    requests = []
    replay_policies = []
    for family_id, family in families.items():
        snapshot = (
            "value_peak_regression"
            if family_id in ("union_original", "union_conditioned")
            else "table2_regression"
        )
        for model_id in family["model_ids"] or [None]:
            policy = family["low_policy_id_template"].format(
                macro="rule", model_id=model_id
            )
            require(
                policy in inventory["panels"][snapshot]["policy_ids"],
                f"Policy is not in the consumed snapshot: {policy}",
            )
            spec = {
                "family": family_id,
                "model_id": model_id,
                "snapshot": snapshot,
                "policy": policy,
            }
            replay_policies.append(spec)
            for day in panel["start_days_zero_based_2023"]:
                for seed in panel["environment_seeds_per_day"]:
                    for bias in panel["biases"]:
                        requests.append(
                            {
                                **spec,
                                "start_day": day,
                                "seed": seed,
                                "regd_load_bias": bias,
                            }
                        )
    require(len(replay_policies) == 21, "Expected 21 fixed policy instances")
    require(len(requests) == 630, "Expected 630 consumed policy/case combinations")
    outputs = {
        "table2_cells.csv": canonical.render_csv(parts),
        "table2_numeric.html": canonical.render_html(parts),
        "table2_numeric.tex": canonical.render_tex(parts),
        "replay_cases.jsonl": "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in requests
        ),
    }
    count = verify_selected_views(
        canonical,
        parts,
        outputs["table2_cells.csv"],
        outputs["table2_numeric.html"],
        validator.expected_tex,
    )
    snapshot = {
        "schema": "dc2grid-updated-table2-snapshot-v1",
        "scope": "Updated main Table 2, both parts, nine controller families only",
        "source_public_commit": PUBLIC_BASE,
        "source_public_result_tag": "repro/20261102/pid-comparison",
        "source_research_commit": mapping["source_repository_head"],
        "source_stages": [
            "repro/20261102/tracking-baseline",
            "repro/20261102/union-tracking",
        ],
        "pdf": mapping["pdf"],
        "table_ids": list(TABLE_IDS),
        "panel": panel,
        "mean_bias": 0.5,
        "qualification_biases": [0.0, 0.5],
        "aggregation": contracts["aggregation_contracts"]["low_level"],
        "gate": contracts["gate_contracts"]["historical_rule"],
        "families": families,
        "checkpoints": models,
        "replay_policies": replay_policies,
        "tables": tables,
        "numeric_rows": 18,
        "numeric_entries_per_view": count,
        "checkpoint_count": len(models),
        "checkpoint_bytes": sum(model["bytes"] for model in models),
        "policy_case_count": len(requests),
        "parent_numeric_validation": parent_metadata["validation"],
        "source_inputs": parent_metadata["source_inputs"],
        "checkpoint_usage_note": (
            "old_sac is corrected/retrained low seed 1007 at step 250002, "
            "shared with Table 1. No learned macro checkpoint is used in "
            "Table 2; Table 1's learned-macro row additionally needs high1007."
        ),
        "training_performed": False,
        "new_evaluation_performed": False,
        "original_archive_or_input_datasets_included": False,
        "generator": {
            "path": "reproducibility/table2_snapshot.py",
            "sha256": file_hash(Path(__file__)),
        },
    }
    outputs["table2_snapshot.json"] = (
        json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="New directory for rebuilt views")
    mode.add_argument("--verify", action="store_true", help="Check packaged views")
    args = parser.parse_args()
    limit_threads()
    outputs = build()
    if args.verify:
        destination = ROOT / "reproducibility/table2"
        for name, text in outputs.items():
            path = destination / name
            require(
                path.is_file() and path.read_bytes() == text.encode("utf-8"),
                f"Packaged Table 2 view differs: {name}",
            )
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        destination = new_output(output, ROOT)
        for name, text in outputs.items():
            (destination / name).write_bytes(text.encode("utf-8"))
    snapshot = json.loads(outputs["table2_snapshot.json"])
    print(
        json.dumps(
            {
                "verified": True,
                "tables": snapshot["table_ids"],
                "families": 9,
                "checkpoints": snapshot["checkpoint_count"],
                "policy_cases": snapshot["policy_case_count"],
                "numeric_entries": snapshot["numeric_entries_per_view"],
                "directory": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
