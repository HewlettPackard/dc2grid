"""Numeric-view tests without manuscript source, original TeX or PDF access."""

import copy
import csv
import hashlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rebuild_numeric_tables import (
    build_parts,
    gate_metadata,
    headers,
    rebuild,
    tex_escape,
    verify_views,
)
from validate_table_provenance import (
    APPROVED_INPUTS,
    Sources,
    ValidationError,
    Validator,
)


class RendererUnitTests(unittest.TestCase):
    def test_gate_erratum_is_explicit(self):
        table = {"id": "Q5"}
        row = {"rule_pass": [26, 30], "full_residual_pass": [30, 30]}
        gates = gate_metadata(table, row)
        self.assertFalse(gates["rule"]["paid_service_floor_enforced"])
        self.assertTrue(gates["full_residual"]["paid_service_floor_enforced"])
        self.assertEqual(gates["full_residual"]["paid_service_floor"], 0.9)
        self.assertEqual(gates["rule"]["qualified_planned"], [26, 30])
        self.assertFalse(gates["full_residual"]["new_admission_test"])

    def test_tex_labels_are_escaped(self):
        self.assertEqual(tex_escape("LFR (%) & a_b"), r"LFR (\%) \& a\_b")

    def test_mixed_unit_columns_do_not_claim_one_unit(self):
        part = {
            "label_headers": ["Quantity"],
            "rows": [
                {
                    "columns": [{"name": "Rule", "unit": unit}],
                    "cells": [["1.00", "0.10"]],
                    "pass": None,
                }
                for unit in ("MW", "percent")
            ],
        }
        self.assertEqual(headers(part)[1], "Rule; mean +/- SD")


@unittest.skipUnless(
    os.environ.get("DC2GRID_RELEASE_ROOT"),
    "Set DC2GRID_RELEASE_ROOT for numeric-only integration tests",
)
class RendererEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parent
        cls.root = Path(os.environ["DC2GRID_RELEASE_ROOT"])
        cls.mapping = json.loads((directory / "table_row_provenance.json").read_text())
        cls.contracts = json.loads(
            (directory / "controller_protocol_provenance.json").read_text()
        )
        cls.acceptance = json.loads(
            (directory / "provenance_acceptance.json").read_text()
        )
        cls.loaded_sources = Sources(cls.root, cls.mapping, check_hashes=True)
        for alias in sorted(APPROVED_INPUTS - {"tex"}):
            cls.loaded_sources.read(alias)

    def sources(self):
        source = Sources(self.root, self.mapping, check_hashes=True)
        source.cache = dict(self.loaded_sources.cache)
        source.receipts = copy.deepcopy(self.loaded_sources.receipts)
        return source

    def test_complete_rebuild_without_any_source_tex_or_extra_reads(self):
        source = self.sources()
        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("No extra input reads allowed"),
        ):
            outputs, metadata = rebuild(
                self.mapping,
                self.contracts,
                self.acceptance,
                source,
                "provenance/numeric_tables",
            )
        self.assertEqual(metadata["validation"]["status"], "passed")
        self.assertEqual(
            metadata["validation"]["counts"]["recorded_source_byte_hashes_compared"], 13
        )
        self.assertEqual(len(list(csv.DictReader(io.StringIO(outputs["csv"])))), 118)
        self.assertEqual(outputs["tex"].count(r"\begin{tabular}"), 13)
        self.assertNotIn("Destressing the Grid", outputs["tex"])
        self.assertNotIn(r"\subsection", outputs["tex"])
        self.assertFalse(metadata["frozen_pdf_view"]["regenerated"])
        self.assertFalse(metadata["input_datasets_required_or_read"])
        self.assertEqual(metadata["numeric_entries_per_view"], 1026)
        for kind in ("csv", "html", "tex"):
            self.assertEqual(
                hashlib.sha256(outputs[kind].encode()).hexdigest(),
                metadata["outputs"][kind]["sha256"],
            )
        q5 = next(row for row in metadata["row_metadata"] if row["id"] == "Q5.pid")
        q6 = next(row for row in metadata["row_metadata"] if row["id"] == "Q6.pid")
        self.assertFalse(q5["gates"]["rule"]["paid_service_floor_enforced"])
        self.assertEqual(q6["gates"]["paid_service_floor"], 0.9)

    def test_preexisting_tex_is_rejected(self):
        source = self.sources()
        source.set_generated_tex("old.tex", "old source must not be used")
        with self.assertRaisesRegex(ValidationError, "must not consume"):
            rebuild(
                self.mapping,
                self.contracts,
                self.acceptance,
                source,
                "provenance/numeric_tables",
            )

    def test_csv_and_html_cell_mutations_fail(self):
        source = self.sources()
        outputs, _ = rebuild(
            self.mapping,
            self.contracts,
            self.acceptance,
            source,
            "provenance/numeric_tables",
        )
        validator = Validator(self.mapping, self.contracts, self.acceptance, source)
        validator.metadata()
        validator.main_tables()
        validator.other_tables()
        parts = build_parts(validator)
        with self.assertRaisesRegex(ValidationError, "CSV numeric mismatch"):
            verify_views(
                parts,
                outputs["csv"].replace("3796.1", "3797.1", 1),
                outputs["html"],
                validator.expected_tex,
            )
        with self.assertRaisesRegex(ValidationError, "HTML numeric mismatch"):
            verify_views(
                parts,
                outputs["csv"],
                outputs["html"].replace("3796.1", "3797.1", 1),
                validator.expected_tex,
            )


if __name__ == "__main__":
    unittest.main()
