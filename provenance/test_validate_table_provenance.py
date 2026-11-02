"""Unit tests; optional read-only integration tests use DC2GRID_RELEASE_ROOT."""

import copy
import json
import os
import statistics
import unittest
from pathlib import Path
from unittest.mock import patch

from validate_table_provenance import (
    APPROVED_INPUTS,
    Sources,
    ValidationError,
    Validator,
    checkpoint_averaged_values,
    finite,
    json_pointer,
    normalize_label,
    relative_locator,
    tabular_rows,
)


class HelperTests(unittest.TestCase):
    def test_relative_public_locator(self):
        self.assertEqual(
            relative_locator("outputs/study/reports/final.json"),
            "outputs/study/reports/final.json",
        )
        for locator in (
            "/private/input.json",
            "../input.json",
            "a/../input.json",
            "a//input.json",
            "./input.json",
            "C:\\input.json",
            "https://example.invalid/input.json",
            "file:///input.json",
        ):
            with self.subTest(locator=locator), self.assertRaises(ValidationError):
                relative_locator(locator)

    def test_pointer_preserves_numeric_object_keys_and_escapes(self):
        document = {"0.5": {"a/b": [{"x~y": 17}]}}
        self.assertEqual(json_pointer(document, "/0.5/a~1b/0/x~0y"), 17)
        with self.assertRaises(ValidationError):
            json_pointer(document, "/0.5/a~1b/-1")

    def test_numbers_are_finite_not_booleans(self):
        for value in (True, None, float("nan"), float("inf"), "1.0"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                finite(value, "test")
        self.assertEqual(finite(0.0, "test"), 0.0)

    def test_models_average_before_case_dispersion(self):
        rows = [
            {
                "cell": {"start_day": day, "seed": 100, "regd_load_bias": 0.5},
                "metric": value,
            }
            for day, values in ((1, [0.0, 2.0]), (2, [10.0, 12.0]))
            for value in values
        ]
        values = checkpoint_averaged_values(rows, "metric")
        self.assertEqual(values, [1.0, 11.0])
        self.assertNotEqual(statistics.stdev(values), statistics.stdev([0, 2, 10, 12]))

    def test_diagnostic_labels_remain_identifiable(self):
        self.assertEqual(normalize_label(r"PID$^\dagger$"), "PID")
        self.assertEqual(normalize_label(r"SAC + credit $\Phi$"), "SAC+creditPhi")

    def test_resource_rows_have_two_label_columns(self):
        tex = "\n".join(
            [
                r"Panel & PID variant & Stress throttle & Pass \\",
                r"New calendar/0.0 & Native & 0.9928 & 68/80 \\",
            ]
        )
        rows = tabular_rows(tex, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1].strip(), "Native")

    def test_forbidden_sources_rejected_before_io(self):
        source = Sources(Path("."), {"sources": {}})
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("No file reads allowed")
        ):
            for alias in (
                "comparison_data",
                "assembler",
                "pid_reproduction_inventory",
                "confirmation_registration",
                "model_checkpoint",
            ):
                with self.subTest(alias=alias), self.assertRaises(ValidationError):
                    source.read(alias)

    def test_input_map_cannot_expand_allowlist(self):
        with self.assertRaises(ValidationError):
            Sources(Path("."), {"sources": {}}, {"comparison_data": "excluded.json"})

    def test_tex_has_no_original_locator_fallback(self):
        source = Sources(Path("."), {"sources": {}})
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("No file reads allowed")
        ):
            with self.assertRaisesRegex(ValidationError, "no original-TeX fallback"):
                source.read("tex")


@unittest.skipUnless(
    os.environ.get("DC2GRID_RELEASE_ROOT"),
    (
        "Set DC2GRID_RELEASE_ROOT for integration tests using freshly "
        "generated numeric TeX"
    ),
)
class EvidenceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["DC2GRID_RELEASE_ROOT"])
        directory = Path(__file__).resolve().parent
        cls.mapping = json.loads((directory / "table_row_provenance.json").read_text())
        cls.contracts = json.loads(
            (directory / "controller_protocol_provenance.json").read_text()
        )
        cls.acceptance = json.loads(
            (directory / "provenance_acceptance.json").read_text()
        )
        cls.loaded = Sources(
            cls.root,
            cls.mapping,
            check_hashes=os.environ.get("DC2GRID_CHECK_SOURCE_HASHES") == "1",
        )
        from rebuild_numeric_tables import rebuild

        rebuild(
            cls.mapping,
            cls.contracts,
            cls.acceptance,
            cls.loaded,
            "provenance/numeric_tables",
        )

    def validator(self):
        mapping = copy.deepcopy(self.mapping)
        source = Sources(self.root, mapping, dict(self.loaded.overrides))
        source.cache = dict(self.loaded.cache)
        source.receipts = copy.deepcopy(self.loaded.receipts)
        return Validator(
            mapping,
            copy.deepcopy(self.contracts),
            copy.deepcopy(self.acceptance),
            source,
        )

    def test_full_scope_without_any_additional_file_reads(self):
        validator = self.validator()
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("Unapproved extra file read")
        ):
            result = validator.run()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["counts"]["TeX_rows"], 118)
        self.assertEqual(result["counts"]["TeX_numeric_entries"], 1026)
        self.assertEqual(result["counts"]["main_value_cells"], 528)
        self.assertEqual(set(validator.sources.cache), APPROVED_INPUTS)

    def test_mapped_numeric_change_fails(self):
        validator = self.validator()
        values = copy.deepcopy(validator.sources.cache["comparison_values"])
        values[0]["value"] += 0.01
        validator.sources.cache["comparison_values"] = values
        validator.metadata()
        with self.assertRaisesRegex(ValidationError, "numeric mismatch"):
            validator.main_tables()

    def test_wrong_qualification_count_fails(self):
        validator = self.validator()
        report = copy.deepcopy(validator.sources.cache["pid_report"])
        report["summary"]["calendar"]["0.5"]["pid_native"]["qualified"] = 55
        validator.sources.cache["pid_report"] = report
        validator.metadata()
        with self.assertRaisesRegex(ValidationError, "pass/planned"):
            validator.main_tables()

    def test_failed_case_still_contributes_to_aggregation(self):
        validator = self.validator()
        rows = list(validator.sources.cache["low_case_records"])
        index = next(
            i
            for i, row in enumerate(rows)
            if row["panel"] == "regression"
            and row["macro"] == "rule"
            and row["family"] == "pid"
            and row["bias"] == 0.5
            and not row["qualified"]
        )
        rows[index] = dict(rows[index])
        rows[index]["rmse_kw"] += 100.0
        validator.sources.cache["low_case_records"] = rows
        validator.metadata()
        with self.assertRaisesRegex(ValidationError, "numeric mismatch"):
            validator.low_aggregations()

    def test_floor_enforcement_erratum_is_enforced(self):
        validator = self.validator()
        rows = list(validator.sources.cache["low_case_records"])
        index = next(
            i
            for i, row in enumerate(rows)
            if row["panel"] == "regression"
            and row["macro"] == "full_residual"
            and row["family"] == "pid"
        )
        rows[index] = dict(rows[index], floor_enforced=False)
        validator.sources.cache["low_case_records"] = rows
        validator.metadata()
        with self.assertRaisesRegex(ValidationError, "service-floor"):
            validator.low_aggregations()

    def test_explicit_null_execution_commit_cannot_be_packaging_commit(self):
        validator = self.validator()
        validator.contracts["training_protocols"]["bc"]["execution_commit"] = (
            self.mapping["source_repository_head"]
        )
        with self.assertRaisesRegex(ValidationError, "explicit null"):
            validator.metadata()

    def test_archive_cannot_be_upgraded_to_reproduced(self):
        validator = self.validator()
        archive = next(
            table for table in validator.mapping["tables"] if table["id"] == "Q1"
        )
        archive["default_cell_evidence_status"] = "independently_reproduced"
        with self.assertRaisesRegex(ValidationError, "reported-only"):
            validator.metadata()

    def test_tex_numeric_change_fails(self):
        validator = self.validator()
        validator.metadata()
        validator.main_tables()
        validator.other_tables()
        tex = validator.sources.cache["tex"]
        self.assertIn("3796.1", tex)
        validator.sources.cache["tex"] = tex.replace("3796.1", "3797.1", 1)
        with self.assertRaisesRegex(ValidationError, "Q5.pid"):
            validator.tex()

    def test_resource_maximum_not_mean_is_checked(self):
        validator = self.validator()
        supplement = copy.deepcopy(validator.sources.cache["pid_audit_supplement"])
        supplement["summary"]["calendar"]["0.5"]["pid_native"]["worst_backlog_kwh"] = (
            0.0
        )
        validator.sources.cache["pid_audit_supplement"] = supplement
        validator.metadata()
        with self.assertRaisesRegex(ValidationError, "Worst queue"):
            validator.pid_aggregations_and_pairing()


if __name__ == "__main__":
    unittest.main()
