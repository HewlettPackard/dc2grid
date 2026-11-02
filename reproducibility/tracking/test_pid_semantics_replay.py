"""PID portability tests use only synthetic files, ASTs and archived receipts."""

from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

import pid_semantics_replay as pid
import tracking_release as release


def inert_probe_source():
    tree = ast.parse(Path(pid.__file__).read_text())
    frame = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "trace_frame"
        )
    )
    frame.name = "frame"
    frame.body = [
        node
        for node in frame.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    summary = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "summarize_pair"
        )
    )
    summary.name = "main"
    summary.body = [node for node in summary.body if isinstance(node, ast.Assign)]
    return (
        "raise RuntimeError('The archival probe must never be imported')\n"
        + ast.unparse(ast.Module(body=[frame, summary], type_ignores=[]))
    )


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.home = (
            Path(__file__).resolve().parent / "pid_test_workspace" / uuid.uuid4().hex
        )
        self.root = self.home / "new-checkout"
        self.root.mkdir(parents=True)
        self.index = {
            "archive_root_label": "/archived/original",
            "files": {},
            "source_identity_files": ["c2g_env/core.py"],
            "data_directories": {
                "energy_dir": "data/energy",
                "weather_dir": "data/weather",
                "trace_dir": "data/workload",
            },
        }
        self.write("c2g_env/core.py", "# hash-only source, never executed\n")
        for directory in ("energy", "weather", "workload"):
            (self.root / "data" / directory).mkdir(parents=True)
        self.write("data/energy/NYC.csv", "load\n100\n")
        self.write(pid.PROBE, inert_probe_source())
        self.index["source_id"] = release.identity(
            {"c2g_env/core.py": self.index["files"]["c2g_env/core.py"]["sha256"]}
        )
        self.report = {
            "source_id": self.index["source_id"],
            "recorded_probe_source_sha256": self.index["files"][pid.PROBE]["sha256"],
            "case_pair_count": 30,
            "diagnostic_run_count": 60,
            "per_mode_case_pairs": 15,
            "role": "diagnostic_reproduction_not_new_publication_evidence",
            "all_60_trajectories_complete_24h_service": True,
            (
                "recovery_not_run_or_needed_for_this_service_only_metric_reproduction"
            ): True,
            "cases": [
                {
                    "mode": mode,
                    "day": day,
                    "seed": seed,
                    "agent_ticks": 17280,
                    "baseline_ticks": 17280,
                }
                for mode in ("legacy_v0", "physical_v1")
                for day in (207, 248, 249)
                for seed in range(100, 105)
            ],
        }
        self.write(pid.REPORT, self.report)
        self.request = {"mode": "physical_v1", "day": 207, "seed": 100}

    def tearDown(self):
        shutil.rmtree(self.home)
        if not any(self.home.parent.iterdir()):
            self.home.parent.rmdir()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
        )
        self.index["files"][name] = {
            "sha256": release.file_hash(path),
            "required_for_replay": True,
        }

    def prepare(self):
        return pid._prepare(self.root, self.request, self.index)

    def test_preflight_never_executes_archival_probe_or_uses_its_globals(self):
        plan = self.prepare()
        self.assertEqual(plan["request"], self.request)
        self.assertFalse(plan["formula_check"]["archived_probe_imported_or_executed"])
        self.assertEqual(plan["expected_native_guards"]["paired_ticks"], 17280)
        self.assertEqual(plan["source_id"], self.index["source_id"])

    def test_missing_data_is_an_explicit_blocker(self):
        (self.root / "data/energy/NYC.csv").unlink()
        with self.assertRaisesRegex(
            release.ReleaseError, "Missing required scientific input"
        ):
            self.prepare()

    def test_changed_native_source_is_not_accepted(self):
        (self.root / "c2g_env/core.py").write_text("changed physics\n")
        with self.assertRaisesRegex(release.ReleaseError, "SHA-256 mismatch"):
            self.prepare()

    def test_added_helper_in_native_source_identity_trees_is_not_ignored(self):
        (self.root / "scripts").mkdir()
        (self.root / "scripts/new_release_helper.py").write_text("# new helper\n")
        with self.assertRaisesRegex(release.ReleaseError, "exact original four"):
            self.prepare()

    def test_removed_pair_is_rejected_even_with_updated_fixture_file_hash(self):
        self.report["cases"].pop()
        self.write(pid.REPORT, self.report)
        with self.assertRaisesRegex(release.ReleaseError, "original 30 pairs"):
            self.prepare()

    def test_duplicate_pair_is_not_relabelled_as_another_original_pair(self):
        self.report["cases"][0] = self.report["cases"][1]
        self.write(pid.REPORT, self.report)
        with self.assertRaisesRegex(release.ReleaseError, "original 30 pairs"):
            self.prepare()

    def test_probe_hash_link_is_validated(self):
        self.report["recorded_probe_source_sha256"] = "0" * 64
        self.write(pid.REPORT, self.report)
        with self.assertRaisesRegex(
            release.ReleaseError, "diagnostic contract changed"
        ):
            self.prepare()

    def test_rebased_configuration_only_changes_three_data_directories(self):
        original = {
            "global": {
                "simulation_mode": "legacy_v0",
                "energy_dir": "data/energy",
                "weather_dir": "data/weather",
                "trace_dir": "data/workload",
                "episode_ticks": 17280,
            },
            "reward": {"alpha": 1.0, "beta": 2.0},
            "default": {"weather_market": "nyiso_nyc"},
        }
        before = copy.deepcopy(original)
        actual = pid.rebase_config(original, self.root, self.index)
        for key in self.index["data_directories"]:
            self.assertTrue(Path(actual["global"][key]).is_relative_to(self.root))
            actual["global"][key] = before["global"][key]
        self.assertEqual(actual, before)
        self.assertEqual(original, before)


class PureContractTests(unittest.TestCase):
    def test_only_original_mode_date_seed_requests_are_accepted(self):
        for mode in ("legacy_v0", "physical_v1"):
            for day in (207, 248, 249):
                for seed in range(100, 105):
                    request = {"mode": mode, "day": day, "seed": seed}
                    self.assertEqual(pid.normalize_request(request), request)
        request = {"mode": "legacy_v0", "day": 207, "seed": 100}
        for change in (
            {"mode": "new_mode"},
            {"day": 208},
            {"seed": 105},
            {"day": True},
            {"seed": 100.0},
            {"bias": 0.0},
            {"OUT": "other"},
            {"macro": "full_residual"},
        ):
            with self.subTest(change=change), self.assertRaises(release.ReleaseError):
                pid.normalize_request({**request, **change})

    def test_component_formula_ast_must_match_archived_probe(self):
        adapter = Path(pid.__file__).read_text()
        result = pid.verify_formula_parity(inert_probe_source(), adapter)
        self.assertTrue(result["component_assignment_ASTs_identical"])
        changed = adapter.replace("actual_pcc = facility +", "actual_pcc = facility -")
        with self.assertRaisesRegex(release.ReleaseError, "component formulas differ"):
            pid.verify_formula_parity(inert_probe_source(), changed)

    def test_default_csv_reader_is_preserved_not_silently_changed(self):
        changed = (
            Path(pid.__file__)
            .read_text()
            .replace(
                'pd.read_csv(directory / "hardware.csv.gz")',
                'pd.read_csv(directory / "hardware.csv.gz", '
                'float_precision="round_trip")',
            )
        )
        with self.assertRaisesRegex(release.ReleaseError, "trace reader differs"):
            pid.verify_formula_parity(inert_probe_source(), changed)

    def test_comparison_requires_all_values_and_exact_qualification(self):
        self.assertEqual(
            pid.compare_values(
                {"mw": 8.0, "qualified": True}, {"mw": 8.0000000001, "qualified": True}
            ),
            [],
        )
        for expected, actual in (
            ({"mw": 8.0}, {}),
            ({"qualified": True}, {"qualified": 1}),
            ({"ticks": 17280}, {"ticks": 17279}),
            ({"mw": 8.0}, {"mw": float("nan")}),
            ({"mw": 8.0}, {"mw": 8.01}),
        ):
            with self.subTest(actual=actual):
                self.assertTrue(pid.compare_values(expected, actual))

    def test_bundled_targets_include_entire_power_balance_for_both_modes(self):
        index = pid.inventory()
        self.assertEqual(len(index["source_identity_files"]), 61)
        self.assertEqual(len(index["runtime_modules"]), 31)
        self.assertEqual(len(index["allowed_requests"]), 30)
        self.assertEqual(index["original_run_count"], 60)
        self.assertEqual(
            index["source_id"],
            "af19cdcd74eb7ac9b5330f4afeab4983ade823c1151eb530c7dc501495c590f7",
        )
        by_mode = {
            row["request"]["mode"]: row["expected_full_components"]
            for row in index["representatives"]
        }
        self.assertEqual(set(by_mode), {"legacy_v0", "physical_v1"})
        self.assertEqual(
            by_mode["legacy_v0"]["reported_formula_relief_mw"], 8.268034860731056
        )
        self.assertEqual(
            by_mode["physical_v1"]["reported_formula_relief_mw"], 3.208615293152879
        )
        self.assertEqual(
            by_mode["physical_v1"]["stress_battery_discharge_mw"], -0.1889728900205709
        )
        for result in by_mode.values():
            pid.check_power_balance(result)
            changed = {**result, "stress_true_metered_error_mw": 123.0}
            with self.assertRaisesRegex(release.ReleaseError, "balance failed"):
                pid.check_power_balance(changed)


@unittest.skipUnless(
    os.environ.get("DC2GRID_RELEASE_ROOT"),
    "Set DC2GRID_RELEASE_ROOT for read-only native dependency checks",
)
class ReadOnlyCheckoutTests(unittest.TestCase):
    def test_both_original_pair_preflights_without_native_import_or_rollout(self):
        root = Path(os.environ["DC2GRID_RELEASE_ROOT"])
        for target in pid.inventory()["representatives"]:
            with self.subTest(mode=target["request"]["mode"]):
                plan = pid.prepare_pid_pair(root, target["request"])
                self.assertEqual(plan["expected"], target["expected_full_components"])
                self.assertTrue(plan["expected_native_guards"]["qualified"])
                self.assertTrue(
                    plan["formula_check"]["component_assignment_ASTs_identical"]
                )


if __name__ == "__main__":
    unittest.main()
