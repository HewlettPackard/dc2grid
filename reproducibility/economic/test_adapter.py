"""No simulator episodes, source monkeypatches, network, or training in these tests."""

from __future__ import annotations

import ast
import copy
import os
import shutil
import unittest
import uuid
from pathlib import Path

import adapter


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.policies, cls.cases = adapter.load_metadata()
        cls.representatives = adapter.read_json(
            adapter.HERE / "representative_cases.json"
        )["cases"]
        cls.failures = adapter.read_json(adapter.HERE / "failures.json")

    def test_closed_catalog_and_failure_counts(self):
        self.assertEqual(len(self.policies), 73)
        self.assertEqual(len(self.cases), 12894)
        self.assertEqual(len(self.manifest["models"]), 24)
        self.assertEqual(len(self.representatives), 11)
        failed = self.failures["completed_disqualified_cases"]
        self.assertEqual(len(failed), 86)
        self.assertEqual(len(self.failures["runtime_failure_attempts"]), 0)
        self.assertEqual(
            {key: sum(row["policy_key"] == key for row in failed) for key in
             ("pid/pid_native", "pid/pid_full_work")},
            {"pid/pid_native": 43, "pid/pid_full_work": 43},
        )

    def test_every_case_is_an_explicit_policy_cell_slot(self):
        for key, row in self.cases.items():
            spec = {"policy_key": row["policy_key"], "panel": row["panel"]}
            policy, found = adapter.resolve_request(
                spec, row["cell"], self.policies, self.cases
            )
            self.assertEqual(key, found["case_id"])
            self.assertEqual(policy["policy_key"], row["policy_key"])
            self.assertEqual(len(row["record"]["sha256"]), 64)

    def test_new_and_reserved_cells_cannot_be_opened(self):
        row = self.representatives[0]
        spec = {"policy_key": row["policy_key"], "panel": row["panel"]}
        for cell in (
            {**row["cell"], "seed": 9876543210123},
            {**row["cell"], "start_day": 364},
        ):
            with self.assertRaisesRegex(ValueError, "No consumed outcome"):
                adapter.resolve_request(spec, cell, self.policies, self.cases)
        with self.assertRaisesRegex(ValueError, "No consumed outcome"):
            adapter.resolve_request(
                {**spec, "panel": "unopened_old_reservation"},
                row["cell"],
                self.policies,
                self.cases,
            )

    def test_policy_overrides_are_not_accepted(self):
        row = self.representatives[0]
        spec = {"policy_key": row["policy_key"], "panel": row["panel"]}
        for extra in (
            {"fraction": 0.1},
            {"checkpoint": {"path": "different.zip"}},
            {"full_work": False},
        ):
            with self.assertRaisesRegex(ValueError, "immutable policy_key"):
                adapter.resolve_request(
                    {**spec, **extra}, row["cell"], self.policies, self.cases
                )

    def test_published_case_goldens_remain_self_consistent(self):
        for row in self.representatives:
            expected = self.cases[row["case_id"]]["expected"]
            self.assertEqual(row["expected"], expected)
            self.assertEqual(adapter.compare_expected(expected, expected), [])

    def test_service_soc_and_recovery_closure_are_distinct(self):
        rows = [
            row for row in self.representatives
            if row["coverage"].endswith("failed_service_SOC")
        ]
        self.assertEqual(len(rows), 2)
        for row in rows:
            snapshot = row["expected"]["snapshot"]
            self.assertLess(snapshot["service_terminal"]["soc"], 0.2)
            self.assertFalse(snapshot["service_checks"]["terminal_soc"])
            self.assertTrue(snapshot["recovery"]["soc_closed"])
            self.assertTrue(snapshot["recovery"]["closed"])
            self.assertTrue(snapshot["closed_26h_complete"])
            self.assertFalse(snapshot["qualified"])
            self.assertFalse(snapshot["cash_is_admissible"])
            self.assertIsNotNone(snapshot["cashflow_26h_usd"])

    def test_disqualified_outcomes_are_not_erased(self):
        failed = self.failures["completed_disqualified_cases"]
        for row in failed:
            self.assertEqual(row, self.cases[row["case_id"]])
            self.assertFalse(row["expected"]["snapshot"]["qualified"])
            self.assertIn("failure_reasons", row["expected"]["audit"])
        self.assertTrue(self.failures["no_survivor_only_means"])

    def test_source_data_and_all_weights_are_hash_bound(self):
        for group in ("sources", "configs", "protocols", "data", "models"):
            self.assertTrue(self.manifest[group])
            for row in self.manifest[group]:
                self.assertEqual(len(row["sha256"]), 64)
                self.assertEqual(
                    self.manifest["recorded_path_map"][row["recorded_path"]],
                    row["checkout_relative"],
                )
        self.assertEqual(len(self.manifest["data"]), 5)
        self.assertTrue(all(
            row["hash_also_bound_by_original_registration"]
            for row in self.manifest["sources"]
        ))

    def test_no_private_author_or_historical_inventory_dependency(self):
        required = [
            row["checkout_relative"]
            for group in ("sources", "configs", "protocols", "data", "models")
            for row in self.manifest[group]
        ]
        for name in required:
            self.assertNotIn("attachments/", name)
            self.assertNotIn("pre_amendment", name)
            self.assertNotIn("AAAI-demo", name)
            self.assertFalse(name.endswith(".tex"))
            self.assertNotIn("Benchmark.zip", name)
        self.assertFalse(
            self.manifest["verification_scope"]["legacy_verify_functions_invoked"]
        )

    def test_same_action_space_and_end_to_end_labels_are_not_conflated(self):
        for key in (
            "confirmation/script_zero",
            "confirmation/script_f100_always",
            "confirmation/model__fraction_sac_1200",
            "confirmation/model__potential_sac_1200",
        ):
            self.assertEqual(
                self.policies[key]["action_space_comparison"],
                "same_native_fraction_action_space",
            )
        self.assertEqual(
            self.policies["absolute/economic_sac_1200"]["action_space_comparison"],
            "absolute_0_to_6MW_not_fraction_coordinates",
        )
        for key in ("pid/pid_native", "pid/pid_full_work"):
            self.assertEqual(
                self.policies[key]["action_space_comparison"],
                "end_to_end_four_actuator_reference_not_fraction_action_space",
            )

    def test_all_potential_models_deploy_without_potential(self):
        for entry in self.policies.values():
            self.assertFalse(entry["potential_active_at_deployment"])
        for row in self.manifest["models"]:
            self.assertEqual(row["gamma"], 1.0)
            self.assertEqual(row["final_macro_transitions"], 24576)
            self.assertEqual(row["observation_dimension"], 23)
            self.assertEqual(row["action_dimension"], 1)

    def test_physical_v1_uses_explicit_frozen_simple_battery(self):
        self.assertTrue(
            self.manifest["runtime"][
                "physical_v1_explicitly_selects_simple_BESS_even_if_PySAM_installed"
            ]
        )
        self.assertFalse(self.manifest["runtime"]["cuda_required"])

    def test_historical_reuse_is_disclosed(self):
        reused = [
            row for row in self.cases.values()
            if row["policy_key"].startswith("confirmation/")
            and row["record"]["historical_flat_schema"]
        ]
        self.assertEqual(len(reused), 864)
        self.assertEqual({row["panel"] for row in reused},
                         {"development", "regression"})


class PathAndIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(".test-fixtures") / f"{os.getpid()}-{uuid.uuid4().hex}"
        self.work.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, self.work)
        self.root = (self.work / "new-checkout").resolve()
        self.root.mkdir()
        self.original = self.work.resolve() / "read-only-recorded-tree"
        self.original.mkdir()
        self.manifest = {
            "recorded_root": str(self.original),
            "recorded_path_map": {},
            "config_directory_keys": ["trace_dir", "energy_dir", "weather_dir"],
            **{
                name: []
                for name in ("sources", "configs", "protocols", "data", "models")
            },
        }

    def add_dependency(self, group, relative, content=b"frozen\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        recorded = str(self.original / relative)
        row = {
            "recorded_path": recorded,
            "checkout_relative": relative,
            "sha256": adapter.file_hash(path),
        }
        self.manifest["recorded_path_map"][recorded] = relative
        self.manifest[group].append(row)
        return path, recorded

    def test_explicit_mapping_works_after_checkout_relocation(self):
        path, recorded = self.add_dependency("sources", "outputs/frozen/task.py")
        self.assertEqual(
            adapter.mapped_path(self.root, recorded, self.manifest), path
        )
        self.assertEqual(adapter.verify_files(self.root, self.manifest)["sources"], 1)

    def test_missing_mapping_never_falls_back_to_old_absolute_path(self):
        original_file = self.original / "available-but-unmapped.json"
        original_file.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unmapped recorded runtime path"):
            adapter.mapped_path(self.root, str(original_file), self.manifest)

    def test_all_dependency_categories_fail_closed_on_drift(self):
        for group in ("sources", "configs", "protocols", "data", "models"):
            path, _ = self.add_dependency(group, f"{group}/frozen")
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed/missing"):
                adapter.verify_files(self.root, self.manifest)
            path.write_bytes(b"frozen\n")
            adapter.verify_files(self.root, self.manifest)

    def test_missing_data_does_not_invoke_loader_fallback(self):
        path, _ = self.add_dependency("data", "data/NYC.csv")
        path.unlink()
        with self.assertRaisesRegex(ValueError, "changed/missing"):
            adapter.verify_files(self.root, self.manifest)

    def test_traversal_absolute_and_symlink_escape_are_rejected(self):
        for bad in ("../escape", "/absolute", "a/../b", "a//b", r"a\b", "C:/file"):
            with self.assertRaises(ValueError):
                adapter.checkout_path(self.root, bad)
        (self.root / "escape").symlink_to(self.original, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            adapter.checkout_path(self.root, "escape/file")

    def test_config_projection_changes_only_three_directory_paths(self):
        config = {
            "global": {
                "dt_seconds": 5,
                "episode_ticks": 17280,
                "simulation_mode": "physical_v1",
            },
            "reward": {"tracking_floor_fraction": 0.0004},
            "default": {"bess_soc_init": 0.5},
        }
        for key in self.manifest["config_directory_keys"]:
            relative = f"data/{key}"
            (self.root / relative).mkdir(parents=True)
            old = str(self.original / relative)
            config["global"][key] = old
            self.manifest["recorded_path_map"][old] = relative
        before = copy.deepcopy(config)
        result = adapter.relocate_config(config, self.root, self.manifest)
        self.assertEqual(before, config)
        for key in self.manifest["config_directory_keys"]:
            self.assertEqual(result["global"][key], str(self.root / f"data/{key}"))
            result["global"][key] = before["global"][key]
        self.assertEqual(result, before)

    def test_protocol_and_checkpoint_relocation_leave_gates_and_hashes_unchanged(self):
        protocol = {
            "registration_id": "original-identity",
            "config_path": "old/environment.yaml",
            "service_gate": {
                "min_terminal_soc": 0.2,
                "max_backlog_fraction": 0.01,
                "max_thermal_warning_fraction": 0.01,
            },
            "service_floor": 0.9,
            "recovery": {"duration_seconds": 7200, "soc_tolerance": 1e-6},
            "normalization": {"cash_prefix_bound_usd": 2824000},
            "input_hashes": {"private-provenance-only": "untouched"},
        }
        frozen = copy.deepcopy(protocol)
        runtime = adapter.relocate_protocol(protocol, self.root / "runtime.yaml")
        runtime["config_path"] = protocol["config_path"]
        self.assertEqual(runtime, frozen)
        self.assertEqual(protocol, frozen)
        _, old = self.add_dependency("models", "models/final.zip")
        policy = {"checkpoint": {"path": old, "sha256": "unchanged"},
                  "variant": "potential", "algorithm": "sac"}
        mapped = adapter.relocate_policy(policy, self.root, self.manifest)
        self.assertEqual(mapped["checkpoint"]["sha256"], "unchanged")
        self.assertEqual(policy["checkpoint"]["path"], old)
        self.assertEqual(mapped["variant"], "potential")

    def test_original_tree_and_frozen_output_paths_are_forbidden(self):
        with self.assertRaisesRegex(ValueError, "new checkout"):
            adapter.validate_locations(self.original, self.root, self.manifest)
        with self.assertRaisesRegex(ValueError, "read-only tree"):
            adapter.validate_locations(
                self.root, self.original / "new-result", self.manifest
            )
        with self.assertRaisesRegex(ValueError, "outside frozen outputs"):
            adapter.validate_locations(
                self.root, self.root / "outputs/changed", self.manifest
            )
        with self.assertRaisesRegex(ValueError, "separate from the release adapter"):
            adapter.validate_locations(
                self.root, adapter.HERE / "result", self.manifest
            )


class MetricTests(unittest.TestCase):
    def test_gate_booleans_and_integer_counts_are_exact(self):
        expected = {"qualified": False, "steps": 17280, "metric": 1.0}
        for change in (
            {"qualified": True},
            {"qualified": 0},
            {"steps": 17280.0},
        ):
            self.assertTrue(
                adapter.compare_expected({**expected, **change}, expected)
            )
        self.assertEqual(
            adapter.compare_expected({**expected, "metric": 1 + 1e-9}, expected), []
        )
        self.assertTrue(adapter.compare_expected(
            {**expected, "metric": 1 + 1e-5}, expected
        ))

    def test_missing_none_and_nonfinite_metrics_do_not_pass(self):
        for actual, expected in (
            ({}, {"cash": None}),
            ({"cash": 0.0}, {"cash": None}),
            ({"cash": float("nan")}, {"cash": 1.0}),
            ({"cash": float("inf")}, {"cash": 1.0}),
        ):
            self.assertTrue(adapter.compare_expected(actual, expected))

    def test_pid_service_complete_does_not_mean_closed_26h_complete(self):
        outcome = {
            "complete": True,
            "qualified": False,
            "recovery": {"complete": False, "closed": False},
            "cashflow_26h_usd": None,
        }
        snapshot = adapter.scientific_snapshot(outcome)
        self.assertTrue(snapshot["complete"])
        self.assertFalse(snapshot["closed_26h_complete"])
        self.assertFalse(snapshot["cash_is_admissible"])

    def test_false_closure_cannot_be_hidden_by_cash_value(self):
        outcome = {
            "complete": True,
            "qualified": False,
            "recovery": {"complete": True, "closed": False},
            "cashflow_26h_usd": -123.0,
            "cashflow_closed_recovery_usd": None,
        }
        snapshot = adapter.scientific_snapshot({"outcome": outcome})
        self.assertFalse(snapshot["cash_is_admissible"])
        self.assertIsNone(snapshot["cashflow_closed_recovery_usd"])

    def test_bad_cell_types_are_rejected(self):
        for cell in (
            {"start_day": True, "seed": 1, "regd_load_bias": 0.0},
            {"start_day": 1, "seed": 1.5, "regd_load_bias": 0.0},
            {"start_day": 1, "seed": 1, "regd_load_bias": True},
            {"start_day": 1, "seed": 1, "regd_load_bias": float("nan")},
            {"start_day": 1, "seed": 1, "regd_load_bias": 0.0, "hidden": 1},
        ):
            with self.assertRaises(ValueError):
                adapter.case_key(cell)

    def test_adapter_delegates_to_explicit_frozen_pid_factory(self):
        tree = ast.parse((adapter.HERE / "adapter.py").read_text(encoding="utf-8"))
        factories = [
            item for item in ast.walk(tree)
            if isinstance(item, ast.keyword) and item.arg == "hardware_factory"
        ]
        self.assertEqual(len(factories), 1)
        expression = ast.unparse(factories[0].value)
        self.assertIn("pid_reference", expression)
        self.assertTrue(expression.endswith(".hardware_factory"))
        for item in ast.walk(tree):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
                self.assertNotIn(item.func.attr, {"learn", "fit"})


if __name__ == "__main__":
    unittest.main()
