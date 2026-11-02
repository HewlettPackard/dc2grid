"""Focused tests: pure fixtures, hashes and static consumed receipts; no rollouts."""

from __future__ import annotations

import copy
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

import tracking_release as release


def episode(policy, cell, *, complete=True, qualified=True, ratio=None):
    value = {
        "policy": policy,
        "cell": cell,
        "reset_info": {"simulation_mode": "physical_v1"},
        "complete": complete,
        "qualified": qualified,
        "service_checks": {
            "complete": complete,
            "terminal_soc": True,
            "terminal_work": True,
            "thermal_warnings": complete,
        },
        "service": {
            "steps": 17280 if complete else 9000,
            "tracking_rmse_kw": 12.5,
            "battery_throughput_mwh": 0.1,
            "thermal_warning_fraction": 0.0 if complete else 0.1,
            "work_arrived_kwh": 1000.0,
            "work_served_kwh": 1000.0,
            "backlog_kwh": 0.0,
            "work_conservation_error_kwh": 0.0,
        },
        "service_terminal": {"soc": 0.4, "temp_A": 32.0, "temp_B": 33.0},
        "recovery": {
            "complete": complete,
            "closed": complete,
            **(
                {"ledger": {"steps": 1440, "battery_throughput_mwh": 0.02}}
                if complete
                else {}
            ),
        },
        "grid": {"stress_facility_relief_mw": 5.0} if complete else None,
        "cashflow_26h_usd": -1000.0 if complete else None,
        "cashflow_closed_recovery_usd": -1000.0 if complete else None,
        "regulation_revenue_usd": 20.0,
    }
    if ratio is not None:
        value["delivered_service_ratio"] = ratio
    return value


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.home = (
            Path(__file__).resolve().parent / "test_workspace" / uuid.uuid4().hex
        )
        self.root = self.home / "relocated-checkout"
        self.root.mkdir(parents=True)
        self.cell = {"start_day": 207, "seed": 100, "regd_load_bias": 0.5}
        self.index = {
            "archive_root_label": "/archived/frozen-project",
            "files": {},
            "models": {},
            "panels": {},
            "runtime": {
                "source_files": ["scientific/source.py"],
                "config_files": ["scientific/environment.yaml"],
                "data_files": ["data/energy/NYC.csv"],
                "data_directories": {
                    "energy_dir": "data/energy",
                    "weather_dir": "data/weather",
                    "trace_dir": "data/workload",
                },
            },
        }
        for directory in ("energy", "weather", "workload"):
            (self.root / "data" / directory).mkdir(parents=True, exist_ok=True)
        self.write(
            "scientific/source.py", "# hash-only source fixture; never imported\n"
        )
        self.write("scientific/environment.yaml", "global: {}\n")
        self.write("data/energy/NYC.csv", "load\n100\n")
        self.write("models/selected.zip", "hash-only model fixture; never unpickled\n")
        artifact = {
            "path": "/archived/frozen-project/models/selected.zip",
            "sha256": self.index["files"]["models/selected.zip"]["sha256"],
        }
        self.policy = {
            "id": "rule__guided_sac_1100",
            "family": "guided_sac",
            "hardware": "history_sac",
            "macro": "rule",
            "training_seed": 1100,
            "available": True,
            "checkpoint": artifact,
        }
        self.index["models"]["models/selected.zip"] = {
            "family": self.policy["family"],
            "sha256": artifact["sha256"],
        }
        self.protocol = {
            "config_path": "/archived/frozen-project/scientific/environment.yaml",
            "hardware_reference": {
                "path": "/archived/frozen-project/models/original-low.zip",
                "sha256": "0" * 64,
            },
            "normalization": {"import_abs_mw": 400},
            "service_gate": {"min_terminal_soc": 0.2},
            "recovery": {"duration_seconds": 7200, "throttle": 1.0},
            "code_hashes": {
                "source.py": self.index["files"]["scientific/source.py"]["sha256"]
            },
        }
        self.protocol["registration_id"] = release.identity(self.protocol)
        self.make_table2()

    def tearDown(self):
        shutil.rmtree(self.home)
        parent = self.home.parent
        if not any(parent.iterdir()):
            parent.rmdir()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2) + "\n" if not isinstance(value, str) else value
        )
        self.index["files"][name] = {
            "sha256": release.file_hash(path),
            "tracked": True,
            "kinds": [],
        }

    def make_table2(self):
        self.write(f"{release.TABLE}/registration.json", self.protocol)
        lock = {
            "training_registration_id": self.protocol["registration_id"],
            "inventory": [self.policy],
        }
        lock["lock_id"] = release.identity(lock)
        lock_name = f"{release.TABLE}/final_candidate_lock.json"
        self.write(lock_name, lock)
        final = {
            **self.protocol,
            "candidate_lock_id": lock["lock_id"],
            "candidate_lock_sha256": self.index["files"][lock_name]["sha256"],
            "policies": [self.policy],
            "panels": {"regression": [self.cell]},
            "final_claims": {"secondary_service_floor": 0.9},
        }
        final["final_registration_id"] = release.identity(final)
        self.write(f"{release.TABLE}/final_registration.json", final)
        self.panel = {
            "protocol": f"{release.TABLE}/final_registration.json",
            "manifest": "results/manifest.json",
            "summary": "results/summary.json",
            "evaluator": "table2",
            "authorization": "table2_final",
            "role": "regression",
        }
        self.index["panels"]["fixture"] = self.panel
        self.write(
            "results/manifest.json",
            {
                "policies": [self.policy],
                "cells": [self.cell],
                "role": "regression",
                "registration_id": self.protocol["registration_id"],
                "final_registration_id": final["final_registration_id"],
            },
        )
        self.write(
            "results/summary.json",
            {"episodes": [episode(self.policy, self.cell)], "runtime_failures": []},
        )

    def make_final(self):
        self.policy["id"] = "rule__union_conditioned_1100"
        self.policy["family"] = "union_conditioned"
        self.index["models"]["models/selected.zip"]["family"] = "union_conditioned"
        original = {
            key: value
            for key, value in self.protocol.items()
            if key != "registration_id"
        }
        original.update(
            service_floor=0.9,
            final_cells=[self.cell],
            high_level={"rmcp_normalizer": 100.0},
        )
        original["registration_id"] = release.identity(original)
        candidate = {
            "registration_id": original["registration_id"],
            "inventory": [self.policy],
        }
        candidate["candidate_lock_id"] = release.identity(candidate)
        final = {
            "registration_id": original["registration_id"],
            "candidate_lock_id": candidate["candidate_lock_id"],
            "cells": [self.cell],
            "policies": [self.policy],
        }
        final["lock_id"] = release.identity(final)
        self.write(f"{release.VALUE}/registration.json", original)
        self.write(f"{release.VALUE}/candidate_lock.json", candidate)
        self.write(f"{release.VALUE}/final_lock.json", final)
        self.write(f"{release.STATE}/final_lock.json", final)
        self.panel.update(
            protocol=f"{release.VALUE}/registration.json",
            evaluator="state",
            authorization="value_final",
            role="final",
        )
        self.write(
            "results/manifest.json",
            {
                "policies": [self.policy],
                "cells": [self.cell],
                "role": "final",
                "registration_id": original["registration_id"],
            },
        )
        self.write(
            "results/summary.json",
            {
                "episodes": [episode(self.policy, self.cell, ratio=0.8)],
                "runtime_failures": [],
            },
        )
        return original

    def prepare(self, **overrides):
        spec = {"snapshot": "fixture", "policy_id": self.policy["id"]}
        return release._prepare(
            self.root,
            overrides.get("spec", spec),
            overrides.get("cell", self.cell),
            self.index,
        )

    def test_portable_resolution_preserves_original_documents(self):
        before = (self.root / self.panel["protocol"]).read_bytes()
        plan = self.prepare()
        self.assertEqual(
            plan["policy"]["checkpoint"]["path"], str(self.root / "models/selected.zip")
        )
        self.assertEqual(
            plan["runtime_protocol"]["config_path"],
            str(self.root / "scientific/environment.yaml"),
        )
        self.assertEqual(plan["archived_policy"], self.policy)
        self.assertEqual((self.root / self.panel["protocol"]).read_bytes(), before)
        self.assertEqual(
            plan["runtime_protocol"]["service_gate"], self.protocol["service_gate"]
        )
        self.assertEqual(
            plan["runtime_protocol"]["recovery"], self.protocol["recovery"]
        )

    def test_changed_source_is_a_blocker(self):
        (self.root / "scientific/source.py").write_text("changed source\n")
        with self.assertRaisesRegex(release.ReleaseError, "SHA-256 mismatch"):
            self.prepare()

    def test_scientific_inputs_are_rechecked_after_replay(self):
        plan = self.prepare()
        release.verify_unchanged_inputs(self.root, plan["verified_inputs"])
        (self.root / "scientific/source.py").write_text("changed during execution\n")
        with self.assertRaisesRegex(release.ReleaseError, "changed during replay"):
            release.verify_unchanged_inputs(self.root, plan["verified_inputs"])

    def test_missing_data_does_not_trigger_synthetic_fallback(self):
        (self.root / "data/energy/NYC.csv").unlink()
        with self.assertRaisesRegex(
            release.ReleaseError, "Missing required scientific input"
        ):
            self.prepare()

    def test_changed_selected_model_is_a_blocker(self):
        (self.root / "models/selected.zip").write_text("another model\n")
        with self.assertRaisesRegex(release.ReleaseError, "SHA-256 mismatch"):
            self.prepare()

    def test_model_family_cannot_be_substituted(self):
        self.index["models"]["models/selected.zip"]["family"] = "bc_only"
        with self.assertRaisesRegex(release.ReleaseError, "Checkpoint family"):
            self.prepare()

    def test_new_cell_is_not_a_consumed_case(self):
        with self.assertRaisesRegex(release.ReleaseError, "consumed result"):
            self.prepare(cell={**self.cell, "seed": 999})

    def test_policy_override_is_not_allowed(self):
        with self.assertRaisesRegex(release.ReleaseError, "no policy overrides"):
            self.prepare(
                spec={
                    "snapshot": "fixture",
                    "policy_id": self.policy["id"],
                    "macro": "full_residual",
                }
            )

    def test_wrong_macro_alias_is_not_substituted(self):
        with self.assertRaisesRegex(release.ReleaseError, "consumed result"):
            self.prepare(
                spec={"snapshot": "fixture", "policy_id": "full__guided_sac_1100"}
            )

    def test_final_lock_authorized_before_runtime_resolution(self):
        original = self.make_final()
        before = copy.deepcopy(original)
        plan = self.prepare()
        self.assertEqual(original, before)
        self.assertEqual(plan["authorization"], "value_final")
        self.assertNotIn("final_cells", plan["runtime_protocol"])
        self.assertEqual(plan["runtime_protocol"]["service_floor"], 0.9)
        self.assertEqual(
            plan["runtime_protocol"]["registration_id"], original["registration_id"]
        )
        self.assertTrue(plan["expected_metrics"]["publication_qualified"])
        self.assertFalse(plan["expected_metrics"]["delivered_service_gate"])

    def test_shared_final_lock_cannot_change_even_with_a_new_valid_hash(self):
        self.make_final()
        path = f"{release.STATE}/final_lock.json"
        shared = json.loads((self.root / path).read_text())
        shared["cells"] = [{**self.cell, "seed": 101}]
        shared.pop("lock_id")
        shared["lock_id"] = release.identity(shared)
        self.write(path, shared)
        with self.assertRaisesRegex(release.ReleaseError, "shared Final-B lock"):
            self.prepare()

    def test_candidate_lock_cannot_omit_a_final_policy(self):
        self.make_final()
        path = f"{release.VALUE}/candidate_lock.json"
        candidate = json.loads((self.root / path).read_text())
        candidate["inventory"] = []
        candidate.pop("candidate_lock_id")
        candidate["candidate_lock_id"] = release.identity(candidate)
        self.write(path, candidate)
        with self.assertRaisesRegex(release.ReleaseError, "union candidate lock"):
            self.prepare()

    def test_rebase_config_changes_exactly_three_directory_values(self):
        config = {
            "global": {
                "energy_dir": "/archived/frozen-project/data/energy",
                "weather_dir": "/archived/frozen-project/data/weather",
                "trace_dir": "/archived/frozen-project/data/workload",
                "dt_seconds": 5,
                "episode_ticks": 17280,
            },
            "reward": {"alpha": 1, "beta": 2},
            "default": {"bess_soc_init": 0.5},
        }
        before = copy.deepcopy(config)
        result = release.rebase_configuration(config, self.root, self.index)
        for key in self.index["runtime"]["data_directories"]:
            self.assertTrue(Path(result["global"][key]).is_relative_to(self.root))
            result["global"][key] = config["global"][key]
        self.assertEqual(result, before)
        self.assertEqual(config, before)

    def test_rebase_refuses_unmapped_data_locations(self):
        config = {"global": {"energy_dir": "/another/project/data/energy"}}
        with self.assertRaisesRegex(release.ReleaseError, "Unmapped archival path"):
            release.rebase_configuration(config, self.root, self.index)

    def test_absolute_traversal_and_external_symlinks_are_rejected(self):
        for name in ("/outside/file", "../escape", "a/../../b", "C:\\outside"):
            with self.subTest(name=name), self.assertRaises(release.ReleaseError):
                release.contained_path(self.root, name)
        outside = self.home / "outside.txt"
        outside.write_text("outside checkout\n")
        (self.root / "link").symlink_to(outside)
        with self.assertRaisesRegex(release.ReleaseError, "symlink escapes"):
            release.contained_path(self.root, "link")

    def test_archive_root_is_only_a_label_not_an_accessible_fallback(self):
        self.assertEqual(
            release.archive_relative(
                "/archived/frozen-project/models/selected.zip", self.index
            ),
            "models/selected.zip",
        )
        for name in (
            "/archived/frozen-project-evil/models/selected.zip",
            "/archived/frozen-project/../escape",
            "models/selected.zip",
        ):
            with self.subTest(name=name), self.assertRaises(release.ReleaseError):
                release.archive_relative(name, self.index)

    def test_evaluation_output_must_be_inside_a_new_checkout(self):
        self.assertEqual(
            release._output_path(self.root, Path("replays/case"), self.index),
            self.root / "replays/case",
        )
        with self.assertRaisesRegex(release.ReleaseError, "inside the new checkout"):
            release._output_path(self.root, self.home / "outside", self.index)
        same = {**self.index, "archive_root_label": str(self.root)}
        with self.assertRaisesRegex(release.ReleaseError, "NEW checkout"):
            release._output_path(self.root, Path("replays/case"), same)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"family": "guided_sac", "macro": "rule"}
        self.cell = {"start_day": 207, "seed": 100, "regd_load_bias": 0.5}

    def test_rule_floor_is_diagnostic_not_a_new_quota(self):
        metrics = release.extract_metrics(episode(self.policy, self.cell, ratio=0.5))
        self.assertTrue(metrics["publication_qualified"])
        self.assertFalse(metrics["delivered_service_gate"])
        self.assertFalse(metrics["floor_enforced"])

    def test_full_residual_floor_is_enforced_for_non_nominal(self):
        self.policy["macro"] = "full_residual"
        metrics = release.extract_metrics(episode(self.policy, self.cell, ratio=0.89))
        self.assertTrue(metrics["physical_gate_qualified"])
        self.assertFalse(metrics["publication_qualified"])
        self.assertTrue(metrics["floor_enforced"])

    def test_full_residual_floor_never_defaults_to_pass(self):
        self.policy["macro"] = "full_residual"
        with self.assertRaisesRegex(release.ReleaseError, "without its actual floor"):
            release.extract_metrics(episode(self.policy, self.cell))

    def test_nominal_full_residual_remains_reference_only(self):
        self.policy.update(macro="full_residual", family="nominal")
        metrics = release.extract_metrics(episode(self.policy, self.cell, ratio=0.1))
        self.assertTrue(metrics["publication_qualified"])
        self.assertFalse(metrics["floor_enforced"])

    def test_incomplete_failure_does_not_gain_full_day_metrics_or_recovery(self):
        metrics = release.extract_metrics(
            episode(self.policy, self.cell, complete=False, qualified=False, ratio=0.99)
        )
        self.assertFalse(metrics["publication_qualified"])
        self.assertFalse(metrics["complete_26h"])
        self.assertEqual(metrics["recovery_steps"], 0)
        for key in (
            "tracking_rmse_kw",
            "stress_facility_relief_mw",
            "battery_throughput_26h_mwh",
            "cashflow_26h_usd",
        ):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics["service_prefix_rmse_kw"], 12.5)

    def test_failed_service_soc_is_not_rescued_by_closed_recovery(self):
        value = episode(self.policy, self.cell, qualified=False)
        value["service_checks"]["terminal_soc"] = False
        metrics = release.extract_metrics(value)
        self.assertTrue(metrics["recovery_closed"])
        self.assertTrue(metrics["complete_26h"])
        self.assertFalse(metrics["publication_qualified"])

    def test_numeric_tolerances_do_not_relax_discrete_gates_or_missing_values(self):
        self.assertEqual(release.compare_metrics({"x": 1.0}, {"x": 1.0000001}), [])
        for expected, actual in (
            ({"x": False}, {"x": True}),
            ({"x": False}, {"x": 0}),
            ({"x": None}, {"x": 0.0}),
            ({"steps": 17280}, {"steps": 17279}),
            ({"x": 1.0}, {"x": float("nan")}),
            ({"x": 1.0}, {"x": 1.01}),
        ):
            with self.subTest(actual=actual):
                self.assertTrue(release.compare_metrics(expected, actual))

    def test_cell_validation_rejects_bad_types_and_extra_fields(self):
        for change in (
            {"start_day": True},
            {"seed": -1},
            {"seed": 1.2},
            {"regd_load_bias": float("nan")},
            {"regd_load_bias": True},
            {"extra": "arbitrary"},
        ):
            with self.subTest(change=change), self.assertRaises(release.ReleaseError):
                release.normalize_cell({**self.cell, **change})

    def test_native_four_command_families_are_not_silently_full_work_wrapped(self):
        for family in ("old_sac", "pid", "rule_hardware", "bang_bang"):
            self.assertNotIn(family, release.FULL_WORK)
        for family in (
            "cooling_first",
            "bc_only",
            "guided_sac",
            "union_original",
            "union_conditioned",
        ):
            self.assertIn(family, release.FULL_WORK)


class BundledInventoryTests(unittest.TestCase):
    def test_original_selected_identities_and_promotion_rules(self):
        index = release.inventory()
        self.assertEqual(
            index["original_head"], "34e015461a4c7e6a4fc1df8b61a72af92fc2db7a"
        )
        self.assertEqual(len(index["models"]), 25)
        for path, model in index["models"].items():
            if model["family"] in ("union_original", "union_conditioned"):
                self.assertTrue(path.endswith("step_552960.zip"))
                self.assertEqual(model["checkpoint_num_timesteps"], 552960)
                self.assertFalse(
                    model["context_identity"]["potential_used_at_deployment"]
                )
            elif model["family"] == "old_sac":
                self.assertEqual(model["training_seed"], 1007)
                self.assertEqual(model["checkpoint_num_timesteps"], 250002)
                self.assertEqual(
                    model["sha256"],
                    "5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024",
                )
            elif model["family"] in ("fixed_reference_kl", "replay_reference_kl"):
                self.assertTrue(model["not_promoted"])
                self.assertTrue(path.endswith("step_69120.zip"))
        guided = {
            model["training_seed"]: model["checkpoint_num_timesteps"]
            for model in index["models"].values()
            if model["family"] == "guided_sac"
        }
        self.assertEqual(guided, {1100: 4096, 1101: 8192, 1102: 4096, 1103: 4096})

    def test_no_external_runtime_data_or_manuscript_dependency(self):
        index = release.inventory()
        self.assertEqual(
            index["external_scientific_inputs"][
                "required_for_runtime_not_in_original_git"
            ],
            [],
        )
        self.assertEqual(len(index["runtime"]["data_files"]), 5)
        for path in index["files"]:
            self.assertNotIn("NIPS26_Data_Center_Energy_Grid_Benchmark", path)
            self.assertNotIn("AAAI", path)

    def test_failure_representatives_are_retained(self):
        rows = release.inventory()["representatives"]
        failures = [row for row in rows if not row["metrics"]["publication_qualified"]]
        self.assertGreaterEqual(len(failures), 3)
        self.assertTrue(any(not row["metrics"]["complete"] for row in failures))
        self.assertTrue(
            any(
                row["metrics"]["complete_26h"]
                and not row["metrics"]["service_checks"]["terminal_soc"]
                for row in failures
            )
        )


@unittest.skipUnless(
    os.environ.get("DC2GRID_RELEASE_ROOT"),
    "Set DC2GRID_RELEASE_ROOT for read-only hash/receipt integration checks",
)
class ReadOnlyCheckoutTests(unittest.TestCase):
    def test_representative_preflight_is_portable_and_does_not_evaluate(self):
        root = Path(os.environ["DC2GRID_RELEASE_ROOT"]).resolve()
        index = release.inventory()
        for row in index["representatives"]:
            with self.subTest(case=row["id"]):
                plan = release.prepare(root, row["policy_spec"], row["cell"])
                self.assertEqual(
                    release.compare_metrics(row["metrics"], plan["expected_metrics"]),
                    [],
                )
                self.assertTrue(
                    Path(plan["runtime_protocol"]["config_path"]).is_relative_to(root)
                )


if __name__ == "__main__":
    unittest.main()
