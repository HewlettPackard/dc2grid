import copy
import csv
import io
import json
import unittest
from pathlib import Path

from reproducibility import table1_release as release


class Table1ReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verification = release.verify_export()
        cls.original = release.producer_protocol()
        cls.high_failure = release.read_json(
            release.STUDY / "cases/sac_macro__sac/seed_100/episode.json"
        )

    def test_exact_source_snapshot_models_and_case_inventory(self):
        record = self.verification
        self.assertEqual(record["source_commit"], release.SOURCE_COMMIT)
        self.assertEqual(record["episode_receipts"], 35)
        self.assertEqual(record["complete_service"], 35)
        self.assertEqual(record["qualified_supplemental"], 31)
        self.assertEqual(record["native_source_files"], 61)
        self.assertEqual(record["checkpoints"]["low"]["selected_timesteps"], 250002)
        self.assertEqual(record["checkpoints"]["high"]["selected_timesteps"], 8004)

    def test_original_manuscript_tex_is_not_exported(self):
        self.assertFalse((release.STUDY / "original_table1.json").exists())
        numeric = release.read_json(release.STUDY / "original_table1.numeric.json")
        self.assertNotIn("table_tex", numeric)
        self.assertEqual(len(numeric["rows"]), 9)

    def test_execution_mapping_changes_only_approved_paths_and_provenance(self):
        mapped = release.execution_protocol()
        self.assertNotEqual(mapped["registration_id"], release.SOURCE_REGISTRATION)
        self.assertEqual(
            mapped["public_replay"]["producer_registration_id"],
            release.SOURCE_REGISTRATION,
        )
        self.assertEqual(mapped["public_replay"]["changed_scientific_fields"], [])
        for name in (
            "panel",
            "policies",
            "source_files",
            "data_files",
            "service_gate",
            "recovery",
            "aggregation",
        ):
            self.assertEqual(mapped[name], self.original[name])
        for role, model in mapped["models"].items():
            self.assertTrue(Path(model["path"]).is_relative_to(release.ROOT))
            self.assertEqual(model["sha256"], self.original["models"][role]["sha256"])

    def test_original_registration_identity_is_verified(self):
        original = self.original
        body = {
            key: value for key, value in original.items() if key != "registration_id"
        }
        self.assertEqual(release.identity(body), release.SOURCE_REGISTRATION)

    def test_published_recovery_and_panel_contract_match_the_frozen_evaluator(self):
        evaluator = release.study_module("table1_eval")
        self.assertEqual(evaluator.RECOVERY_SPEC, self.original["recovery"])
        self.assertEqual(evaluator.PANEL_SPEC, self.original["panel"])
        self.assertEqual(evaluator.SERVICE_GATE_SPEC, self.original["service_gate"])

    def test_closed_policy_and_seed_panel(self):
        release.validate_request("rule_macro+sac", 100)
        for policy, seed in (
            ("ppo_macro+sac", 100),
            ("rule_macro+sac", 1007),
            ("rule_macro+sac", True),
            ("rule_macro+sac", 99),
        ):
            with self.subTest(policy=policy, seed=seed):
                with self.assertRaises(ValueError):
                    release.validate_request(policy, seed)

    def test_replay_cannot_replace_the_frozen_case_directory(self):
        with self.assertRaisesRegex(ValueError, "below"):
            release.replay_output(release.STUDY / "cases/rule_macro__sac/seed_100")
        with self.assertRaisesRegex(ValueError, "below"):
            release.replay_output(Path("/tmp/table1-unscoped-output"))

    def test_new_replay_directory_is_within_its_declared_namespace(self):
        path = release.replay_output(
            release.STUDY / "replays/unit-test-path-not-created"
        )
        self.assertTrue(path.is_relative_to(release.STUDY / "replays"))
        self.assertFalse(path.exists())

    def test_a_reproduced_physical_gate_failure_is_not_relabelled(self):
        result = release.compare_episode(
            copy.deepcopy(self.high_failure), self.high_failure
        )
        self.assertTrue(result["verified"])
        self.assertTrue(result["service_complete"])
        self.assertFalse(result["qualified_supplemental"])
        self.assertTrue(result["source_failure_outcome_preserved"])

    def test_changed_qualification_fails(self):
        changed = copy.deepcopy(self.high_failure)
        changed["supplemental_accounting"]["all_supplemental_gates_pass"] = True
        with self.assertRaisesRegex(ValueError, "Discrete result"):
            release.compare_episode(changed, self.high_failure)

    def test_short_service_horizon_fails(self):
        changed = copy.deepcopy(self.high_failure)
        changed["service"]["observed_fast_ticks"] -= 1
        with self.assertRaisesRegex(ValueError, "Discrete result"):
            release.compare_episode(changed, self.high_failure)

    def test_changed_primary_result_fails(self):
        changed = copy.deepcopy(self.high_failure)
        changed["service"]["table1_service_metrics"]["tracking_rmse_kw"] += 1
        with self.assertRaisesRegex(ValueError, "Numeric result"):
            release.compare_episode(changed, self.high_failure)

    def test_recovery_state_preservation_is_not_skipped(self):
        changed = copy.deepcopy(self.high_failure)
        changed["supplemental_recovery"]["extension"]["state_digest_after"] = "changed"
        with self.assertRaisesRegex(ValueError, "preserve"):
            release.compare_episode(changed, self.high_failure)

    def test_all_pdf_csv_values_and_not_rerun_rows_are_preserved(self):
        outputs, values = release.numerical_views()
        rows = list(csv.DictReader(io.StringIO(outputs["table1_cells.csv"])))
        self.assertEqual(len(rows), 96)
        self.assertEqual(sum(row["mean"] != "" for row in rows), 93)
        self.assertEqual(len(values["corrected"]), 7)
        self.assertEqual(len(values["archived"]), 9)
        self.assertEqual(set(values["not_rerun"]), {"ppo_macro+sac", "llm_macro+sac"})
        self.assertEqual(
            json.loads(outputs["table1_values.json"])["qualified_supplemental"], 31
        )


if __name__ == "__main__":
    unittest.main()
