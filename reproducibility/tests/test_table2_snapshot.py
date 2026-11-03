import csv
import io
import json
import unittest

from reproducibility import table2_snapshot as snapshot


class Table2SnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = snapshot.build()
        cls.record = json.loads(cls.outputs["table2_snapshot.json"])

    def test_only_the_two_updated_table2_parts_are_exported(self):
        rows = list(csv.DictReader(io.StringIO(self.outputs["table2_cells.csv"])))
        self.assertEqual(len(rows), 18)
        self.assertEqual({row["table_id"] for row in rows}, {"T2a", "T2b"})
        self.assertEqual(len(self.record["families"]), 9)

    def test_all_seventeen_exact_checkpoints_are_identified(self):
        models = self.record["checkpoints"]
        self.assertEqual(len(models), 17)
        self.assertEqual(len({model["path"] for model in models}), 17)
        low = next(model for model in models if model["id"] == "corrected_sac_1007")
        self.assertEqual(low["selected_step"], 250002)
        self.assertEqual(
            low["sha256"],
            "5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024",
        )
        self.assertTrue(all("/high/" not in model["path"] for model in models))
        self.assertTrue(all(model["bytes"] > 0 for model in models))

    def test_complete_replay_matrix_has_no_duplicate_or_unplanned_cell(self):
        rows = [
            json.loads(line) for line in self.outputs["replay_cases.jsonl"].splitlines()
        ]
        self.assertEqual(len(rows), 630)
        keys = {
            (
                row["snapshot"],
                row["policy"],
                row["start_day"],
                row["seed"],
                row["regd_load_bias"],
            )
            for row in rows
        }
        self.assertEqual(len(keys), 630)
        self.assertEqual({row["start_day"] for row in rows}, {207, 248, 249})
        self.assertEqual({row["seed"] for row in rows}, set(range(100, 105)))
        self.assertEqual({row["regd_load_bias"] for row in rows}, {0.0, 0.5})
        self.assertTrue(all(row["policy"].startswith("rule__") for row in rows))

    def test_pid_failures_and_different_mean_gate_scopes_are_preserved(self):
        self.assertEqual(self.record["mean_bias"], 0.5)
        self.assertEqual(self.record["qualification_biases"], [0.0, 0.5])
        for table in self.record["tables"]:
            pid = next(row for row in table["rows"] if row["family"] == "pid")
            self.assertEqual(pid["pass"], [26, 30])
            self.assertTrue(pid["diagnostic"])
            self.assertEqual(pid["pass_by_bias"]["0.5"], [14, 15])

    def test_no_new_training_or_results_are_claimed(self):
        self.assertFalse(self.record["training_performed"])
        self.assertFalse(self.record["new_evaluation_performed"])
        self.assertFalse(self.record["original_archive_or_input_datasets_included"])


if __name__ == "__main__":
    unittest.main()
