from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np
import pytest

from c2g_env import C2GFastEnv
from c2g_env.env_high_level import C2GMacroEnv

"""
tests/test_datalogging.py  —  Sanity checks for logged state-action-reward data from benchmark runs
"""

# Derive expected dimensions from the environment itself
_EXPECTED_OBS_DIM = C2GFastEnv(scenario="default").observation_space.shape[0]
_EXPECTED_ACT_DIM = C2GFastEnv(scenario="default").action_space.shape[0]

_EXPECTED_OBS_DIM_MACRO = C2GMacroEnv(scenario="default").observation_space.shape[0]
_EXPECTED_ACT_DIM_MACRO = C2GMacroEnv(scenario="default").action_space.shape[0]

class TestExperimentTransitionLogging:
    def test_capacity_and_thermal_conditions_have_distinct_logs(self, tmp_path):
        from baselines.metrics_callback import C2GTransitionLoggerCallback

        contexts = [
            {"plant_profile": "plant_50mw", "thermal_overrides": {}},
            {"plant_profile": "plant_500mw", "thermal_overrides": {}},
            {"plant_profile": "plant_50mw", "thermal_overrides": {"C_A": 2700.0}},
        ]
        for context in contexts:
            logger = C2GTransitionLoggerCallback(
                output_dir=tmp_path, algorithm_name="rule_based", scenario_name="default",
                episode_number=0, experiment_context=context,
            )
            try:
                logger.record_transition(np.zeros(18), np.zeros(4), np.zeros(18), 1.0, True)
            finally:
                logger.close()
        assert len(list(tmp_path.rglob("episode0.csv"))) == 3
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.rglob("experiment.json")]
        assert all(context in manifests for context in contexts)

    def test_existing_context_episode_is_not_overwritten(self, tmp_path):
        from baselines.metrics_callback import C2GTransitionLoggerCallback

        kwargs = dict(output_dir=tmp_path, algorithm_name="rule_based", scenario_name="default",
                      episode_number=0, experiment_context={"plant_profile": "plant_50mw"})
        logger = C2GTransitionLoggerCallback(**kwargs)
        logger.record_transition(np.zeros(18), np.zeros(4), np.zeros(18), 1.0, True)
        logger.close()
        path = next(tmp_path.rglob("episode0.csv"))
        original = path.read_bytes()
        duplicate = C2GTransitionLoggerCallback(**kwargs)
        try:
            with pytest.raises(FileExistsError):
                duplicate.record_transition(np.zeros(18), np.zeros(4), np.zeros(18), 2.0, True)
        finally:
            duplicate.close()
        assert path.read_bytes() == original

class TestRuntimeTransitionLogging:
    def test_runtime_preserves_reset_nested_info_and_end_flags(self, tmp_path):
        import gzip
        from baselines.metrics_callback import C2GTransitionLoggerCallback

        path = tmp_path / "hardware.jsonl.gz"
        logger = C2GTransitionLoggerCallback(runtime_path=path, experiment_context={"episode_id": "test"})
        info = {"power": np.float64(2.0), "flag": np.bool_(True), "nested": {"value": 3.0},
                "new_field": np.array([1.0, 2.0]), "last_inner_info": {"power": 99.0}}
        try:
            logger.record_runtime_reset(np.zeros(18), {"reset_field": "present"}, {"bess_age_frac": 0.1})
            logger.record_runtime_transition(
                np.zeros(18), np.zeros(4), np.ones(18), 1.0, False, True, info,
                hardware_step=1, macro_step=1, duration_seconds=5.0,
            )
            summary = logger.runtime_summary()
        finally:
            logger.close()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        assert len(records) == 2
        assert records[0]["info"] == {"reset_field": "present"}
        assert records[1]["info"]["new_field"] == [1.0, 2.0]
        assert records[1]["info"]["last_inner_info"] == {"power": 99.0}
        assert records[1]["terminated"] is False and records[1]["truncated"] is True
        assert summary["steps"] == 1
        assert summary["scalars"]["info.nested.value"]["mean"] == 3.0
        assert summary["scalars"]["info.flag"]["first_true_step"] == 1
        assert "info.last_inner_info.power" not in summary["scalars"]

    def test_runtime_summary_weights_durations_and_preserves_missing_fields(self, tmp_path):
        from baselines.metrics_callback import C2GTransitionLoggerCallback

        logger = C2GTransitionLoggerCallback(
            runtime_path=tmp_path / "macro.jsonl.gz", agent_type="macro", experiment_context={"episode_id": "test"},
        )
        try:
            for step, power, duration in [(1, 2.0, 5.0), (2, 4.0, 15.0)]:
                info = {"power": power, "flag": step == 2, "optional": 3.0 if step == 1 else None}
                logger.record_runtime_transition(
                    np.zeros(19), np.zeros(2), np.zeros(19), 1.0, False, step == 2, info,
                    hardware_step=step, macro_step=step, duration_seconds=duration,
                )
            summary = logger.runtime_summary()
        finally:
            logger.close()
        power = summary["scalars"]["info.power"]
        assert power["mean"] == 3.0
        assert power["duration_weighted_mean"] == 3.5
        assert power["integral_seconds"] == 70.0
        assert power["rms"] == pytest.approx(np.sqrt(10.0))
        assert summary["scalars"]["info.flag"]["true_count"] == 1
        assert summary["scalars"]["info.flag"]["first_true_step"] == 2
        assert summary["scalars"]["info.optional"]["count"] == 1

    def test_runtime_rejects_nonfinite_records(self, tmp_path):
        from baselines.metrics_callback import C2GTransitionLoggerCallback

        logger = C2GTransitionLoggerCallback(runtime_path=tmp_path / "bad.jsonl.gz", experiment_context={"episode_id": "test"})
        try:
            with pytest.raises(ValueError):
                logger.record_runtime_reset(np.array([np.nan]), {}, {})
        finally:
            logger.close()


class TestDataLogging:
    def test_eval_log_schema_and_continuity(self):
        # sanity check: previous state = next observation in log
        runs_dir = Path("runs")
        if not runs_dir.exists() or not runs_dir.is_dir():
            return

        subfolders = [p for p in runs_dir.iterdir() if p.is_dir()]
        non_empty_subfolders = [p for p in subfolders if any(p.iterdir())]

        for folder in non_empty_subfolders:
            for csv_path in folder.rglob("episode*.csv"):
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                fieldnames = reader.fieldnames or []
                
                # Extract state columns with semantic names (s_temp_A_norm, s_bess_soc, etc.)
                s_cols = sorted([name for name in fieldnames if name.startswith("s_")])

                # For every consecutive pair (prev_row, curr_row) from index 1 to
                # the last row: curr_row state must equal prev_row observation.
                for row_idx, (prev_row, curr_row) in enumerate(zip(rows, rows[1:]), start=1):
                    for s_col in s_cols:
                        o_col = s_col.replace("s_", "o_", 1)
                        
                        if o_col not in fieldnames:
                            raise Exception(f"Expected observation column {o_col} corresponding to state column {s_col} not found in {csv_path.name}")
                        
                        assert float(curr_row[s_col]) == pytest.approx(float(prev_row[o_col]), abs=1e-6), (
                            f"Transition mismatch in {csv_path.name} at row {row_idx}, {s_col}: "
                            f"{s_col}={curr_row[s_col]} vs prev {o_col}={prev_row[o_col]}"
                        )

    def test_datalogging_column_schema_hardware(self):
        """
        Verify that every non-macro CSV file in runs/ has the correct column schema:
        - {obs_dim} o_* columns (observations)
        - {obs_dim} s_* columns (states)
        - {act_dim} a_* columns (actions)
        - 7 r_* columns (reward components)
        - 1 r column (total reward)
        """
        runs_dir = "runs"
        
        # Exit if runs/ doesn't exist
        if not os.path.exists(runs_dir) or not os.path.isdir(runs_dir):
            pytest.skip("runs/ directory does not exist")
        
        # Recursively walk through runs/ and collect all CSV files
        all_csv_files = []
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(full_path)).lower()
                if file.endswith(".csv") and not folder_name.endswith("_macro"):
                    all_csv_files.append(full_path)
        
        # Exit if no CSV files found
        if not all_csv_files:
            pytest.skip("No CSV files found in runs/")
        
        obs_dim = _EXPECTED_OBS_DIM
        act_dim = _EXPECTED_ACT_DIM
        errors = []
        
        for csv_path in all_csv_files:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
            
            # Count columns by prefix
            o_cols = [col for col in fieldnames if col.startswith("o_")]
            s_cols = [col for col in fieldnames if col.startswith("s_")]
            a_cols = [col for col in fieldnames if col.startswith("a_")]
            r_component_cols = [col for col in fieldnames if col.startswith("r_")]
            r_total_cols = [col for col in fieldnames if col == "r"]
            
            file_errors = []
            
            if len(o_cols) != obs_dim:
                file_errors.append(f"o_* columns: expected {obs_dim}, found {len(o_cols)}")
            if len(s_cols) != obs_dim:
                file_errors.append(f"s_* columns: expected {obs_dim}, found {len(s_cols)}")
            if len(a_cols) != act_dim:
                file_errors.append(f"a_* columns: expected {act_dim}, found {len(a_cols)}")
            if len(r_component_cols) != 7:
                file_errors.append(f"r_* columns: expected 7, found {len(r_component_cols)}")
            if len(r_total_cols) != 1:
                file_errors.append(f"r column: expected 1, found {len(r_total_cols)}")
            
            if file_errors:
                rel_path = os.path.relpath(csv_path, runs_dir)
                error_msg = f"{rel_path}: " + "; ".join(file_errors)
                errors.append(error_msg)
        
        if errors:
            error_report = "\n".join(errors)
            raise AssertionError(f"Column schema violations found:\n{error_report}")

    def test_datalogging_column_schema_macro(self):
        """
        Verify that every macro CSV file in runs/ has the correct column schema:
        - {obs_dim} o_* columns (observations)
        - {obs_dim} s_* columns (states)
        - {act_dim} a_* columns (actions)
        - 7 r_* columns (reward components)
        - 1 r column (total reward)
        """
        runs_dir = "runs"

        # Exit if runs/ doesn't exist
        if not os.path.exists(runs_dir) or not os.path.isdir(runs_dir):
            pytest.skip("runs/ directory does not exist")

        # Recursively walk through runs/ and collect only macro CSV files
        all_csv_files = []
        for root, dirs, files in os.walk(runs_dir):
            for file in files:
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(full_path)).lower()
                if file.endswith(".csv") and folder_name.endswith("_macro"):
                    all_csv_files.append(full_path)

        # Exit if no macro CSV files found
        if not all_csv_files:
            pytest.skip("No macro CSV files found in runs/")

        obs_dim = _EXPECTED_OBS_DIM_MACRO
        act_dim = _EXPECTED_ACT_DIM_MACRO
        errors = []

        for csv_path in all_csv_files:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

            # Count columns by prefix
            o_cols = [col for col in fieldnames if col.startswith("o_")]
            s_cols = [col for col in fieldnames if col.startswith("s_")]
            a_cols = [col for col in fieldnames if col.startswith("a_")]
            r_component_cols = [col for col in fieldnames if col.startswith("r_")]
            r_total_cols = [col for col in fieldnames if col == "r"]

            file_errors = []

            if len(o_cols) != obs_dim:
                file_errors.append(f"o_* columns: expected {obs_dim}, found {len(o_cols)}")
            if len(s_cols) != obs_dim:
                file_errors.append(f"s_* columns: expected {obs_dim}, found {len(s_cols)}")
            if len(a_cols) != act_dim:
                file_errors.append(f"a_* columns: expected {act_dim}, found {len(a_cols)}")
            if len(r_component_cols) != 4:
                file_errors.append(f"r_* columns: expected 4, found {len(r_component_cols)}")
            if len(r_total_cols) != 1:
                file_errors.append(f"r column: expected 1, found {len(r_total_cols)}")

            if file_errors:
                rel_path = os.path.relpath(csv_path, runs_dir)
                error_msg = f"{rel_path}: " + "; ".join(file_errors)
                errors.append(error_msg)

        if errors:
            error_report = "\n".join(errors)
            raise AssertionError(f"Column schema violations found:\n{error_report}")

    def test_smoke_benchmark_rule_macro(self):
        """Smoke test: run_benchmark.py with rule_macro agent."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_benchmark.py",
             "--record_transitions", "--agents", "rule_macro", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_benchmark_rule_based(self):
        """Smoke test: run_benchmark.py with rule_based agent."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_benchmark_rule_based_with_bess_ablation(self):
        """Smoke test: run_benchmark.py with rule_based agent and BESS fixed to 0.5."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--fixed-action", "bess_dispatch=0.5", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_ha_benchmark_rule_based(self):
        """Smoke test: run_ha_benchmark.py with rule_based agent."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_ha_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    def test_smoke_ha_benchmark_rule_based_with_bess_ablation(self):
        """Smoke test: run_ha_benchmark.py with rule_based agent and BESS fixed to 0.5."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "evaluation/run_ha_benchmark.py",
             "--record_transitions", "--agents", "rule_based", "--fixed-action", "bess_dispatch=0.5", "--n_episodes", "1"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Command failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"