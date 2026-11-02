from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PYTHON = sys.executable
def _run_ha_benchmark_with_fixed_actions(
    output_path: Path,
    fixed_actions: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    cmd = [
        PYTHON,
        "evaluation/run_ha_benchmark.py",
        "--agents",
        "random",
        "--scenarios",
        "default",
        "--n_episodes",
        "1",
        "--seed",
        "123",
        "--output",
        str(output_path),
        "--no-record_transitions",
    ]
    if fixed_actions:
        cmd += ["--fixed-action", *fixed_actions]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_benchmark_with_fixed_actions(
    fixed_actions: tuple[str, ...],
    output_path: Path,
) -> subprocess.CompletedProcess:
    cmd = [
        PYTHON,
        "evaluation/run_benchmark.py",
        "--agents",
        "random",
        "--scenarios",
        "default",
        "--n_episodes",
        "1",
        "--seed",
        "123",
        "--output",
        str(output_path),
        "--no-record_transitions",
    ]
    if fixed_actions:
        cmd += ["--fixed-action", *fixed_actions]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_cmd(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.smoke
class TestActionAblationSmoke:
    def test_run_benchmark_with_fixed_actions_only(self, tmp_path: Path):
        output_path = tmp_path / "ablation_fixed_only.csv"
        result = _run_benchmark_with_fixed_actions(
            fixed_actions=("hvac_effort=0.7",),
            output_path=output_path,
        )

        assert result.returncode == 0, (
            f"run_benchmark.py failed for fixed-only ablation with code {result.returncode}.\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        assert output_path.exists(), f"Expected output CSV was not created: {output_path}"

    def test_run_ha_benchmark_with_fixed_actions_only(self, tmp_path: Path):
        output_path = tmp_path / "ha_ablation_fixed_only.csv"
        result = _run_ha_benchmark_with_fixed_actions(
            fixed_actions=("hvac_effort=0.7",),
            output_path=output_path,
        )

        assert result.returncode == 0, (
            f"run_ha_benchmark.py failed for fixed-only ablation with code {result.returncode}.\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        assert output_path.exists(), f"Expected output CSV was not created: {output_path}"


class TestAblationCliValidation:
    def test_run_benchmark_rejects_invalid_fixed_action_value(self):
        result = _run_cmd(
            [
                "evaluation/run_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--fixed-action", "hvac_effort=1.5",
            ]
        )
        assert result.returncode != 0
        assert "Invalid fixed value for action 'hvac_effort'" in (result.stderr + result.stdout)

    def test_run_benchmark_rejects_unknown_fixed_action_name(self):
        result = _run_cmd(
            [
                "evaluation/run_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--fixed-action", "invalid_act=0.2",
            ]
        )
        assert result.returncode != 0
        assert "Invalid action 'invalid_act' in --fixed-action" in (result.stderr + result.stdout)

    def test_run_ha_benchmark_rejects_invalid_fixed_action_value(self):
        result = _run_cmd(
            [
                "evaluation/run_ha_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--fixed-action", "bess_dispatch=-1.2",
            ]
        )
        assert result.returncode != 0
        assert "Invalid fixed value for action 'bess_dispatch'" in (result.stderr + result.stdout)

    def test_run_ha_benchmark_rejects_unknown_fixed_action_name(self):
        result = _run_cmd(
            [
                "evaluation/run_ha_benchmark.py",
                "--agents", "random",
                "--scenarios", "default",
                "--n_episodes", "1",
                "--no-record_transitions",
                "--fixed-action", "invalid_act=0.2",
            ]
        )
        assert result.returncode != 0
        assert "Invalid action 'invalid_act' in --fixed-action" in (result.stderr + result.stdout)
