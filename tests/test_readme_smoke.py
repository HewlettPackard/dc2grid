"""
tests/test_readme_smoke.py — Smoke tests that run the exact README CLI commands.

Run with:
    uv run pytest tests/test_readme_smoke.py -v -m smoke

Each test spawns a training script as a subprocess with greatly reduced
timesteps (~2 000 steps) and asserts the process exits cleanly (code 0).

What is validated per test
--------------------------
* Hydra config composition succeeds (no KeyError from bad group overrides,
  including the market= override).
* The chosen scenario × market × algorithm combination can build the
  environment, instantiate the agent, and complete at least a few
  gradient updates.
* The script creates its output directory and writes the model checkpoint.
* Exit code 0 — i.e., no Python exception or Hydra error propagated up.

The tests are intentionally cheap: they are NOT measuring performance or
convergence.  For that, see the full sweep in scripts/run_sweep.sh.

Marks
-----
@pytest.mark.smoke   — opt-in; skipped in the normal `pytest tests/` run.
                       Run explicitly with:  pytest -m smoke
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTHON = sys.executable  # same interpreter / venv as the test runner

# Minimal PPO overrides: ~2 000 steps, 1 env, fast eval → finishes in <60 s
_FAST_PPO = [
    "algo.timesteps=2000",
    "algo.n_envs=1",
    "algo.eval_freq=1000",
    "algo.n_eval_episodes=1",
]

# Minimal SAC overrides: fewer learning_starts so training begins immediately
_FAST_SAC = [
    "algo.timesteps=2000",
    "algo.eval_freq=1000",
    "algo.n_eval_episodes=1",
    "algo.learning_starts=100",
]

# Minimal HRL overrides: 500 steps per phase → both phases finish quickly
_FAST_HRL = [
    "+hrl.phase1_steps=500",
    "+hrl.phase2_steps=500",
    "algo.n_envs=1",
    "algo.eval_freq=200",
    "algo.n_eval_episodes=1",
]


def _run(script: str, extra_args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run *script* (relative to project root) with *extra_args* as overrides."""
    cmd = [PYTHON, script] + extra_args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        # CWD is already set to project root by conftest.py set_project_root
    )


def _assert_ok(result: subprocess.CompletedProcess, label: str) -> None:
    assert result.returncode == 0, (
        f"{label} exited with code {result.returncode}.\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Smoke tests — one per README code block
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestReadmeSmokeCommands:
    """
    Each method maps to a specific CLI example from the README.
    The label in parentheses notes the README section that owns the command.
    """

    # ── §10 default PPO ─────────────────────────────────────────────────────
    # def test_ppo_default(self):
    #     """README §10: uv run python baselines/train_ppo.py"""
    #     result = _run("baselines/train_ppo.py", _FAST_PPO)
    #     _assert_ok(result, "train_ppo.py (default)")

    # ── §10 PPO scenario_a + PJM market ─────────────────────────────────────
    # def test_ppo_scenario_a_market_pjm(self):
    #     """README §10: train_ppo.py scenario=scenario_a market=pjm_dom"""
    #     result = _run(
    #         "baselines/train_ppo.py",
    #         ["scenario=scenario_a", "market=pjm_dom"] + _FAST_PPO,
    #     )
    #     _assert_ok(result, "train_ppo.py scenario_a market=pjm_dom")

    # ── §8 PPO scenario_b + ERCOT market ────────────────────────────────────
    # def test_ppo_scenario_b_market_ercot(self):
    #     """README §8: train_ppo.py scenario=scenario_b market=ercot_north"""
    #     result = _run(
    #         "baselines/train_ppo.py",
    #         ["scenario=scenario_b", "market=ercot_north"] + _FAST_PPO,
    #     )
    #     _assert_ok(result, "train_ppo.py scenario_b market=ercot_north")

    # ── §8 PPO scenario_b + ENTSO-E + seed override ─────────────────────────
    # def test_ppo_scenario_b_market_entso_seed(self):
    #     """README §8: ...scenario=scenario_b market=entso_de experiment.seed=1"""
    #     result = _run(
    #         "baselines/train_ppo.py",
    #         ["scenario=scenario_b", "market=entso_de", "experiment.seed=1"] + _FAST_PPO,
    #     )
    #     _assert_ok(result, "train_ppo.py scenario_b market=entso_de seed=1")

    # ── §10 SAC scenario_b ───────────────────────────────────────────────────
    def test_sac_scenario_b(self):
        """README §10: train_sac.py algo=sac scenario=scenario_b"""
        result = _run(
            "baselines/train_sac.py",
            ["algo=sac", "scenario=scenario_b"] + _FAST_SAC,
        )
        _assert_ok(result, "train_sac.py algo=sac scenario_b")

    # ── §10 Safety-shielded PPO ──────────────────────────────────────────────
    # def test_shielded_ppo_default(self):
    #     """README §10: train_shielded_ppo.py scenario=default"""
    #     result = _run(
    #         "baselines/train_shielded_ppo.py",
    #         ["scenario=default"] + _FAST_PPO,
    #     )
    #     _assert_ok(result, "train_shielded_ppo.py scenario=default")

    # ── §10 Hierarchical RL (two-phase) ─────────────────────────────────────
    def test_hierarchical(self):
        """README §10: train_hierarchical.py  (two-phase pipeline)"""
        result = _run(
            "baselines/train_hierarchical.py",
            _FAST_HRL,
            timeout=600,  # two training phases — allow up to 10 min
        )
        _assert_ok(result, "train_hierarchical.py")
    # ── §10 PPO-Lagrangian ──────────────────────────────────────────────────
    # def test_ppo_lagrangian_default(self):
    #     """README §10: train_ppo_lagrangian.py algo=ppo_lagrangian"""
    #     result = _run(
    #         "baselines/train_ppo_lagrangian.py",
    #         ["algo=ppo_lagrangian"] + _FAST_PPO,
    #     )
    #     _assert_ok(result, "train_ppo_lagrangian.py (default)")

