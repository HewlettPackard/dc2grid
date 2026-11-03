"""
evaluation/run_sachigh_saclow_benchmark.py
==========================================
Hierarchical benchmark: trained SAC macro agent + trained SAC low-level agent.

The macro agent (SAC trained on C2GMacroEnv) outputs 2-D actions
(bid_mw, price_eur_mwh) every 15-minute macro step.  At each of the
180 sub-steps within a macro tick, the low-level SAC agent (trained on
C2GFastEnv) maps the current 18-D low-level observation to 4-D hardware
actions (throttle_batch, pump_speed_A, hvac_effort, bess_dispatch).

Usage
-----
  cd <project_root>

  # Default paths: trained_models/sac_macro_default_s42  +  trained_models/sac_default_s42
  python evaluation/run_sachigh_saclow_benchmark.py

  # Custom model directories
  python evaluation/run_sachigh_saclow_benchmark.py \
      --macro-model-dir trained_models/sac_macro_default_s42 \
      --inner-model-dir trained_models/sac_default_s42

  # Specific scenarios
  python evaluation/run_sachigh_saclow_benchmark.py --scenarios default scenario_a

  # More episodes, different seed
  python evaluation/run_sachigh_saclow_benchmark.py --n_episodes 10 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from c2g_env import C2GFastEnv, C2GMacroEnv
from baselines.metrics_callback import C2GTransitionLoggerCallback

SCENARIOS = ["default", "scenario_a", "scenario_b", "scenario_c"]


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------

class SB3Agent:
    """Wraps a loaded SB3 model (PPO or SAC)."""

    def __init__(self, model, algo_name: str):
        self._model = model
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._model.predict(obs, deterministic=deterministic)


def _load_sac_model(
    model_dir: Path, label: str, *, simulation_mode: str = "physical_v1",
    allow_semantic_mismatch: bool = False,
) -> SB3Agent:
    """Load a SAC model from a directory or direct path.

    Accepts:
      - A directory containing final_model.zip  (e.g. trained_models/sac_macro_default_s42/)
      - A direct path to a .zip file            (e.g. .../best_model/final_model.zip)
      - A path without .zip extension           (e.g. .../best_model/final_model)
    """
    from stable_baselines3 import SAC

    model_dir = Path(model_dir)

    # Case 1: path is already a .zip file
    if model_dir.suffix == ".zip" and model_dir.exists():
        zip_path = model_dir
        load_path = str(model_dir.with_suffix(""))
    # Case 2: path without extension but .zip exists alongside
    elif model_dir.with_suffix(".zip").exists():
        zip_path = model_dir.with_suffix(".zip")
        load_path = str(model_dir)
    # Case 3: path is a directory containing final_model.zip
    elif (model_dir / "final_model.zip").exists():
        zip_path = model_dir / "final_model.zip"
        load_path = str(model_dir / "final_model")
    else:
        # Try the default and report error
        zip_path = model_dir / "final_model.zip"
        raise FileNotFoundError(
            f"No trained {label} model at {zip_path}"
        )

    model = SAC.load(load_path)
    from c2g_env.experiments.identity import check_policy_semantics
    check_policy_semantics(model, simulation_mode, allow_mismatch=allow_semantic_mismatch)
    return SB3Agent(model, algo_name=label)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_hierarchical_episode(
    macro_agent: SB3Agent,
    inner_agent: SB3Agent,
    scenario: str,
    seed: int,
    episode_number: int = 0,
    record_transitions: bool = True,
    simulation_mode: str = "physical_v1",
) -> dict[str, float]:
    """Run one macro episode with SAC-high driving SAC-low and return metrics."""

    # Build the inner-action callback that C2GMacroEnv calls every sub-step.
    def inner_action_fn(inner_obs: np.ndarray, _macro_action: np.ndarray) -> np.ndarray:
        action, _ = inner_agent.predict(inner_obs, deterministic=True)
        return action

    env = C2GMacroEnv(scenario=scenario, inner_action_fn=inner_action_fn, simulation_mode=simulation_mode)
    obs, _ = env.reset(seed=seed)

    combo_name = f"{macro_agent.algo_name}+{inner_agent.algo_name}"

    transition_logger = None
    if record_transitions:
        transition_logger = C2GTransitionLoggerCallback(
            output_dir="runs",
            algorithm_name=combo_name,
            scenario_name=scenario,
            agent_type="macro",
            episode_number=episode_number,
            fixed_action_values=None,
            verbose=0,
        )

    rewards: list[float] = []
    bids_accepted: list[float] = []
    reg_revenues: list[float] = []
    elec_costs: list[float] = []
    perf_scores: list[float] = []
    committed_mws: list[float] = []
    tracking_errs: list[float] = []
    tracking_steps = 0
    temp_a_maxes: list[float] = []
    temp_b_maxes: list[float] = []
    flex_reductions: list[float] = []
    bess_actuals: list[float] = []
    cool_deltas: list[float] = []
    p_pumps: list[float] = []
    p_hvacs: list[float] = []
    inner_throttles: list[float] = []
    inner_pumps: list[float] = []
    inner_hvacs: list[float] = []
    inner_besses: list[float] = []

    done = False
    while not done:
        state = obs.copy()
        action, _ = macro_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info.get(k, 0.0)
                for k in ("reward_regulation", "reward_sub", "reward_elec", "reward_churn")
            }
            transition_logger.record_transition(
                state=state,
                action=action,
                observation=obs,
                reward=float(reward),
                done=done,
                reward_components=reward_components,
            )

        rewards.append(float(reward))
        bids_accepted.append(float(info.get("bid_accepted", False)))
        reg_revenues.append(float(info.get("regulation_revenue", 0.0)))
        elec_costs.append(float(info.get("electricity_cost", 0.0)))
        perf_scores.append(float(info.get("perf_score", 0.0)))
        committed_mws.append(float(info.get("committed_mw", 0.0)))
        if env._physical:
            tracking_errs.append(info["tracking_error_sq_sum"])
            tracking_steps += info["sub_steps_run"]
        else:
            tracking_errs.append(float(info.get("mean_tracking_err", 0.0)) ** 2)
        temp_a_maxes.append(float(info.get("temp_A_max", 0.0)))
        temp_b_maxes.append(float(info.get("temp_B_max", 0.0)))
        flex_reductions.append(float(info.get("mean_flex_reduction_kw", 0.0)))
        bess_actuals.append(float(info.get("mean_bess_actual_kw", 0.0)))
        cool_deltas.append(float(info.get("mean_cool_delta_kw", 0.0)))
        p_pumps.append(float(info.get("mean_p_pump_mw", 0.0)))
        p_hvacs.append(float(info.get("mean_p_hvac_mw", 0.0)))
        inner_throttles.append(float(info.get("mean_inner_throttle", 0.0)))
        inner_pumps.append(float(info.get("mean_inner_pump", 0.0)))
        inner_hvacs.append(float(info.get("mean_inner_hvac", 0.0)))
        inner_besses.append(float(info.get("mean_inner_bess", 0.0)))

    if transition_logger is not None:
        transition_logger.close()

    n_steps = len(rewards)
    macro_ticks_full = env._episode_macro_ticks
    survived = 1.0 if n_steps >= macro_ticks_full else 0.0

    return {
        "mean_reward":          float(np.mean(rewards)),
        "total_reward":         float(np.sum(rewards)),
        "bid_acceptance_rate":  float(np.mean(bids_accepted)),
        "total_reg_revenue":    float(np.sum(reg_revenues)),
        "total_elec_cost":      float(np.sum(elec_costs)),
        "mean_perf_score":      float(np.mean(perf_scores)),
        "mean_committed_mw":    float(np.mean(committed_mws)),
        "tracking_rmse":        float(np.sqrt(
            np.sum(tracking_errs) / tracking_steps if env._physical else np.mean(tracking_errs)
        )) if tracking_errs else 0.0,
        "temp_A_max":           float(np.max(temp_a_maxes)) if temp_a_maxes else 0.0,
        "temp_B_max":           float(np.max(temp_b_maxes)) if temp_b_maxes else 0.0,
        "episode_length":       n_steps,
        "survived":             survived,
        "mean_flex_kw":         float(np.mean(flex_reductions)) if flex_reductions else 0.0,
        "mean_bess_kw":         float(np.mean(bess_actuals)) if bess_actuals else 0.0,
        "mean_cool_delta_kw":   float(np.mean(cool_deltas)) if cool_deltas else 0.0,
        "mean_p_pump_mw":       float(np.mean(p_pumps)) if p_pumps else 0.0,
        "mean_p_hvac_mw":       float(np.mean(p_hvacs)) if p_hvacs else 0.0,
        "mean_inner_throttle":  float(np.mean(inner_throttles)) if inner_throttles else 0.0,
        "mean_inner_pump":      float(np.mean(inner_pumps)) if inner_pumps else 0.0,
        "mean_inner_hvac":      float(np.mean(inner_hvacs)) if inner_hvacs else 0.0,
        "mean_inner_bess":      float(np.mean(inner_besses)) if inner_besses else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(
    scenarios: list[str],
    n_episodes: int,
    seed_start: int,
    macro_model_dir: Path,
    inner_model_dir: Path,
    record_transitions: bool,
    simulation_mode: str = "physical_v1",
    allow_semantic_mismatch: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if record_transitions:
        project_root = Path(__file__).resolve().parent.parent
        (project_root / "runs").mkdir(parents=True, exist_ok=True)

    for scenario in tqdm(scenarios, desc="Scenarios", position=0):
        # Load both agents once per scenario
        macro_agent = _load_sac_model(macro_model_dir, label="sac_macro",
                                     simulation_mode=simulation_mode,
                                     allow_semantic_mismatch=allow_semantic_mismatch)
        inner_agent = _load_sac_model(inner_model_dir, label="sac_low",
                                     simulation_mode=simulation_mode,
                                     allow_semantic_mismatch=allow_semantic_mismatch)

        combo_name = f"{macro_agent.algo_name}+{inner_agent.algo_name}"
        ep_metrics: list[dict[str, float]] = []
        t0 = time.perf_counter()

        for ep in tqdm(range(n_episodes), desc=f"{combo_name}/{scenario}", position=1, leave=False):
            m = run_hierarchical_episode(
                macro_agent=macro_agent,
                inner_agent=inner_agent,
                scenario=scenario,
                seed=seed_start + ep,
                episode_number=ep,
                record_transitions=record_transitions,
                simulation_mode=simulation_mode,
            )
            ep_metrics.append(m)

        elapsed = time.perf_counter() - t0

        # Aggregate across episodes
        keys = [k for k in ep_metrics[0].keys() if k != "survived"]
        agg = {k: float(np.mean([m[k] for m in ep_metrics])) for k in keys}
        std = {
            k: float(np.std([m[k] for m in ep_metrics], ddof=1)) if n_episodes > 1 else 0.0
            for k in keys
        }
        agg["survival_rate"] = float(np.mean([m["survived"] for m in ep_metrics]))

        row: dict[str, Any] = {
            "scenario":   scenario,
            "simulation_mode": simulation_mode,
            "agent":      combo_name,
            "n_episodes": n_episodes,
            "wall_time_s": round(elapsed, 2),
            **{k: round(v, 4) for k, v in agg.items()},
            **{f"{k}_std": round(v, 4) for k, v in std.items()},
        }
        rows.append(row)

        tqdm.write(
            f"  {combo_name}/{scenario}  "
            f"reward={agg['mean_reward']:7.2f}  "
            f"accept_rate={agg['bid_acceptance_rate']:.3f}  "
            f"reg_rev={agg['total_reg_revenue']:8.1f}  "
            f"survive={agg['survival_rate']:.2f}  "
            f"flex={agg['mean_flex_kw']:.0f}kW  "
            f"bess={agg['mean_bess_kw']:.0f}kW  "
            f"cool_d={agg['mean_cool_delta_kw']:.0f}kW  "
            f"inner[thr={agg['mean_inner_throttle']:.2f} "
            f"pmp={agg['mean_inner_pump']:.2f} "
            f"hvac={agg['mean_inner_hvac']:.2f} "
            f"bess={agg['mean_inner_bess']:+.2f}]"
        )

    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: len(c) for c in cols}
    str_rows = []
    for row in rows:
        sr = {}
        for c in cols:
            v = row[c]
            sr[c] = f"{v:.4f}" if isinstance(v, float) else str(v)
            widths[c] = max(widths[c], len(sr[c]))
        str_rows.append(sr)

    header = " | ".join(c.rjust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(f"\n{header}")
    print(sep)
    for sr in str_rows:
        print(" | ".join(sr[c].rjust(widths[c]) for c in cols))
    print()


def save_csv(rows: list[dict[str, Any]], path: Path, append: bool = False) -> None:
    if not rows:
        print("No results to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if append and path.exists():
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)
        print(f"\nResults appended -> {path}")
    else:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical benchmark: SAC macro + SAC low-level"
    )
    parser.add_argument(
        "--scenarios", nargs="+", type=str,
        default=["default"],
        choices=SCENARIOS,
        help="Evaluation scenarios (default: [default])",
    )
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--simulation-mode", choices=["physical_v1", "legacy_v0"], default="physical_v1")
    parser.add_argument("--allow-semantic-mismatch", action="store_true")
    parser.add_argument(
        "--macro-model-dir", type=str, default=None,
        help="Directory containing the trained SAC macro model "
             "(expects final_model.zip inside). "
             "Default: trained_models/sac_macro_<scenario>_s<seed>",
    )
    parser.add_argument(
        "--inner-model-dir", type=str, default=None,
        help="Directory containing the trained SAC low-level model "
             "(expects final_model.zip inside). "
             "Default: trained_models/sac_<scenario>_s<seed>",
    )
    parser.add_argument(
        "--output", type=str, default='c2g-bench-neurips2026/outputs/sac_default/sacmacro_rulelowlevel/sachigh_saclow/sachigh_saclow_benchmark.csv',
        help="Output CSV path. Default: evaluation/results/sac_macro+sac_low_<scenario>.csv",
    )
    parser.add_argument(
        "--record_transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log per-step transitions under runs/",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append results to existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    all_rows: list[dict[str, Any]] = []

    for scenario in args.scenarios:
        # Resolve model directories (allow per-scenario defaults)
        macro_dir = (
            Path(args.macro_model_dir)
            if args.macro_model_dir
            else project_root / "trained_models" / f"sac_macro_{scenario}_s{args.seed}"
        )
        inner_dir = (
            Path(args.inner_model_dir)
            if args.inner_model_dir
            else project_root / "trained_models" / f"sac_{scenario}_s{args.seed}"
        )

        print(f"\n{'='*60}")
        print(f"Scenario : {scenario}")
        print(f"Macro SAC: {macro_dir}")
        print(f"Inner SAC: {inner_dir}")
        print(f"Episodes : {args.n_episodes}  |  Seed start: {args.seed}")
        print(f"{'='*60}")

        rows = benchmark(
            scenarios=[scenario],
            n_episodes=args.n_episodes,
            seed_start=args.seed,
            macro_model_dir=macro_dir,
            inner_model_dir=inner_dir,
            record_transitions=args.record_transitions,
            simulation_mode=args.simulation_mode,
            allow_semantic_mismatch=args.allow_semantic_mismatch,
        )
        all_rows.extend(rows)

    print_results_table(all_rows)

    scenario_tag = args.scenarios[0] if len(args.scenarios) == 1 else "multi"
    output_path = (
        Path(args.output)
        if args.output
        else project_root / "evaluation" / "results" / f"sac_macro+sac_low_{scenario_tag}.csv"
    )
    save_csv(all_rows, output_path, append=args.append)
