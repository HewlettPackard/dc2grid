"""
evaluation/run_benchmark.py  --  Benchmark Evaluation Runner
============================================================
Runs all registered agents on all 4 evaluation scenarios, collects
per-episode metrics, and writes under evaluation/results.

Metrics (per episode)
---------------------
  mean_reward         — mean step reward
  total_reward        — sum of step rewards
  tracking_rmse       — RMSE of regulation signal tracking error (MW)
  thermal_viol_rate   — fraction of ticks with temp > T_warn (33°C)
  throughput_ratio    — mean(p_flex_kw / p_flex_nom_kw) over episode
  bess_degradation    — cumulative cycle ageing fraction * 1e4
  episode_length      — ticks in episode (< episode_ticks means thermal termination)
  survival_rate       — fraction of episodes that ran to full episode_ticks ticks

Usage
-----
  cd ../C2G-Macro
  python evaluation/run_benchmark.py                         # rule-based only
  python evaluation/run_benchmark.py --agents rule_based ppo --n_episodes 10
  python evaluation/run_benchmark.py --model_dir trained_models/ppo_default_s42

  # Hierarchical: macro agent + low-level controller (explicit combo form)
  python evaluation/run_benchmark.py --agents rule_macro+pid rule_macro+bang_bang rule_macro+rule_based

Agents
------
  rule_based   — heuristic controller from baselines/rule_based_mpc.py
  rule_macro   — macro-level heuristic from baselines/rule_based_macro.py
  bang_bang     — hysteresis on/off controller
  pid          — multi-loop PID controller
  mpc_fast     — rolling-horizon MPC (fast env, scipy SLSQP)
  mpc_macro    — long-horizon MPC (macro env, scipy SLSQP)
  milp         — MILP economic dispatch (macro env)
  ppo          — loads trained_models/ppo_<scenario>_s42/final_model
  sac          — loads trained_models/sac_<scenario>_s42/final_model
  random       — np.random uniform (lower bound)
  random_macro — np.random uniform over macro action space (bid MW + price)
"""
from __future__ import annotations
import argparse, csv, json, random as _py_random, re, time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml
from tqdm import tqdm

from c2g_env import C2GFastEnv, C2GMacroEnv
from c2g_env.experiments.identity import file_fingerprint, source_id
from c2g_env.plant_profiles import available_plant_profiles, load_plant_profile
from baselines.rule_based_mpc import RuleBasedController
from baselines.rule_based_macro import RuleBasedMacroController
from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.mpc_fast import MPCFastController
from baselines.mpc_macro import MPCMacroController
from baselines.milp_dispatch import MILPDispatchController
from baselines.metrics_callback import C2GTransitionLoggerCallback, build_ablation_suffix, STATE_COLUMNS
from baselines.train_llm_agents import (
    LLMPolicyAgent,
    load_prompt_templates,
)
from c2g_env.thermal_limits import T_SAFE, T_WARN

SCENARIOS    = ["default", "scenario_a", "scenario_b", "scenario_c"]
T_WARN_NORM  = T_WARN / T_SAFE   # normalised warning threshold
# Facility-capacity override applied to every constructed env; set from the
# --plant_profile CLI flag. None = built-in 250 MW facility.
_PLANT_OVERRIDES: dict | None = None
_PLANT_PROFILE = "none"
_RL_ALGOS = {
    "ppo",
    "sac"
}
_VALID_ACTIONS = ("throttle_batch", "pump_speed_A", "hvac_effort", "bess_dispatch")
_ACTION_BOUNDS: dict[str, tuple[float, float]] = {
    "throttle_batch": (0.0, 1.0),
    "pump_speed_A": (0.0, 1.0),
    "hvac_effort": (0.0, 1.0),
    "bess_dispatch": (-1.0, 1.0),
}


def _make_env(
    scenario: str,
    fixed_action_values: dict[str, float] | None = None,
    **kwargs,
) -> C2GFastEnv:
    """Return ActionAblationFastEnv when fixed-action overrides are active."""
    kwargs.setdefault("plant_overrides", _PLANT_OVERRIDES)
    if fixed_action_values:
        from c2g_env.experiments.action_ablation_env import ActionAblationFastEnv
        return ActionAblationFastEnv(
            scenario=scenario,
            fixed_action_values=fixed_action_values,
            **kwargs,
        )
    return C2GFastEnv(scenario=scenario, **kwargs)


def _make_macro_env(scenario: str, sub_step_callback=None, **kwargs) -> C2GMacroEnv:
    """Return a C2GMacroEnv for macro-level agent evaluation."""
    kwargs.setdefault("plant_overrides", _PLANT_OVERRIDES)
    return C2GMacroEnv(scenario=scenario, sub_step_callback=sub_step_callback, **kwargs)


def _experiment_context(env, seed: int, agent, inner_action_fn=None) -> dict[str, Any]:
    fast_env = getattr(env, "_fast_env", env)
    return {
        "run_id": uuid4().hex,
        "source_id": source_id(Path(__file__).resolve().parents[1]),
        "seed": seed,
        "scenario": fast_env._scenario,
        "plant_profile": _PLANT_PROFILE,
        "facility_capacity_mw": fast_env._facility_cap_kw / 1_000.0,
        "plant_overrides": fast_env._plant_overrides,
        "thermal_overrides": fast_env._thermal_overrides,
        "environment": {"global": fast_env._gcfg, "reward": fast_env._rcfg, "scenario": fast_env._scfg},
        "checkpoint": getattr(agent, "checkpoint", None),
        "inner_checkpoint": getattr(inner_action_fn, "checkpoint", None),
    }


def _parse_fixed_action_args(values: list[str] | None) -> dict[str, float]:
    if not values:
        return {}

    fixed_action_values: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --fixed-action '{item}'. Expected format action=value."
            )
        action_name, raw_value = item.split("=", 1)
        action_name = action_name.strip()
        if action_name not in _ACTION_BOUNDS:
            valid = ", ".join(_VALID_ACTIONS)
            raise ValueError(
                f"Invalid action '{action_name}' in --fixed-action. Valid actions: {valid}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fixed value for action '{action_name}': {raw_value}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"Invalid fixed value for action '{action_name}': {raw_value}. Value must be finite."
            )
        lo, hi = _ACTION_BOUNDS[action_name]
        if value < lo or value > hi:
            raise ValueError(
                f"Invalid fixed value for action '{action_name}': {value}. "
                f"Expected range [{lo}, {hi}]."
            )
        fixed_action_values[action_name] = value
    return fixed_action_values


_INNER_CONTROLLERS = {"random", "pid", "bang_bang", "rule_based", "mpc_fast", "ppo", "sac"}


def _make_inner_controller(
    name: str,
    env: "C2GFastEnv | None" = None,
    scenario: str = "default",
    seed: int = 42,
    model_dir: str | None = None,
):
    """Instantiate a low-level controller by name."""
    if name == "random":
        if env is None:
            raise ValueError("env must be provided to build a random inner controller")
        return RandomAgent(env)
    if name == "pid":
        return PIDController()
    if name == "bang_bang":
        return BangBangController()
    if name == "rule_based":
        _committed = float(env._committed_mw_max)
        _bess_pmax = float(getattr(env._bess, "P_MAX_MW", 5.0))
        return RuleBasedController(committed_mw_max=_committed, bess_p_max_mw=_bess_pmax)
    if name == "mpc_fast":
        return MPCFastController()
    if name == "ppo":
        return load_sb3_agent("ppo", scenario, seed, model_dir)
    if name == "sac":
        return load_sb3_agent("sac", scenario, seed, model_dir)
    raise ValueError(f"Unknown inner controller '{name}'. Choose from: {_INNER_CONTROLLERS}")


def _infer_agent_type(agent_name: str) -> str:
    """Classify benchmark agents as macro vs hardware controllers."""
    macro_agents = {"rule_macro", "random_macro", "mpc_macro", "milp", "ppo_macro", "sac_macro", "llm_policy_macro"}
    # Hierarchical combos like rule_macro+pid are also macro agents
    if "+" in agent_name:
        return "macro"
    return "macro" if agent_name in macro_agents else "hardware"


def _default_output_path(
    agents: list[str],
    scenarios: list[str],
    fixed_action_values: dict[str, float] | None = None,
    plant_profile: str = "none",
) -> Path:
    """Build a deterministic output path stem when --output is not provided.

    For single-type runs returns e.g. evaluation/results/rule_based_default_hardware_base.csv.
    For combo runs the stem omits the agent_type suffix — save_results will append _macro
    and _hardware itself.
    """
    unique_agents = list(dict.fromkeys(agents))
    unique_scenarios = list(dict.fromkeys(scenarios))

    algo_tag = unique_agents[0] if len(unique_agents) == 1 else "multi"
    scenario_tag = unique_scenarios[0] if len(unique_scenarios) == 1 else "multi"

    has_combo = any("+" in a for a in unique_agents)
    agent_types = {_infer_agent_type(name) for name in unique_agents}
    if has_combo:
        # Combo runs: no agent_type suffix — save_results appends _macro/_hardware
        agent_type_tag = ""
    elif not agent_types:
        agent_type_tag = "_unknown"
    elif len(agent_types) == 1:
        agent_type_tag = f"_{next(iter(agent_types))}"
    else:
        agent_type_tag = "_mixed"

    ablation_suffix = build_ablation_suffix(fixed_action_values)
    ablation_tag = ablation_suffix.lstrip("_") if ablation_suffix else "base"

    capacity_suffix = "" if plant_profile == "none" else f"_{plant_profile}"
    return Path("evaluation") / "results" / f"{algo_tag}_{scenario_tag}{agent_type_tag}_{ablation_tag}{capacity_suffix}.csv"


# ---------------------------------------------------------------------------
# Agent wrappers — expose a common predict(obs) -> (action, state) interface
# ---------------------------------------------------------------------------

class RandomAgent:
    """Samples uniformly from the action space."""
    def __init__(self, env: C2GFastEnv, algo_name: str = "random"):
        self._space = env.action_space
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._space.sample(), None


class MacroRandomAgent:
    """Samples uniformly from the macro (2-D) action space."""
    def __init__(self, env: C2GMacroEnv, algo_name: str = "random_macro"):
        self._space = env.action_space
        self.algo_name = algo_name

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._space.sample(), None


class SB3Agent:
    """Wraps a loaded SB3 model (PPO or SAC)."""
    def __init__(self, model, algo_name: str, obs_normalizer=None, checkpoint=None):
        self._model = model
        self.algo_name = algo_name
        self._obs_normalizer = obs_normalizer
        self.checkpoint = checkpoint

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self._obs_normalizer is not None:
            obs = self._obs_normalizer.normalize_obs(np.asarray(obs, dtype=np.float32))
        return self._model.predict(obs, deterministic=deterministic)


def _resolve_sb3_spec(algo: str):
    from stable_baselines3 import PPO, SAC

    algo_key = algo.lower()
    if algo_key in ("sac", "sac_macro"):
        return SAC, algo_key, False
    if algo_key in ("ppo_macro",):
        return PPO, "ppo_macro", True
    if algo_key in _RL_ALGOS:
        return PPO, algo_key, True
    raise ValueError(f"Unknown algo '{algo}'. Use ppo, sac, ppo_macro, or sac_macro.")


def _maybe_load_obs_normalizer(stats_path: Path, scenario: str):
    if not stats_path.exists():
        return None

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_env = DummyVecEnv([lambda: C2GFastEnv(scenario=scenario)])
    vec_norm = VecNormalize.load(str(stats_path), vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return vec_norm


def load_sb3_agent(algo: str, scenario: str, seed: int, model_dir: str | None):
    cls, train_key, should_restore_norm = _resolve_sb3_spec(algo)
    if model_dir:
        path = Path(model_dir) / "final_model"
    else:
        path = Path("trained_models") / f"{train_key}_{scenario}_s{seed}" / "final_model"
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"No trained model at {path}.zip — run baselines/train_{algo}.py first."
        )
    model = cls.load(str(path))
    obs_normalizer = None
    if should_restore_norm:
        obs_normalizer = _maybe_load_obs_normalizer(path.parent / "vec_normalize.pkl", scenario)
    return SB3Agent(model, algo_name=algo.lower(), obs_normalizer=obs_normalizer,
                    checkpoint=file_fingerprint(path.with_suffix(".zip")))


# Metric collection
# ---------------------------------------------------------------------------

def run_episode(
    agent,
    scenario: str,
    seed: int,
    algo_name: str | None = None,
    agent_type: str = "hardware",
    episode_number: int = 0,
    record_transitions: bool = True,
    fixed_action_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run one episode and return a metrics dict."""
    env = _make_env(
        scenario=scenario,
        fixed_action_values=fixed_action_values,
    )
    obs, _ = env.reset(seed=seed)
    algo_for_logging = getattr(agent, "algo_name", (algo_name or "unknown"))

    transition_logger = None
    if record_transitions:
        transition_logger = C2GTransitionLoggerCallback(
            output_dir="runs",
            algorithm_name=algo_for_logging,
            scenario_name=scenario,
            agent_type=agent_type,
            episode_number=episode_number,
            fixed_action_values=fixed_action_values,
            verbose=0,
            experiment_context=_experiment_context(env, seed, agent),
        )

    rewards       : list[float] = []
    tracking_errs : list[float] = []
    thermal_viols : int = 0
    throughputs   : list[float] = []
    bess_init_age : float | None = None
    # Cumulative power metrics
    cumul_p_pump_mw     : float = 0.0
    cumul_p_hvac_mw     : float = 0.0
    cumul_flex_reduction_kw : float = 0.0
    cumul_bess_actual_kw : float = 0.0

    done = False
    try:
      while not done:
        state = obs.copy()
        if getattr(agent, "uses_env_context", False):
            action, _ = agent.predict(obs, deterministic=True, env=env, scenario=scenario)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info[k] for k in (
                    "reward_throughput", "reward_tracking", "reward_thermal",
                    "reward_soc", "reward_freq", "reward_volt", "reward_backlog",
                )
            }

            print(f"[Transition] Step reward: {reward:.3f}, components: {reward_components}", flush=True)
            transition_logger.record_transition(
                state=state,
                action=action,
                observation=obs,
                reward=float(reward),
                done=done,
                reward_components=reward_components,
            )

        rewards.append(float(reward))

        # Push reward to ICRL buffer if the agent supports it
        if hasattr(agent, "push_reward"):
            agent.push_reward(float(reward))

        # Thermal violations
        if obs[0] >= T_WARN_NORM or obs[1] >= T_WARN_NORM:
            thermal_viols += 1

        # Tracking error: reuse the env's own computation (against prev regd signal)
        tracking_errs.append(info["tracking_err_kw"] ** 2)

        # Throughput ratio: flex_kw / flex_nom_kw
        tp = info.get("throughput_ratio", None)
        if tp is not None:
            throughputs.append(float(tp))

        # Accumulate power metrics
        cumul_p_pump_mw += float(info.get("p_pump_mw", 0.0))
        cumul_p_hvac_mw += float(info.get("p_hvac_mw", 0.0))
        cumul_flex_reduction_kw += float(info.get("flex_reduction_kw", 0.0))
        cumul_bess_actual_kw += float(info.get("bess_actual_kw", 0.0))

        # BESS degradation
        if bess_init_age is None:
            bess_init_age = float(info.get("bess_age_frac", 0.0))

    finally:
        if transition_logger is not None:
            transition_logger.close()

    bess_final_age = float(info.get("bess_age_frac", bess_init_age or 0.0))
    bess_degradation = (bess_final_age - (bess_init_age or 0.0)) * 1e4

    n_ticks    = len(rewards)
    survived   = 1.0 if n_ticks >= env._episode_ticks else 0.0

    return {
        "mean_reward"       : float(np.mean(rewards)),
        "total_reward"      : float(np.sum(rewards)),
        "tracking_rmse"     : float(np.sqrt(np.mean(tracking_errs))) if tracking_errs else 0.0,
        "thermal_viol_rate" : thermal_viols / max(n_ticks, 1),
        "throughput_ratio"  : float(np.mean(throughputs)) if throughputs else 0.0,
        "bess_degradation"  : bess_degradation,
        "episode_length"    : n_ticks,
        "survived"          : survived,
        "cumul_p_pump_mw"   : cumul_p_pump_mw,
        "cumul_p_hvac_mw"   : cumul_p_hvac_mw,
        "cumul_flex_reduction_kw" : cumul_flex_reduction_kw,
        "cumul_bess_actual_kw" : cumul_bess_actual_kw,
    }


def run_macro_episode(
    agent,
    scenario: str,
    seed: int,
    algo_name: str | None = None,
    agent_type: str = "macro",
    episode_number: int = 0,
    inner_action_fn: Any = None,
    record_transitions: bool = True,
    inner_algo_name: str | None = None,
    combo_name: str | None = None,
    fixed_action_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run one macro-level episode and return metrics dict."""
    # sub_step_callback fires after every _fast_env.step() inside C2GMacroEnv
    # (180 calls per macro tick).  We always collect hardware metrics for combo
    # agents; the transition logger is only created when record_transitions=True.
    _HW_RC_KEYS = (
        "reward_throughput", "reward_tracking", "reward_thermal",
        "reward_soc", "reward_freq", "reward_volt", "reward_backlog",
    )
    env = _make_macro_env(
        scenario=scenario,
        inner_action_fn=inner_action_fn,
        fixed_action_values=fixed_action_values or None,
    )
    obs, _ = env.reset(seed=seed)
    experiment_context = _experiment_context(env, seed, agent, inner_action_fn) if record_transitions else None
    hw_step_metrics: list[dict] = []
    inner_transition_logger = None
    sub_step_callback = None
    if inner_action_fn is not None:
        if record_transitions:
            inner_transition_logger = C2GTransitionLoggerCallback(
                output_dir="runs",
                algorithm_name=combo_name or inner_algo_name or "inner",
                scenario_name=scenario,
                agent_type="hardware",
                episode_number=episode_number,
                fixed_action_values=None,
                verbose=0,
                experiment_context=experiment_context,
            )
        _lg_ref = inner_transition_logger  # capture before closure
        def sub_step_callback(
            pre_obs, low_action, next_obs, rew, done, info,
            _hw=hw_step_metrics, _lg=_lg_ref, _keys=_HW_RC_KEYS,
        ):
            _hw.append({
                "reward":            float(rew),
                "tracking_err_sq":   float(info.get("tracking_err_kw", 0.0)) ** 2,
                "thermal_viol":      int(next_obs[0] >= T_WARN_NORM or next_obs[1] >= T_WARN_NORM),
                "throughput_ratio":  float(info.get("throughput_ratio", 0.0)),
                "bess_age_frac":     float(info.get("bess_age_frac", 0.0)),
                "p_pump_mw":         float(info.get("p_pump_mw", 0.0)),
                "p_hvac_mw":         float(info.get("p_hvac_mw", 0.0)),
                "flex_reduction_kw": float(info.get("flex_reduction_kw", 0.0)),
                "bess_actual_kw":    float(info.get("bess_actual_kw", 0.0)),
            })
            if _lg is not None:
                _lg.record_transition(
                    state=pre_obs,
                    action=low_action,
                    observation=next_obs,
                    reward=float(rew),
                    done=bool(done),
                    reward_components={k: info.get(k, 0.0) for k in _keys},
                )
            # Push sub-step reward to inner agent's ICRL buffer if applicable
            if inner_action_fn is not None and hasattr(inner_action_fn, "push_reward"):
                inner_action_fn.push_reward(float(rew))

    env._sub_step_callback = sub_step_callback
    algo_for_logging = combo_name or getattr(agent, "algo_name", (algo_name or "unknown"))

    # Extract static env attributes once for LLM agents that need them
    # (kept for non-LLM agents that use static_env_info; LLM agents receive env directly)
    macro_env_info: dict[str, Any] = {
        "committed_mw_max": float(getattr(env, "_committed_max_mw", 30.0)),
        "dr_baseline_mw":   float(getattr(env, "_dr_baseline_mw", 5.0)),
        "bess_p_max_mw":    float(getattr(env._fast_env._bess, "P_MAX_MW", 5.0)),
    } if getattr(agent, "uses_env_context", False) else {}

    transition_logger = None
    if record_transitions:
        transition_logger = C2GTransitionLoggerCallback(
            output_dir="runs",
            algorithm_name=algo_for_logging,
            scenario_name=scenario,
            agent_type=agent_type,
            episode_number=episode_number,
            fixed_action_values=None,
            verbose=0,
            experiment_context=experiment_context,
        )
    rewards: list[float] = []
    bids_accepted: list[float] = []
    reg_revenues: list[float] = []
    elec_costs: list[float] = []
    perf_scores: list[float] = []
    committed_mws: list[float] = []
    tracking_errs: list[float] = []
    temp_a_maxes: list[float] = []
    temp_b_maxes: list[float] = []
    # Per-lever tracking contributions (mean across sub-steps, per macro tick)
    flex_reductions: list[float] = []
    bess_actuals: list[float] = []
    cool_deltas: list[float] = []
    p_pumps: list[float] = []
    p_hvacs: list[float] = []

    done = False
    try:
      while not done:
        state = obs.copy()
        if getattr(agent, "uses_env_context", False):
            action, _ = agent.predict(obs, deterministic=True, env=env, scenario=scenario)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if transition_logger is not None:
            reward_components = {
                k: info.get(k, 0.0) for k in (
                    "reward_regulation","reward_sub", "reward_elec",  "reward_churn"
                )
            }
            print(f"[Transition] Step reward: {reward:.3f}, components: {reward_components}", flush=True)
            transition_logger.record_transition(
                state=state,
                action=action,
                observation=obs,
                reward=float(reward),
                done=done,
                reward_components=reward_components,
            )

        rewards.append(float(reward))

        # Push reward to ICRL buffer if the macro agent supports it
        if hasattr(agent, "push_reward"):
            agent.push_reward(float(reward), info=info)

        bids_accepted.append(float(info.get("bid_accepted", False)))
        reg_revenues.append(float(info.get("regulation_revenue", 0.0)))
        elec_costs.append(float(info.get("electricity_cost", 0.0)))
        perf_scores.append(float(info.get("perf_score", 0.0)))
        committed_mws.append(float(info.get("committed_mw", 0.0)))
        tracking_errs.append(float(info.get("mean_tracking_err", 0.0)) ** 2)
        temp_a_maxes.append(float(info.get("temp_A_max", 0.0)))
        temp_b_maxes.append(float(info.get("temp_B_max", 0.0)))
        # Per-lever tracking contributions (mean over 180 sub-steps)
        flex_reductions.append(float(info.get("mean_flex_reduction_kw", 0.0)))
        bess_actuals.append(float(info.get("mean_bess_actual_kw", 0.0)))
        cool_deltas.append(float(info.get("mean_cool_delta_kw", 0.0)))
        p_pumps.append(float(info.get("mean_p_pump_mw", 0.0)))
        p_hvacs.append(float(info.get("mean_p_hvac_mw", 0.0)))
    finally:
      if transition_logger is not None:
          transition_logger.close()
      if inner_transition_logger is not None:
          inner_transition_logger.close()

    n_steps = len(rewards)
    macro_ticks_full = env._episode_macro_ticks
    survived = 1.0 if n_steps >= macro_ticks_full else 0.0

    # Hardware-level aggregate metrics for inner controller (empty if no inner_action_fn)
    inner_agg: dict[str, float] = {}
    if hw_step_metrics:
        n_hw = len(hw_step_metrics)
        bess_ages = [m["bess_age_frac"] for m in hw_step_metrics]
        inner_agg = {
            "inner_mean_reward":           float(np.mean([m["reward"] for m in hw_step_metrics])),
            "inner_total_reward":          float(np.sum([m["reward"] for m in hw_step_metrics])),
            "inner_tracking_rmse":         float(np.sqrt(np.mean([m["tracking_err_sq"] for m in hw_step_metrics]))),
            "inner_thermal_viol_rate":     float(np.mean([m["thermal_viol"] for m in hw_step_metrics])),
            "inner_throughput_ratio":      float(np.mean([m["throughput_ratio"] for m in hw_step_metrics])),
            "inner_bess_degradation":      (bess_ages[-1] - bess_ages[0]) * 1e4,
            "inner_episode_length":        float(n_hw),
            "inner_survived":              survived,  # same episode, same termination
            "inner_cumul_p_pump_mw":       float(np.sum([m["p_pump_mw"] for m in hw_step_metrics])),
            "inner_cumul_p_hvac_mw":       float(np.sum([m["p_hvac_mw"] for m in hw_step_metrics])),
            "inner_cumul_flex_reduction_kw": float(np.sum([m["flex_reduction_kw"] for m in hw_step_metrics])),
            "inner_cumul_bess_actual_kw":  float(np.sum([m["bess_actual_kw"] for m in hw_step_metrics])),
        }

    return {
        "mean_reward":          float(np.mean(rewards)),
        "total_reward":         float(np.sum(rewards)),
        "bid_acceptance_rate":  float(np.mean(bids_accepted)),
        "total_reg_revenue":    float(np.sum(reg_revenues)),
        "total_elec_cost":      float(np.sum(elec_costs)),
        "mean_perf_score":      float(np.mean(perf_scores)),
        "mean_committed_mw":    float(np.mean(committed_mws)),
        "tracking_rmse":        float(np.sqrt(np.mean(tracking_errs))) if tracking_errs else 0.0,
        "temp_A_max":           float(np.max(temp_a_maxes)) if temp_a_maxes else 0.0,
        "temp_B_max":           float(np.max(temp_b_maxes)) if temp_b_maxes else 0.0,
        "episode_length":       n_steps,
        "survived":             survived,
        "mean_flex_kw":         float(np.mean(flex_reductions)) if flex_reductions else 0.0,
        "mean_bess_kw":         float(np.mean(bess_actuals)) if bess_actuals else 0.0,
        "mean_cool_delta_kw":   float(np.mean(cool_deltas)) if cool_deltas else 0.0,
        "mean_p_pump_mw":       float(np.mean(p_pumps)) if p_pumps else 0.0,
        "mean_p_hvac_mw":       float(np.mean(p_hvacs)) if p_hvacs else 0.0,
        **inner_agg,
    }


def benchmark(
    agents      : list[str],
    scenarios   : list[str],
    n_episodes  : int,
    seed_start  : int,
    model_dir   : str | None,
    macro_model_dir: str | None = None,
    hw_model_dir: str | None = None,
    record_transitions: bool = False,
    fixed_action_values: dict[str, float] | None = None,
    llm_api_base: str = "http://localhost:8000/v1",
    llm_mode: str = "hardware",
    llm_template_path: str = "conf/chat_templates/run_benchmark_rbc+ICRL.yaml",
    llm_max_new_tokens: int = 9216,
    llm_temperature: float = 0.0,
    llm_enable_thinking: bool = False,
    llm_context_num_steps: int = 10,
    llm_context_stride: int = 1,
    llm_icrl_mode: str = "autonomous",
    llm_skip_episodes: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Seed all global RNGs for cross-run reproducibility.
    # PyTorch BLAS (multi-threaded CPU mat-mul) is non-deterministic unless
    # torch.manual_seed() is called before any model inference.
    torch.manual_seed(seed_start)
    _py_random.seed(seed_start)
    np.random.seed(seed_start)

    if llm_mode not in ["hardware", "macro"]:
        raise ValueError(f"Invalid agent mode '{llm_mode}'. Must be 'hardware' or 'macro'.")

    if record_transitions:
        project_root = Path(__file__).resolve().parent.parent
        (project_root / "runs").mkdir(parents=True, exist_ok=True)

    _needs_llm = any(
        "llm_policy" in a for a in agents
    )
    try:
        prompt_templates = load_prompt_templates(llm_template_path)
    except FileNotFoundError:
        prompt_templates = {}

    if _needs_llm and not prompt_templates: raise FileNotFoundError(f"LLM agent specified but no prompt templates found at '{llm_template_path}'.")

    _icrl = prompt_templates.get("icrl", {})
    # ICRL is active only when the template file actually defines attempt templates.
    # When inactive, context_num_steps and icrl_mode are irrelevant.
    _hw_icrl_active    = bool(_icrl.get("hardware_attempt"))
    _macro_icrl_active = bool(_icrl.get("macro_attempt"))

    scenario_bar = tqdm(scenarios, desc="Scenarios", position=0)
    for scenario in scenario_bar:
        scenario_bar.set_description(f"Scenario: {scenario}")

        agent_bar = tqdm(agents, desc="Agents", position=1, leave=False)
        for agent_name in agent_bar:
            agent_bar.set_description(f"Agent: {agent_name}")
            agent_type = llm_mode if agent_name == "llm_policy" else _infer_agent_type(agent_name)

            # Instantiate agent once per (agent, scenario) to share weights
            if agent_type == "macro":
                env_for_space = _make_macro_env(scenario=scenario)
            else:
                env_for_space = _make_env(
                    scenario=scenario,
                    fixed_action_values=fixed_action_values,
                )
            env_for_space.reset(seed=0)

            # ── Hierarchical combo agents (e.g. rule_macro+pid) ────
            inner_action_fn = None
            macro_part = agent_name
            if "+" in agent_name:
                macro_part, inner_part = agent_name.split("+", 1)
                inner_env = _make_env(scenario=scenario)
                inner_env.reset(seed=0)
                if inner_part == "llm_policy":
                    state_names = [name.removeprefix("s_") for name in STATE_COLUMNS]
                    _inner_agent = LLMPolicyAgent(
                        mode="hardware",
                        prompts=prompt_templates,
                        state_names=state_names,
                        max_new_tokens=llm_max_new_tokens,
                        temperature=llm_temperature,
                        api_base=llm_api_base,
                        enable_thinking=llm_enable_thinking,
                        context_num_steps=llm_context_num_steps if _hw_icrl_active else 0,
                        context_stride=llm_context_stride,
                        icrl_mode=llm_icrl_mode if _hw_icrl_active else "autonomous",
                    )
                    inner_action_fn = lambda obs, _act, c=_inner_agent, e=inner_env, sc=scenario: \
                        c.predict(obs, env=e, scenario=sc)[0]
                    if hasattr(_inner_agent, "push_reward"):
                        inner_action_fn.push_reward = _inner_agent.push_reward
                else:
                    _inner_agent = _make_inner_controller(
                        inner_part, env=inner_env,
                        scenario=scenario, seed=seed_start,
                        model_dir=hw_model_dir or model_dir,
                    )
                    inner_action_fn = lambda obs, _act, c=_inner_agent: c.predict(obs)[0]
                    if hasattr(_inner_agent, "push_reward"):
                        inner_action_fn.push_reward = _inner_agent.push_reward

                inner_action_fn.checkpoint = getattr(_inner_agent, "checkpoint", None)

            if agent_name == "rule_based":
                agent = _make_inner_controller("rule_based", env=env_for_space)
            elif agent_name in ("llm_policy", "llm_policy_macro") or macro_part in ("llm_policy", "llm_policy_macro"):
                state_names = [name.removeprefix("s_") for name in STATE_COLUMNS]
                # llm_policy_macro is always macro; llm_policy respects --llm-mode
                if agent_name == "llm_policy_macro" or macro_part == "llm_policy_macro":
                    _mode = "macro"
                else:
                    _mode = llm_mode if agent_name == "llm_policy" else "macro"
                _icrl_active  = _hw_icrl_active if _mode == "hardware" else _macro_icrl_active
                agent = LLMPolicyAgent(
                    mode=_mode,
                    prompts=prompt_templates,
                    state_names=state_names,
                    max_new_tokens=llm_max_new_tokens,
                    temperature=llm_temperature,
                    api_base=llm_api_base,
                    enable_thinking=llm_enable_thinking,
                    context_num_steps=llm_context_num_steps if _icrl_active else 0,
                    context_stride=llm_context_stride,
                    icrl_mode=llm_icrl_mode if _icrl_active else "autonomous",
                )
            elif macro_part == "rule_macro":
                agent = RuleBasedMacroController()
            elif macro_part == "random_macro":
                agent = MacroRandomAgent(env_for_space, algo_name=agent_name)
            elif agent_name == "bang_bang":
                agent = BangBangController()
            elif agent_name == "pid":
                agent = PIDController()
            elif agent_name == "mpc_fast":
                agent = MPCFastController()
            elif macro_part == "mpc_macro":
                agent = MPCMacroController()
            elif macro_part == "milp":
                agent = MILPDispatchController()
            elif agent_name == "random":
                agent = RandomAgent(env_for_space, algo_name=agent_name)
            elif macro_part in ("sac_macro", "ppo_macro"):
                try:
                    agent = load_sb3_agent(macro_part, scenario, seed_start, macro_model_dir or model_dir)
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue
            else:
                try:
                    agent = load_sb3_agent(agent_name, scenario, seed_start, hw_model_dir or model_dir)
                except FileNotFoundError as exc:
                    print(f"    SKIP: {exc}")
                    continue

            if not hasattr(agent, "algo_name"):
                setattr(agent, "algo_name", agent_name)

            ep_metrics: list[dict[str, float]] = []
            t0 = time.perf_counter()
            for ep in tqdm(range(n_episodes), desc="Episodes", position=2, leave=False):
                if agent_type == "macro":
                    m = run_macro_episode(
                        agent=agent,
                        scenario=scenario,
                        seed=seed_start + ep,
                        agent_type=agent_type,
                        episode_number=ep,
                        inner_action_fn=inner_action_fn,
                        record_transitions=record_transitions,
                        inner_algo_name=inner_part if "+" in agent_name else None,
                        combo_name=agent_name if "+" in agent_name else None,
                        fixed_action_values=fixed_action_values or None,
                    )
                else:
                    m = run_episode(
                        agent=agent,
                        scenario=scenario,
                        seed=seed_start + ep,
                        agent_type=agent_type,
                        episode_number=ep,
                        record_transitions=record_transitions,
                        fixed_action_values=fixed_action_values,
                    )
                ep_metrics.append(m)

            elapsed = time.perf_counter() - t0

            # Drop warmup episodes from reported metrics (buffer already filled)
            eval_metrics = ep_metrics[llm_skip_episodes:] if llm_skip_episodes else ep_metrics
            if not eval_metrics:
                eval_metrics = ep_metrics  # guard: skip > n_episodes, report all

            # Aggregate across episodes (mean + std)
            keys = list(eval_metrics[0].keys())
            agg  = {k: float(np.mean([m[k] for m in eval_metrics])) for k in keys}
            std  = {k: float(np.std([m[k] for m in eval_metrics], ddof=1)) if len(eval_metrics) > 1 else 0.0 for k in keys}
            agg["survival_rate"] = float(np.mean([m["survived"] for m in eval_metrics]))

            row = {
                "scenario"          : scenario,
                "agent"             : agent_name,
                "facility_capacity_mw": getattr(env_for_space, "_fast_env", env_for_space)._facility_cap_kw / 1_000.0,
                "n_episodes"        : n_episodes,
                "wall_time_s"       : round(elapsed, 2),
                **{k: round(v, 4) for k, v in agg.items() if k != "survived"},
                **{f"{k}_std": round(v, 4) for k, v in std.items() if k != "survived"},
            }
            rows.append(row)

            # Emit a separate hardware-schema row for the inner controller so
            # it can be compared directly with standalone hardware agent results.
            if "+" in agent_name and any(k.startswith("inner_") for k in agg):
                inner_part_name = agent_name.split("+", 1)[1]
                inner_row = {
                    "scenario"              : scenario,
                    "agent"                 : inner_part_name,
                    "facility_capacity_mw"  : row["facility_capacity_mw"],
                    "n_episodes"            : n_episodes,
                    "wall_time_s"           : round(elapsed, 2),
                    "mean_reward"           : round(agg.get("inner_mean_reward", 0.0), 4),
                    "total_reward"          : round(agg.get("inner_total_reward", 0.0), 4),
                    "tracking_rmse"         : round(agg.get("inner_tracking_rmse", 0.0), 4),
                    "thermal_viol_rate"     : round(agg.get("inner_thermal_viol_rate", 0.0), 4),
                    "throughput_ratio"      : round(agg.get("inner_throughput_ratio", 0.0), 4),
                    "bess_degradation"      : round(agg.get("inner_bess_degradation", 0.0), 4),
                    "episode_length"        : round(agg.get("inner_episode_length", 0.0), 4),
                    "survival_rate"         : round(agg.get("survival_rate", 0.0), 4),
                    "cumul_p_pump_mw"       : round(agg.get("inner_cumul_p_pump_mw", 0.0), 4),
                    "cumul_p_hvac_mw"       : round(agg.get("inner_cumul_p_hvac_mw", 0.0), 4),
                    "cumul_flex_reduction_kw": round(agg.get("inner_cumul_flex_reduction_kw", 0.0), 4),
                    "cumul_bess_actual_kw"  : round(agg.get("inner_cumul_bess_actual_kw", 0.0), 4),
                    # std columns (tracking_rmse_std etc.) if available
                    **{f"{k[len('inner_'):]}__std": round(std.get(k, 0.0), 4)
                       for k in std if k.startswith("inner_") and k != "inner_survived"},
                }
                rows.append(inner_row)
            if agent_type == "macro":
                tqdm.write(
                    f"  {agent_name}/{scenario}  "
                    f"reward={agg['mean_reward']:7.2f}  "
                    f"accept_rate={agg['bid_acceptance_rate']:.3f}  "
                    f"reg_rev={agg['total_reg_revenue']:8.1f}  "
                    f"survive={agg['survival_rate']:.2f}  "
                    f"flex={agg['mean_flex_kw']:.0f}kW  "
                    f"bess={agg['mean_bess_kw']:.0f}kW  "
                    f"cool_d={agg['mean_cool_delta_kw']:.0f}kW"
                )
            else:
                tqdm.write(
                    f"  {agent_name}/{scenario}  "
                    f"reward={agg['mean_reward']:7.2f}  "
                    f"tracking_rmse={agg['tracking_rmse']:8.0f}kW  "
                    f"thermal_viol={agg['thermal_viol_rate']:.3f}  "
                    f"survive={agg['survival_rate']:.2f}"
                )

    return rows


def print_results_table(rows: list[dict[str, Any]]) -> None:
    """Print results as a formatted table."""
    if not rows:
        return
    # Collect union of all keys to handle mixed hardware/macro schemas
    seen: dict[str, None] = {}
    for row in rows:
        seen.update(dict.fromkeys(row.keys()))
    cols = list(seen.keys())
    # Compute column widths
    widths = {c: len(c) for c in cols}
    str_rows = []
    for row in rows:
        sr = {}
        for c in cols:
            v = row.get(c, "")
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


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        print("No results to save.")
        return
    # Collect union of all keys to handle mixed hardware/macro schemas
    seen: dict[str, None] = {}
    for row in rows:
        seen.update(dict.fromkeys(row.keys()))
    fieldnames = list(seen.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved -> {path}")


# Hardware-schema keys present only in hardware/inner rows (not macro rows)
_HW_SCHEMA_KEYS = {"thermal_viol_rate", "throughput_ratio", "bess_degradation",
                   "cumul_p_pump_mw", "cumul_p_hvac_mw",
                   "cumul_flex_reduction_kw", "cumul_bess_actual_kw"}


def save_results(
    rows: list[dict[str, Any]],
    path: Path,
    agents: list[str],
) -> None:
    """Save results CSV(s).

    For combo-agent runs (any agent contains '+'), splits rows into two files:
      <stem>_macro.csv   — macro-level rows (agent == macro_part)
      <stem>_hardware.csv — hardware/inner rows
    and names them ``<combo>_<scenario>_macro.csv`` etc., matching the runs/
    folder convention.  For pure single-type runs, writes one file.
    """
    has_combo = any("+" in a for a in agents)
    if not has_combo:
        save_csv(rows, path)
        return

    # Split: macro rows lack hardware-schema keys; inner rows have them
    macro_rows    = [r for r in rows if not any(k in r for k in _HW_SCHEMA_KEYS)]
    hardware_rows = [r for r in rows if     any(k in r for k in _HW_SCHEMA_KEYS)]

    stem = path.with_suffix("")  # drop .csv
    # If _default_output_path already appended _macro/_hardware (custom --output),
    # respect that; otherwise append suffixes.
    stem_str = str(stem)
    if stem_str.endswith("_macro") or stem_str.endswith("_hardware"):
        save_csv(rows, path)
        return

    if macro_rows:
        save_csv(macro_rows, Path(f"{stem_str}_macro.csv"))
    if hardware_rows:
        save_csv(hardware_rows, Path(f"{stem_str}_hardware.csv"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2G-Bench evaluation runner")
    parser.add_argument(
        "--agents", nargs="+",
        default=["rule_based", "bang_bang", "pid", "random"],
        help="Agents to evaluate: rule_based rule_macro random_macro bang_bang pid mpc_fast "
               "mpc_macro milp ppo sac random llm_policy llm_policy_macro",
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        default=SCENARIOS,
        choices=SCENARIOS,
    )
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=100)
    parser.add_argument(
        "--plant_profile",
        default="none",
        choices=available_plant_profiles() + ["all"],
        help="Facility capacity profile from conf/plant_profiles/ "
             "(e.g. plant_50mw, plant_100mw, plant_500mw). Default 'none' = 250 MW. "
             "Use 'all' to sweep every profile and tag each row with plant_profile.",
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Override model directory for SB3 agents (fallback for both macro and hardware).",
    )
    parser.add_argument(
        "--macro-model-dir", dest="macro_model_dir", default=None,
        help="Model directory for the macro-level SB3 agent (overrides --model_dir for macro).",
    )
    parser.add_argument(
        "--hw-model-dir", dest="hw_model_dir", default=None,
        help="Model directory for the hardware/inner SB3 agent (overrides --model_dir for hardware).",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Path to write the results CSV. If omitted, defaults to "
            "evaluation/results/algoname_scenario_agenttype_ablation.csv"
        ),
    )
    parser.add_argument(
        "--record_transitions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable per-step transition logging under runs/<algo>_<scenario>_<agent>/transitions_<episode>.csv",
    )
    parser.add_argument(
        "--fixed-action",
        action="append",
        default=[],
        help="Optional fixed value for a disabled action, e.g. --fixed-action hvac_effort=0.8",
    )
    parser.add_argument(
        "--llm-api-base",
        default="http://localhost:8000/v1",
        help="vLLM server base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["hardware", "macro"],
        default="hardware",
        help="Control mode for llm_policy agent",
    )
    parser.add_argument(
        "--llm-template-path",
        default="conf/chat_templates/run_benchmark_rbc+ICRL.yaml",
        help="YAML path with hardware_prompt and macro_prompt templates",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=8192,
        help="Max new tokens for llm_policy generation. vLLM has no per-request thinking-budget "
             "parameter, so this is the only knob: the model fills <think> first, then emits JSON. "
             "Default 2048 suits --llm-no-thinking (JSON ~80 tok + headroom for any preamble). "
             "Use 8192+ when thinking is enabled. "
             "Must be less than the server's --max-model-len minus the prompt length.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for llm_policy generation (0 = greedy)",
    )
    parser.add_argument(
        "--llm-no-thinking",
        dest="llm_enable_thinking",
        action="store_false",
        default=False,
        help="Disable <think> reasoning for all LLM agents (faster, lower token cost).",
    )
    parser.add_argument(
        "--llm-context-num-steps",
        dest="llm_context_num_steps",
        type=int,
        default=10,
        help="Number of past steps to keep in the ICRL buffer (0 = disabled).",
    )
    parser.add_argument(
        "--llm-context-stride",
        dest="llm_context_stride",
        type=int,
        default=1,
        help="Store every K-th env step in the ICRL buffer (default 1 = store all steps).",
    )
    parser.add_argument(
        "--llm-skip-episodes",
        dest="llm_skip_episodes",
        type=int,
        default=0,
        help="Run this many buffer-warmup episodes before evaluation (not counted in results).",
    )
    parser.add_argument(
        "--llm-icrl-mode",
        dest="llm_icrl_mode",
        choices=["autonomous", "preset", "exploit"],
        default="autonomous",
        help=(
            "ICRL instruction mode: "
            "'autonomous' = model chooses explore/exploit each step; "
            "'preset' = alternates explore (even steps) / exploit (odd steps); "
            "'exploit' = always use exploitation instruction."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    evaluation_results_dir = project_root / "evaluation" / "results"
    if not (evaluation_results_dir.exists() and evaluation_results_dir.is_dir()):
        evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    # Apply the capacity profile globally so every constructed env re-instantiates
    # at the requested nameplate (e.g. --plant_profile plant_500mw).
    plant_profiles = (
        available_plant_profiles()
        if args.plant_profile == "all"
        else [args.plant_profile]
    )

    try:
        fixed_action_values = _parse_fixed_action_args(args.fixed_action)
    except ValueError as exc:
        parser.error(str(exc))

    # Expand --inner-agents: replace each standalone macro agent with its combos.
    # e.g. --agents rule_macro --inner-agents pid llm_policy
    #   → agents = [rule_macro+pid, rule_macro+llm_policy]
    # To also keep the standalone macro agent, pass it explicitly twice or use
    # the explicit combo form: --agents rule_macro rule_macro+pid
    agents = list(args.agents)

    rows = []
    for plant_profile in plant_profiles:
        _PLANT_OVERRIDES = load_plant_profile(plant_profile)
        _PLANT_PROFILE = plant_profile
        profile_rows = benchmark(
            agents     = agents,
            scenarios  = args.scenarios,
            n_episodes = args.n_episodes,
            seed_start = args.seed,
            model_dir  = args.model_dir,
            macro_model_dir = args.macro_model_dir,
            hw_model_dir    = args.hw_model_dir,
            record_transitions = args.record_transitions,
            fixed_action_values = fixed_action_values,
            llm_api_base = args.llm_api_base,
            llm_mode = args.llm_mode,
            llm_template_path = args.llm_template_path,
            llm_max_new_tokens = args.llm_max_new_tokens,
            llm_temperature = args.llm_temperature,
            llm_enable_thinking = args.llm_enable_thinking,
            llm_context_num_steps = args.llm_context_num_steps,
            llm_context_stride    = args.llm_context_stride,
            llm_icrl_mode         = args.llm_icrl_mode,
            llm_skip_episodes     = args.llm_skip_episodes,
        )
        for row in profile_rows:
            row["plant_profile"] = plant_profile
        rows.extend(profile_rows)
    print_results_table(rows)
    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(
            agents=agents,
            scenarios=args.scenarios,
            fixed_action_values=fixed_action_values,
            plant_profile=args.plant_profile,
        )
    )
    save_results(rows, output_path, agents=agents)
