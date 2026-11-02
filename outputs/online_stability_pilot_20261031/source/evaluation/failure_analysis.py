"""
evaluation/failure_analysis.py  —  Failure-Case Analysis for HA Benchmark
==========================================================================
Identifies, categorises, and visualises failure modes across agents.

NeurIPS reviewers want to see:
  1. WHERE each method fails (which constraints, which timesteps)
  2. WHY it fails (causal chain: what state led to the violation?)
  3. HOW OFTEN (per-constraint violation breakdown)
  4. WORST CASES (the single most dangerous episode per agent)
  5. COMPARATIVE: where ha_c2g succeeds but ablations fail

This module provides the data structures and analysis pipeline.
Plotting is handled by generate_ha_plots.py (extended).

Usage
-----
  python evaluation/failure_analysis.py evaluation/ha_results_multiseed.csv
  python evaluation/failure_analysis.py results.csv --agents ha_c2g cbm_only --top_k 5
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from c2g_env import C2GFastEnv
from c2g_env.thermal_limits import T_SAFE


# ── C2G obs indices and constraint params ────────────────────────

_I_TEMP_A   = 0
_I_TEMP_B   = 1
_I_SOC      = 2
_I_FREQ_DEV = 14
_I_VPCC     = 15
_I_BACKLOG  = 16

SOC_MIN  = 0.10
SOC_MAX  = 0.95
FREQ_MAX = 0.5   # Hz
V_MIN    = 0.90  # pu

CONSTRAINT_NAMES = ["C1:T_A<35", "C2:T_B<35", "C3:SOC_lo",
                    "C3:SOC_hi", "C4:freq", "C5:voltage"]

SCENARIOS = ["default", "scenario_b", "scenario_c", "high_solar"]


# ═══════════════════════════════════════════════════════════════════
# FAILURE EVENT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConstraintViolation:
    """A single constraint violation at a specific timestep."""
    timestep: int
    constraint: str        # e.g. "C1:T_A<35"
    margin: float          # negative = violated, magnitude = severity
    obs_snapshot: list[float]  # full obs at violation
    action_snapshot: list[float] | None = None


@dataclass
class ShieldIntervention:
    """A single shield intervention event."""
    timestep: int
    raw_action: list[float]
    safe_action: list[float]
    delta_norm: float      # L2 norm of correction
    constraint_at_risk: str | None = None


@dataclass
class EpisodeTrace:
    """Full trace of one episode for failure analysis."""
    agent: str
    scenario: str
    seed: int
    total_reward: float
    episode_length: int
    survived: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    interventions: list[ShieldIntervention] = field(default_factory=list)
    margin_trajectory: list[float] = field(default_factory=list)
    per_constraint_margins: dict[str, list[float]] = field(default_factory=dict)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def n_interventions(self) -> int:
        return len(self.interventions)

    @property
    def violation_rate(self) -> float:
        return self.n_violations / max(self.episode_length, 1)

    @property
    def worst_margin(self) -> float:
        return min(self.margin_trajectory) if self.margin_trajectory else 0.0

    @property
    def violated_constraints(self) -> set[str]:
        return {v.constraint for v in self.violations}

    @property
    def first_violation_step(self) -> int | None:
        return self.violations[0].timestep if self.violations else None


# ═══════════════════════════════════════════════════════════════════
# CONSTRAINT CHECKING (per-constraint breakdown)
# ═══════════════════════════════════════════════════════════════════

def compute_per_constraint_margins(obs: NDArray) -> dict[str, float]:
    """Compute margin to each constraint boundary (negative = violated)."""
    T_A = float(obs[_I_TEMP_A]) * T_SAFE
    T_B = float(obs[_I_TEMP_B]) * T_SAFE
    soc = float(obs[_I_SOC])
    freq_dev = abs(float(obs[_I_FREQ_DEV]) * 0.5)
    v_pcc = float(obs[_I_VPCC])
    return {
        "C1:T_A<35":  T_SAFE - T_A,
        "C2:T_B<35":  T_SAFE - T_B,
        "C3:SOC_lo":  soc - SOC_MIN,
        "C3:SOC_hi":  SOC_MAX - soc,
        "C4:freq":    FREQ_MAX - freq_dev,
        "C5:voltage": v_pcc - V_MIN,
    }


def identify_violations(margins: dict[str, float]) -> list[str]:
    """Return list of violated constraint names."""
    return [name for name, m in margins.items() if m < 0]


def closest_constraint(margins: dict[str, float]) -> tuple[str, float]:
    """Return the constraint closest to violation and its margin."""
    return min(margins.items(), key=lambda x: x[1])


# ═══════════════════════════════════════════════════════════════════
# EPISODE TRACE COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_episode_trace(
    agent,
    shield,
    scenario: str,
    seed: int,
    agent_name: str = "unknown",
) -> EpisodeTrace:
    """
    Run one episode and collect a full failure-analysis trace.

    Parameters
    ----------
    agent : object with .predict(obs, deterministic=True) → (action, _)
    shield : object with .filter(action, obs) → (safe_action, modified, info)
             and .reset()
    scenario : C2G scenario name
    seed : random seed
    agent_name : name for labelling

    Returns
    -------
    EpisodeTrace with all violations, interventions, and margin trajectories.
    """
    env = C2GFastEnv(scenario=scenario)
    obs, _ = env.reset(seed=seed)
    shield.reset()

    trace = EpisodeTrace(
        agent=agent_name,
        scenario=scenario,
        seed=seed,
        total_reward=0.0,
        episode_length=0,
        survived=False,
    )
    trace.per_constraint_margins = {c: [] for c in CONSTRAINT_NAMES}

    done = False
    t = 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        raw_action = np.array(action, dtype=np.float32).copy()

        safe_action, was_modified, shield_info = shield.filter(action, obs)

        if was_modified:
            delta = np.linalg.norm(safe_action - raw_action)
            # Determine which constraint was at risk
            margins_pre = compute_per_constraint_margins(obs)
            at_risk, _ = closest_constraint(margins_pre)
            trace.interventions.append(ShieldIntervention(
                timestep=t,
                raw_action=raw_action.tolist(),
                safe_action=safe_action.tolist() if hasattr(safe_action, 'tolist') else list(safe_action),
                delta_norm=float(delta),
                constraint_at_risk=at_risk,
            ))

        obs, reward, terminated, truncated, info = env.step(safe_action)
        done = terminated or truncated
        trace.total_reward += float(reward)
        t += 1

        # Per-constraint margins
        margins = compute_per_constraint_margins(obs)
        min_margin = min(margins.values())
        trace.margin_trajectory.append(min_margin)

        for cname, cval in margins.items():
            trace.per_constraint_margins[cname].append(cval)

        # Check violations
        violated = identify_violations(margins)
        for cname in violated:
            trace.violations.append(ConstraintViolation(
                timestep=t,
                constraint=cname,
                margin=margins[cname],
                obs_snapshot=obs.tolist(),
                action_snapshot=safe_action.tolist() if hasattr(safe_action, 'tolist') else list(safe_action),
            ))

    trace.episode_length = t
    trace.survived = t >= 288
    return trace


# ═══════════════════════════════════════════════════════════════════
# AGGREGATE FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentFailureProfile:
    """Aggregated failure profile across all seeds for one agent/scenario."""
    agent: str
    scenario: str
    n_seeds: int
    total_violations: int
    total_interventions: int
    mean_violation_rate: float
    mean_intervention_rate: float
    per_constraint_violation_counts: dict[str, int]
    per_constraint_violation_rates: dict[str, float]
    worst_episode_seed: int
    worst_episode_violations: int
    worst_margin_ever: float
    mean_worst_margin: float
    mean_first_violation_step: float | None
    survival_rate: float
    # Intervention analysis
    mean_intervention_magnitude: float
    max_intervention_magnitude: float
    intervention_by_constraint: dict[str, int]


def build_failure_profile(
    traces: list[EpisodeTrace],
) -> AgentFailureProfile:
    """Aggregate EpisodeTraces into an AgentFailureProfile."""
    assert len(traces) > 0
    agent = traces[0].agent
    scenario = traces[0].scenario

    total_viols = sum(t.n_violations for t in traces)
    total_intervs = sum(t.n_interventions for t in traces)

    # Per-constraint violation counts
    per_c_counts: dict[str, int] = {c: 0 for c in CONSTRAINT_NAMES}
    for t in traces:
        for v in t.violations:
            per_c_counts[v.constraint] = per_c_counts.get(v.constraint, 0) + 1

    total_steps = sum(t.episode_length for t in traces)
    per_c_rates = {c: cnt / max(total_steps, 1) for c, cnt in per_c_counts.items()}

    # Worst episode
    worst_trace = max(traces, key=lambda t: t.n_violations)
    worst_margins = [t.worst_margin for t in traces]

    # First violation step
    first_viols = [t.first_violation_step for t in traces if t.first_violation_step is not None]
    mean_first = float(np.mean(first_viols)) if first_viols else None

    # Intervention magnitudes
    all_deltas = [i.delta_norm for t in traces for i in t.interventions]
    interv_by_c: dict[str, int] = {c: 0 for c in CONSTRAINT_NAMES}
    for t in traces:
        for i in t.interventions:
            if i.constraint_at_risk:
                interv_by_c[i.constraint_at_risk] = interv_by_c.get(i.constraint_at_risk, 0) + 1

    return AgentFailureProfile(
        agent=agent,
        scenario=scenario,
        n_seeds=len(traces),
        total_violations=total_viols,
        total_interventions=total_intervs,
        mean_violation_rate=float(np.mean([t.violation_rate for t in traces])),
        mean_intervention_rate=float(np.mean(
            [t.n_interventions / max(t.episode_length, 1) for t in traces])),
        per_constraint_violation_counts=per_c_counts,
        per_constraint_violation_rates=per_c_rates,
        worst_episode_seed=worst_trace.seed,
        worst_episode_violations=worst_trace.n_violations,
        worst_margin_ever=float(min(worst_margins)) if worst_margins else 0.0,
        mean_worst_margin=float(np.mean(worst_margins)),
        mean_first_violation_step=mean_first,
        survival_rate=float(np.mean([1.0 if t.survived else 0.0 for t in traces])),
        mean_intervention_magnitude=float(np.mean(all_deltas)) if all_deltas else 0.0,
        max_intervention_magnitude=float(np.max(all_deltas)) if all_deltas else 0.0,
        intervention_by_constraint=interv_by_c,
    )


# ═══════════════════════════════════════════════════════════════════
# COMPARATIVE FAILURE ANALYSIS (ablation-focused)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ComparativeFailure:
    """Cases where one agent fails but another succeeds."""
    scenario: str
    seed: int
    failing_agent: str
    succeeding_agent: str
    failing_violations: int
    succeeding_violations: int
    failing_worst_margin: float
    succeeding_worst_margin: float
    constraints_only_failing_violates: list[str]


def find_comparative_failures(
    traces_a: list[EpisodeTrace],
    traces_b: list[EpisodeTrace],
    agent_a_name: str = "A",
    agent_b_name: str = "B",
) -> list[ComparativeFailure]:
    """
    Find episodes where agent_a fails but agent_b succeeds.

    Useful for ablation analysis: e.g., where does cbm_only fail
    but ha_c2g succeed?
    """
    # Match by seed
    a_by_seed = {t.seed: t for t in traces_a}
    b_by_seed = {t.seed: t for t in traces_b}

    failures = []
    for seed in sorted(set(a_by_seed) & set(b_by_seed)):
        ta, tb = a_by_seed[seed], b_by_seed[seed]
        # A fails, B succeeds (or B has fewer violations)
        if ta.n_violations > 0 and ta.n_violations > tb.n_violations:
            only_a = ta.violated_constraints - tb.violated_constraints
            failures.append(ComparativeFailure(
                scenario=ta.scenario,
                seed=seed,
                failing_agent=agent_a_name,
                succeeding_agent=agent_b_name,
                failing_violations=ta.n_violations,
                succeeding_violations=tb.n_violations,
                failing_worst_margin=ta.worst_margin,
                succeeding_worst_margin=tb.worst_margin,
                constraints_only_failing_violates=sorted(only_a),
            ))
    return failures


# ═══════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════

def print_failure_report(
    profiles: list[AgentFailureProfile],
    comparisons: list[ComparativeFailure] | None = None,
) -> None:
    """Print a detailed failure-case analysis report."""

    scenarios = sorted(set(p.scenario for p in profiles))
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"  FAILURE ANALYSIS — {scenario}")
        print(f"{'='*80}")

        scene_profiles = sorted(
            [p for p in profiles if p.scenario == scenario],
            key=lambda p: p.total_violations,
        )

        # ── Summary table ─────────────────────────────────────
        print(f"\n  {'Agent':22s} {'Viols':>7s} {'Interv':>7s} "
              f"{'Viol%':>7s} {'IntRt%':>7s} {'WorstM':>8s} "
              f"{'Surv%':>7s}")
        print(f"  {'-'*70}")
        for p in scene_profiles:
            print(f"  {p.agent:22s} {p.total_violations:7d} "
                  f"{p.total_interventions:7d} "
                  f"{p.mean_violation_rate*100:6.2f}% "
                  f"{p.mean_intervention_rate*100:6.2f}% "
                  f"{p.worst_margin_ever:8.4f} "
                  f"{p.survival_rate*100:6.1f}%")

        # ── Per-constraint breakdown ──────────────────────────
        print(f"\n  Per-constraint violation counts:")
        print(f"  {'Agent':22s}", end="")
        for c in CONSTRAINT_NAMES:
            print(f" {c:>10s}", end="")
        print()
        print(f"  {'-'*82}")
        for p in scene_profiles:
            print(f"  {p.agent:22s}", end="")
            for c in CONSTRAINT_NAMES:
                cnt = p.per_constraint_violation_counts.get(c, 0)
                print(f" {cnt:10d}", end="")
            print()

        # ── Worst episodes ────────────────────────────────────
        print(f"\n  Worst episodes:")
        for p in scene_profiles:
            if p.total_violations > 0:
                fvs = f"step {p.mean_first_violation_step:.0f}" if p.mean_first_violation_step else "N/A"
                print(f"    {p.agent:22s}  worst_seed={p.worst_episode_seed}  "
                      f"worst_viols={p.worst_episode_violations}  "
                      f"mean_first_viol={fvs}")

        # ── Intervention analysis ─────────────────────────────
        print(f"\n  Shield intervention magnitudes:")
        for p in scene_profiles:
            if p.total_interventions > 0:
                print(f"    {p.agent:22s}  mean_Δ={p.mean_intervention_magnitude:.4f}  "
                      f"max_Δ={p.max_intervention_magnitude:.4f}")
                # Top constraints triggering interventions
                top_c = sorted(p.intervention_by_constraint.items(),
                               key=lambda x: -x[1])[:3]
                if top_c:
                    cstr = ", ".join(f"{c}({n})" for c, n in top_c if n > 0)
                    print(f"    {'':22s}  triggers: {cstr}")

    # ── Comparative failures ──────────────────────────────────
    if comparisons:
        print(f"\n{'='*80}")
        print(f"  COMPARATIVE FAILURES (ablation → full method)")
        print(f"{'='*80}")
        for cf in sorted(comparisons, key=lambda x: -x.failing_violations)[:20]:
            print(f"  seed={cf.seed:4d}  {cf.failing_agent:15s}→{cf.succeeding_agent:15s}  "
                  f"viols: {cf.failing_violations}→{cf.succeeding_violations}  "
                  f"margin: {cf.failing_worst_margin:.4f}→{cf.succeeding_worst_margin:.4f}  "
                  f"unique_fails: {cf.constraints_only_failing_violates}")


# ═══════════════════════════════════════════════════════════════════
# JSON EXPORT (for plotting / paper figures)
# ═══════════════════════════════════════════════════════════════════

def export_failure_data(
    profiles: list[AgentFailureProfile],
    comparisons: list[ComparativeFailure] | None,
    output: Path,
) -> None:
    """Export failure analysis to JSON for plotting scripts."""
    data = {
        "profiles": [],
        "comparisons": [],
    }
    for p in profiles:
        data["profiles"].append({
            "agent": p.agent,
            "scenario": p.scenario,
            "n_seeds": p.n_seeds,
            "total_violations": p.total_violations,
            "total_interventions": p.total_interventions,
            "mean_violation_rate": p.mean_violation_rate,
            "mean_intervention_rate": p.mean_intervention_rate,
            "per_constraint_violation_counts": p.per_constraint_violation_counts,
            "per_constraint_violation_rates": p.per_constraint_violation_rates,
            "worst_margin_ever": p.worst_margin_ever,
            "mean_worst_margin": p.mean_worst_margin,
            "survival_rate": p.survival_rate,
            "mean_intervention_magnitude": p.mean_intervention_magnitude,
            "max_intervention_magnitude": p.max_intervention_magnitude,
            "intervention_by_constraint": p.intervention_by_constraint,
        })
    if comparisons:
        for cf in comparisons:
            data["comparisons"].append({
                "scenario": cf.scenario,
                "seed": cf.seed,
                "failing_agent": cf.failing_agent,
                "succeeding_agent": cf.succeeding_agent,
                "failing_violations": cf.failing_violations,
                "succeeding_violations": cf.succeeding_violations,
                "constraints_only_failing_violates": cf.constraints_only_failing_violates,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(f"\nFailure data → {output}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Failure-case analysis for HA benchmark")
    parser.add_argument(
        "--agents", nargs="+",
        default=["ha_c2g", "cbm_only", "cbm_gate", "cbm_shield",
                 "simplex_ppo", "cbf_ppo", "random"])
    parser.add_argument("--scenarios", nargs="+", default=["default"])
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output", default="evaluation/failure_analysis.json")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of worst comparative failures to show")
    args = parser.parse_args()

    # Lazy imports to avoid dependency issues in test environments
    from evaluation.run_ha_benchmark import load_agent, get_shield

    all_traces: dict[str, list[EpisodeTrace]] = {}
    all_profiles: list[AgentFailureProfile] = []

    for scenario in args.scenarios:
        for agent_name in args.agents:
            print(f"\nCollecting traces: {agent_name}/{scenario} "
                  f"({args.n_seeds} seeds)...")
            agent, _ = load_agent(
                agent_name, scenario, args.seed_start, args.model_dir)
            shield_eval = get_shield(agent_name)

            traces = []
            for i in range(args.n_seeds):
                seed = args.seed_start + i
                trace = collect_episode_trace(
                    agent, shield_eval, scenario, seed, agent_name)
                traces.append(trace)

            key = f"{agent_name}/{scenario}"
            all_traces[key] = traces
            profile = build_failure_profile(traces)
            all_profiles.append(profile)

    # Comparative: ablation vs ha_c2g
    comparisons: list[ComparativeFailure] = []
    for scenario in args.scenarios:
        ha_key = f"ha_c2g/{scenario}"
        if ha_key not in all_traces:
            continue
        ha_traces = all_traces[ha_key]
        for ablation in ["cbm_only", "cbm_gate", "cbm_shield"]:
            abl_key = f"{ablation}/{scenario}"
            if abl_key not in all_traces:
                continue
            cfs = find_comparative_failures(
                all_traces[abl_key], ha_traces,
                ablation, "ha_c2g")
            comparisons.extend(cfs)

    print_failure_report(all_profiles, comparisons)
    export_failure_data(all_profiles, comparisons, Path(args.output))
