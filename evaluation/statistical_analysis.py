"""
evaluation/statistical_analysis.py  —  Multi-Seed Statistical Analysis
=======================================================================
Computes confidence intervals, significance tests, and summary tables
from multi-seed HA benchmark results.

Designed for NeurIPS-quality reporting:
  - Per-agent, per-scenario bootstrap CIs on all 11 metrics
  - Welch's t-test (or Mann-Whitney U) for pairwise comparisons
  - LaTeX table generation with CIs and significance markers
  - Effect size (Cohen's d) for key comparisons

Usage
-----
  python evaluation/statistical_analysis.py evaluation/ha_results_multiseed.csv
  python evaluation/statistical_analysis.py results.csv --baseline ha_c2g --alpha 0.05
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from c2g_env.experiments.identity import CONDITION_COLUMNS, configuration_id, result_condition


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MetricSummary:
    """Statistical summary of a metric across seeds."""
    metric: str
    agent: str
    scenario: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    median: float
    iqr_lower: float      # 25th percentile
    iqr_upper: float      # 75th percentile
    min_val: float
    max_val: float
    n_seeds: int
    ci_level: float = 0.95
    condition: dict[str, Any] = field(default_factory=dict)

    @property
    def ci_half_width(self) -> float:
        return (self.ci_upper - self.ci_lower) / 2


def bootstrap_ci(
    values: NDArray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values : 1-D array of metric values across seeds.
    confidence : CI level (default 0.95).
    n_bootstrap : Number of bootstrap resamples.
    rng : Numpy random generator.

    Returns
    -------
    (ci_lower, ci_upper) : Confidence interval bounds.
    """
    if len(values) < 2:
        v = float(values[0]) if len(values) == 1 else 0.0
        return v, v
    rng = rng or np.random.default_rng(42)
    boot_means = np.empty(n_bootstrap)
    n = len(values)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    alpha = 1 - confidence
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def t_ci(values: NDArray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Student-t confidence interval (parametric, for comparison).
    Falls back to bootstrap if n < 4.
    """
    n = len(values)
    if n < 2:
        v = float(values[0]) if n == 1 else 0.0
        return v, v
    if n < 4:
        return bootstrap_ci(values, confidence)

    from scipy import stats
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(mean - t_crit * se), float(mean + t_crit * se)


def summarise_metric(
    values: NDArray,
    metric: str,
    agent: str,
    scenario: str,
    confidence: float = 0.95,
    use_bootstrap: bool = True,
) -> MetricSummary:
    """Compute full statistical summary for one metric/agent/scenario."""
    values = np.asarray(values, dtype=float)
    ci_fn = bootstrap_ci if use_bootstrap else t_ci
    ci_lo, ci_hi = ci_fn(values, confidence)
    return MetricSummary(
        metric=metric,
        agent=agent,
        scenario=scenario,
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        median=float(np.median(values)),
        iqr_lower=float(np.percentile(values, 25)),
        iqr_upper=float(np.percentile(values, 75)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        n_seeds=len(values),
        ci_level=confidence,
    )


# ═══════════════════════════════════════════════════════════════════
# SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PairwiseComparison:
    """Result of a pairwise statistical comparison."""
    metric: str
    scenario: str
    agent_a: str
    agent_b: str
    mean_a: float
    mean_b: float
    delta: float           # mean_a - mean_b
    p_value: float
    test_name: str         # "welch_t" or "mann_whitney_u"
    significant: bool      # p < alpha
    effect_size: float     # Cohen's d
    effect_label: str      # "negligible", "small", "medium", "large"
    alpha: float = 0.05
    higher_is_better: bool = True  # for interpreting the sign
    condition: dict[str, Any] = field(default_factory=dict)


def cohens_d(a: NDArray, b: NDArray) -> float:
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def pairwise_test(
    values_a: NDArray,
    values_b: NDArray,
    metric: str,
    scenario: str,
    agent_a: str,
    agent_b: str,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> PairwiseComparison:
    """
    Run a pairwise significance test between two agents.

    Uses Welch's t-test if n ≥ 5 per group, Mann-Whitney U otherwise.
    """
    from scipy import stats

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    if len(values_a) >= 5 and len(values_b) >= 5:
        stat, p = stats.ttest_ind(values_a, values_b, equal_var=False)
        test_name = "welch_t"
    else:
        stat, p = stats.mannwhitneyu(
            values_a, values_b, alternative="two-sided")
        test_name = "mann_whitney_u"

    d = cohens_d(values_a, values_b)

    return PairwiseComparison(
        metric=metric,
        scenario=scenario,
        agent_a=agent_a,
        agent_b=agent_b,
        mean_a=float(np.mean(values_a)),
        mean_b=float(np.mean(values_b)),
        delta=float(np.mean(values_a) - np.mean(values_b)),
        p_value=float(p),
        test_name=test_name,
        significant=float(p) < alpha,
        effect_size=d,
        effect_label=effect_label(d),
        alpha=alpha,
        higher_is_better=higher_is_better,
    )


# ═══════════════════════════════════════════════════════════════════
# MULTI-SEED ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════

# Metrics where lower is better (for correct interpretation)
LOWER_IS_BETTER = {
    "hard_violation_rate",
    "shield_intervention_rate",
    "thermal_viol_rate",
    "tracking_rmse",
    "bess_degradation",
    "computational_overhead_ms",
}

# Key HA metrics for the NeurIPS tables
HA_KEY_METRICS = [
    "mean_reward",
    "hard_violation_rate",
    "shield_intervention_rate",
    "constraint_margin",
    "worst_case_margin",
    "survival_rate",
]


def load_multiseed_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV with per-seed rows (requires 'seed' column)."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def group_by_agent_scenario(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Group rows by (agent, scenario, capacity/thermal condition identity)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["agent"]), str(row["scenario"]), configuration_id(result_condition(row)))
        groups.setdefault(key, []).append(row)
    return groups


def compute_all_summaries(
    rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
    confidence: float = 0.95,
    use_bootstrap: bool = True,
) -> list[MetricSummary]:
    """Compute MetricSummary for every (agent, scenario, metric) triple."""
    groups = group_by_agent_scenario(rows)
    if metrics is None:
        # Auto-detect numeric columns
        sample = rows[0]
        metrics = [k for k, v in sample.items()
                   if isinstance(v, (int, float)) and k not in ("seed", "n_episodes", "value")
                   and k not in CONDITION_COLUMNS]

    summaries = []
    for (agent, scenario, _condition_id), group_rows in sorted(groups.items()):
        for metric in metrics:
            values = np.array([r[metric] for r in group_rows if metric in r])
            if len(values) == 0:
                continue
            s = summarise_metric(values, metric, agent, scenario, confidence,
                                use_bootstrap=use_bootstrap)
            s.condition = result_condition(group_rows[0])
            summaries.append(s)
    return summaries


def compute_pairwise_comparisons(
    rows: list[dict[str, Any]],
    baseline: str = "ha_c2g",
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> list[PairwiseComparison]:
    """Compare every agent against a baseline on key metrics."""
    groups = group_by_agent_scenario(rows)
    if metrics is None:
        metrics = HA_KEY_METRICS

    comparisons = []
    ordered_groups = sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
    for (agent, scenario, condition_id), agent_rows in ordered_groups:
        if agent == baseline:
            continue
        baseline_rows = groups.get((baseline, scenario, condition_id), [])
        if not baseline_rows:
            continue
        for metric in metrics:
            vals_base = np.array([r[metric] for r in baseline_rows if metric in r])
            vals_agent = np.array([r[metric] for r in agent_rows if metric in r])
            if len(vals_base) < 2 or len(vals_agent) < 2:
                continue
            higher_better = metric not in LOWER_IS_BETTER
            comp = pairwise_test(
                vals_base, vals_agent, metric, scenario,
                baseline, agent, alpha, higher_better)
            comp.condition = result_condition(agent_rows[0])
            comparisons.append(comp)
    return comparisons


# ═══════════════════════════════════════════════════════════════════
# LATEX TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════

def _fmt_ci(s: MetricSummary, precision: int = 3) -> str:
    """Format mean ± CI for a LaTeX cell."""
    return f"${s.mean:.{precision}f} \\pm {s.ci_half_width:.{precision}f}$"


def _fmt_ci_sig(
    s: MetricSummary,
    comp: PairwiseComparison | None,
    bold_best: bool = True,
    precision: int = 3,
) -> str:
    """Format mean ± CI with significance marker."""
    txt = _fmt_ci(s, precision)
    if comp is not None and comp.significant:
        # Add significance star
        if comp.p_value < 0.001:
            txt += "$^{***}$"
        elif comp.p_value < 0.01:
            txt += "$^{**}$"
        else:
            txt += "$^{*}$"
    return txt


def generate_latex_table(
    summaries: list[MetricSummary],
    comparisons: list[PairwiseComparison] | None = None,
    metrics: list[str] | None = None,
    scenario: str = "default",
    caption: str = "High-assurance benchmark results",
    label: str = "tab:ha_benchmark",
    condition: dict[str, Any] | None = None,
) -> str:
    """
    Generate a publication-ready LaTeX table.

    Rows = agents, Columns = metrics, Cells = mean ± CI with sig markers.
    """
    if metrics is None:
        metrics = HA_KEY_METRICS

    # Filter to scenario
    sums = [s for s in summaries if s.scenario == scenario]
    conditions = {configuration_id(summary.condition): summary.condition for summary in sums}
    if condition is None and len(conditions) > 1:
        raise ValueError("Multiple capacity/thermal conditions; select a condition explicitly.")
    if condition is None:
        condition = next(iter(conditions.values()), {})
    sums = [summary for summary in sums if summary.condition == condition]
    agents = sorted(set(s.agent for s in sums))

    # Build lookup: (agent, metric) → MetricSummary
    sum_lookup: dict[tuple[str, str], MetricSummary] = {}
    for s in sums:
        sum_lookup[(s.agent, s.metric)] = s

    # Build comp lookup: (agent, metric) → PairwiseComparison
    comp_lookup: dict[tuple[str, str], PairwiseComparison] = {}
    if comparisons:
        for c in comparisons:
            if c.scenario == scenario and c.condition == condition:
                comp_lookup[(c.agent_b, c.metric)] = c

    # Find best agent per metric
    best_agent: dict[str, str] = {}
    for metric in metrics:
        lower_better = metric in LOWER_IS_BETTER
        best_val = float("inf") if lower_better else float("-inf")
        best_a = ""
        for agent in agents:
            s = sum_lookup.get((agent, metric))
            if s is None:
                continue
            if (lower_better and s.mean < best_val) or \
               (not lower_better and s.mean > best_val):
                best_val = s.mean
                best_a = agent
        best_agent[metric] = best_a

    # Short metric labels
    short = {
        "mean_reward": "Reward",
        "hard_violation_rate": "Hard Viol. ↓",
        "shield_intervention_rate": "Shield Int. ↓",
        "constraint_margin": "Margin ↑",
        "worst_case_margin": "Worst Margin ↑",
        "survival_rate": "Survival ↑",
        "tracking_rmse": "RMSE ↓",
        "thermal_viol_rate": "Therm. Viol. ↓",
        "throughput_ratio": "Throughput ↑",
        "bess_degradation": "BESS Deg. ↓",
        "computational_overhead_ms": "Overhead (ms) ↓",
    }

    # Agent display names
    agent_names = {
        "simplex_ppo": "Simplex-PPO",
        "cbf_ppo": "CBF-PPO",
        "hj_ppo": "HJ-PPO",
        "mpcsf_ppo": "MPC-SF-PPO",
        "ppo_lagrangian": "PPO-Lag",
        "cpo": "CPO",
        "reward_shaping": "Reward Shaping",
        "ha_c2g": "\\textbf{HA-C2G}",
        "cbm_only": "CBM Only",
        "cbm_gate": "CBM+Gate",
        "cbm_shield": "CBM+Shield",
        "random": "Random",
    }

    ncols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)

    lines = [
        f"\\begin{{table}}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\centering",
        f"\\resizebox{{\\textwidth}}{{!}}{{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
    ]

    # Header
    header = "Agent & " + " & ".join(short.get(m, m) for m in metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for agent in agents:
        display = agent_names.get(agent, agent)
        cells = [display]
        for metric in metrics:
            s = sum_lookup.get((agent, metric))
            if s is None:
                cells.append("—")
                continue
            comp = comp_lookup.get((agent, metric))
            prec = 2 if "margin" in metric or metric == "mean_reward" else 3
            txt = _fmt_ci_sig(s, comp, precision=prec)
            if agent == best_agent.get(metric):
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        f"\\vspace{{0.5em}}",
        f"\\\\\\footnotesize{{$^{{***}}p<0.001$, $^{{**}}p<0.01$, "
        f"$^{{*}}p<0.05$ vs. HA-C2G (Welch's $t$-test or Mann-Whitney $U$ "
        f"when $n<5$). CIs from $N$ seeds.}}",
        f"\\end{{table}}",
    ])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════

def print_summary_report(
    summaries: list[MetricSummary],
    comparisons: list[PairwiseComparison],
    metrics: list[str] | None = None,
) -> None:
    """Print a readable console report with CIs and significance."""
    if metrics is None:
        metrics = HA_KEY_METRICS

    conditions = {(summary.scenario, configuration_id(summary.condition)): summary.condition
                  for summary in summaries}
    for (scenario, _condition_id), condition in sorted(conditions.items()):
        print(f"\n{'='*80}")
        print(f"  Scenario: {scenario}")
        if condition:
            print(f"  Condition: {json.dumps(condition, sort_keys=True)}")
        print(f"{'='*80}")

        sums = [s for s in summaries if s.scenario == scenario and s.condition == condition]
        agents = sorted(set(s.agent for s in sums))

        for metric in metrics:
            lower_better = metric in LOWER_IS_BETTER
            arrow = "↓" if lower_better else "↑"
            print(f"\n  {metric} ({arrow})")
            print(f"  {'-'*60}")
            for agent in agents:
                ms = next((s for s in sums if s.agent == agent and s.metric == metric), None)
                if ms is None:
                    continue
                sig = ""
                comp = next(
                    (c for c in comparisons
                     if c.scenario == scenario
                     and c.condition == condition
                     and c.metric == metric
                     and c.agent_b == agent),
                    None,
                )
                if comp is not None:
                    stars = "***" if comp.p_value < 0.001 else \
                            "**" if comp.p_value < 0.01 else \
                            "*" if comp.p_value < 0.05 else ""
                    sig = f"  p={comp.p_value:.4f}{stars}  d={comp.effect_size:+.2f}({comp.effect_label})"
                print(
                    f"    {agent:22s}  {ms.mean:8.4f} ± {ms.ci_half_width:.4f}  "
                    f"[{ms.ci_lower:.4f}, {ms.ci_upper:.4f}]  "
                    f"n={ms.n_seeds}{sig}"
                )


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical analysis of multi-seed HA benchmark results")
    parser.add_argument("csv", type=Path, help="Path to multi-seed results CSV")
    parser.add_argument("--baseline", default="ha_c2g", help="Baseline agent for pairwise tests")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--confidence", type=float, default=0.95, help="CI level")
    parser.add_argument("--latex", type=Path, default=None, help="Output LaTeX table file")
    parser.add_argument("--scenario", default="default", help="Scenario for LaTeX table")
    parser.add_argument("--ci_method", choices=["bootstrap", "t"], default="bootstrap",
                        help="CI method: 'bootstrap' (non-parametric) or 't' (Student-t)")
    args = parser.parse_args()

    rows = load_multiseed_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    use_bootstrap = args.ci_method == "bootstrap"
    summaries = compute_all_summaries(rows, confidence=args.confidence,
                                      use_bootstrap=use_bootstrap)
    print(f"CI method: {args.ci_method}  confidence: {args.confidence}")
    comparisons = compute_pairwise_comparisons(
        rows, baseline=args.baseline, alpha=args.alpha)

    print_summary_report(summaries, comparisons)

    if args.latex:
        conditions = {configuration_id(summary.condition): summary.condition for summary in summaries
                      if summary.scenario == args.scenario}
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        for condition_id, condition in conditions.items():
            output = args.latex
            if len(conditions) > 1:
                output = args.latex.with_name(f"{args.latex.stem}_{condition_id[:12]}{args.latex.suffix}")
            latex = generate_latex_table(
                summaries, comparisons, scenario=args.scenario, condition=condition,
                caption=f"HA benchmark results ({args.scenario}), {args.confidence*100:.0f}\\% CIs",
                label="tab:ha_benchmark" + (f"_{condition_id[:12]}" if condition else ""),
            )
            output.write_text(latex, encoding="utf-8")
            if condition:
                output.with_suffix(".condition.json").write_text(json.dumps(condition, indent=2), encoding="utf-8")
            print(f"\nLaTeX table: {output}")

    # Print significant findings
    sig_comps = [c for c in comparisons if c.significant]
    if sig_comps:
        print(f"\n{'='*80}")
        print(f"  SIGNIFICANT DIFFERENCES vs {args.baseline} (α={args.alpha})")
        print(f"{'='*80}")
        for c in sorted(sig_comps, key=lambda x: x.p_value):
            better_worse = ""
            if c.higher_is_better:
                better_worse = "better" if c.delta > 0 else "worse"
            else:
                better_worse = "better" if c.delta < 0 else "worse"
            print(
                f"  {c.agent_b:20s} {c.metric:30s}  "
                f"Δ={c.delta:+.4f}  p={c.p_value:.4f}  "
                f"d={c.effect_size:+.2f}  {better_worse}"
                f"  condition={json.dumps(c.condition, sort_keys=True)}"
            )
