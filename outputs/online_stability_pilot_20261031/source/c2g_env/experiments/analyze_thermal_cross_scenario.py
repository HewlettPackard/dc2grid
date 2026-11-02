"""Merge and summarize the full cross-scenario thermal sensitivity study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from c2g_env.experiments.thermal_sensitivity import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "copilot" / "artifacts"
SCENARIOS = ("default", "scenario_a", "scenario_b", "scenario_c")
CONTROLLERS = ("bang_bang", "pid", "rule_based", "sac")
SEEDS = (100, 101, 102, 103, 104)
FAULT_COLUMNS = (
    "thermal_fault",
    "freq_fault",
    "voltage_fault",
    "soc_fault",
    "sla_fault",
)

# Grid values annotated **S** (declared stress condition) or **NR** (not
# recommended) in thermal_sensitivity.md, excluding the nominal ('N') point for
# each parameter. A row is a "stress condition" only when it deviates from
# nominal onto one of these values; merely-boundary (**B**-only) values do not
# qualify. T_amb=40 is the sweep's only S-without-NR value.
_STRESS_VALUES: dict[str, frozenset[float]] = {
    "K_liq": frozenset({18.8, 25.1}),
    "T_supply_A": frozenset({32.0}),
    "fault_factor": frozenset({0.8, 0.6, 0.4}),
    "T_amb": frozenset({40.0}),
}

# Adverse conditions that the evaluation scenarios themselves impose before any
# sweep override is applied (see ``conf/scenario/*.yaml``). Scenario B runs at
# 40 degC ambient (**S**) and Scenario C declares ``cooling_fault_factor: 0.6``
# (**S/NR**), so their episodes already run a declared stress condition.
_SCENARIO_STRESS_OVERRIDES: dict[str, dict[str, float]] = {
    "scenario_b": {"T_amb": 40.0},
    "scenario_c": {"fault_factor": 0.6},
}


def load_partitions(artifact_dir: Path) -> pd.DataFrame:
    paths = {
        scenario: artifact_dir / f"thermal_sensitivity_cross_scenario_{suffix}.csv"
        for scenario, suffix in {
            "default": "default",
            "scenario_a": "a",
            "scenario_b": "b",
            "scenario_c": "c",
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing scenario partitions: {missing}")

    frames = []
    for expected_scenario, path in paths.items():
        frame = pd.read_csv(path)
        observed = set(frame["scenario"].unique())
        if observed != {expected_scenario}:
            raise ValueError(f"{path.name} contains scenarios {sorted(observed)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    """Concatenate arbitrary sweep partitions, e.g. one file per controller."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sweep partitions: {missing}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def validate_matrix(
    frame: pd.DataFrame,
    scenarios: tuple[str, ...] = SCENARIOS,
    seeds: tuple[int, ...] = SEEDS,
    controllers: tuple[str, ...] = CONTROLLERS,
) -> None:
    key = ["config_id", "scenario", "seed", "hardware_controller"]
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(f"Found {int(duplicated.sum())} rows with duplicate episode keys")
    if not frame["completed"].all():
        raise ValueError(f"Found {int((~frame['completed']).sum())} incomplete episodes")

    configs = tuple(sorted(frame["config_id"].unique()))
    expected = len(configs) * len(scenarios) * len(seeds) * len(controllers)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(frame)}")
    if set(frame["scenario"]) != set(scenarios):
        raise ValueError("Scenario set does not match the configured factorial design")
    if set(frame["seed"]) != set(seeds):
        raise ValueError("Seed set does not match the configured factorial design")
    if set(frame["hardware_controller"]) != set(controllers):
        raise ValueError("Controller set does not match the configured factorial design")


def _is_stress_overrides(overrides: dict[str, float], nominal: dict[str, float]) -> bool:
    """True if ``overrides`` deviates from ``nominal`` onto an S or NR value."""
    for key, value in overrides.items():
        nominal_value = nominal.get(key)
        if nominal_value is not None and np.isclose(value, nominal_value):
            continue
        if any(np.isclose(value, sv) for sv in _STRESS_VALUES.get(key, frozenset())):
            return True
    return False


def add_stress_condition(frame: pd.DataFrame, config_path: Path | None = None) -> pd.DataFrame:
    """Add a ``stress_condition`` column derived from ``thermal_overrides``.

    True when at least one swept thermal parameter in the row's plant
    configuration sits at a declared-stress (**S**) or not-recommended
    (**NR**) grid value per ``thermal_sensitivity.md``.
    """
    nominal = dict(load_config(config_path)["nominal"])
    annotated = frame.copy()
    annotated["stress_condition"] = annotated["thermal_overrides"].apply(
        lambda raw: _is_stress_overrides(json.loads(raw), nominal)
    )
    annotated["scenario_stress_condition"] = annotated["scenario"].map(
        lambda scenario: _is_stress_overrides(
            _SCENARIO_STRESS_OVERRIDES.get(scenario, {}), nominal
        )
    )
    annotated["any_stress_condition"] = (
        annotated["stress_condition"] | annotated["scenario_stress_condition"]
    )
    return annotated


def normalize_termination_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in FAULT_COLUMNS:
        normalized.loc[normalized[column].astype(bool), "termination_reason"] = (
            column.removesuffix("_fault")
        )
    return normalized


def summarize_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["survived"] = ~working["terminated"]
    working["warning_episode"] = working["thermal_warning_steps"] > 0
    group_columns = ["scenario", "hardware_controller", "seed"]
    grouped = working.groupby(group_columns, sort=False)

    seed_summary = grouped.agg(
        episodes=("config_id", "size"),
        survived_episodes=("survived", "sum"),
        survival_rate=("survived", "mean"),
        warning_episodes=("warning_episode", "sum"),
        warning_episode_rate=("warning_episode", "mean"),
        fast_steps=("fast_steps", "sum"),
        warning_steps=("thermal_warning_steps", "sum"),
        temp_max_mean=("temp_max", "mean"),
        temp_max_worst=("temp_max", "max"),
        return_mean=("return", "mean"),
        **{f"{column}_count": (column, "sum") for column in FAULT_COLUMNS},
    ).reset_index()
    seed_summary["warning_step_fraction"] = (
        seed_summary["warning_steps"] / seed_summary["fast_steps"]
    )
    return seed_summary


def summarize_scenarios(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "survival_rate",
        "warning_episode_rate",
        "warning_step_fraction",
        "temp_max_mean",
        "temp_max_worst",
        "return_mean",
    ]
    grouped = seed_summary.groupby(["scenario", "hardware_controller"], sort=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    totals = grouped.agg(
        episodes=("episodes", "sum"),
        survived_episodes=("survived_episodes", "sum"),
        warning_episodes=("warning_episodes", "sum"),
        warning_steps=("warning_steps", "sum"),
        fast_steps=("fast_steps", "sum"),
        **{f"{column}_total": (f"{column}_count", "sum") for column in FAULT_COLUMNS},
    ).reset_index()
    return totals.merge(summary, on=["scenario", "hardware_controller"])


def summarize_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["survived"] = ~working["terminated"]
    working["warning_episode"] = working["thermal_warning_steps"] > 0
    group_columns = ["config_id", "scenario", "hardware_controller"]
    return working.groupby(group_columns, sort=False).agg(
        stress_condition=("stress_condition", "first"),
        scenario_stress_condition=("scenario_stress_condition", "first"),
        any_stress_condition=("any_stress_condition", "first"),
        seeds=("seed", "nunique"),
        survival_rate=("survived", "mean"),
        warning_episode_rate=("warning_episode", "mean"),
        warning_step_fraction_mean=("thermal_warning_fraction", "mean"),
        warning_step_fraction_std=("thermal_warning_fraction", "std"),
        temp_max_mean=("temp_max", "mean"),
        temp_max_std=("temp_max", "std"),
        temp_max_worst=("temp_max", "max"),
        return_mean=("return", "mean"),
        return_std=("return", "std"),
        **{f"{column}_total": (column, "sum") for column in FAULT_COLUMNS},
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="Explicit sweep partitions to merge instead of the scenario files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="thermal_sensitivity_cross_scenario",
        help="Basename prefix for the merged and summarized CSV outputs.",
    )
    args = parser.parse_args()

    if args.inputs:
        frame = load_inputs(list(args.inputs))
        validate_matrix(
            frame,
            scenarios=tuple(sorted(frame["scenario"].unique())),
            seeds=tuple(sorted(frame["seed"].unique())),
            controllers=tuple(sorted(frame["hardware_controller"].unique())),
        )
    else:
        frame = load_partitions(args.artifact_dir)
        validate_matrix(frame)
    frame = normalize_termination_reasons(frame)
    frame = add_stress_condition(frame)
    seed_summary = summarize_by_seed(frame)
    scenario_summary = summarize_scenarios(seed_summary)
    config_summary = summarize_configurations(frame)

    prefix = args.output_prefix
    outputs = {
        f"{prefix}.csv": frame,
        f"{prefix}_seed_summary.csv": seed_summary,
        f"{prefix}_summary.csv": scenario_summary,
        f"{prefix}_config_summary.csv": config_summary,
    }
    for name, output in outputs.items():
        path = args.artifact_dir / name
        output.to_csv(path, index=False)
        print(f"Wrote {len(output):,} rows to {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()