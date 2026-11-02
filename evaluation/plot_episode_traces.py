from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from baselines.metrics_callback import (
    ACTION_SHORT_NAMES,
    REWARD_COLUMNS,
    REWARD_COLUMNS_MACRO,
    STATE_COLUMNS,
    STATE_COLUMNS_MACRO,
    VALID_ACTIONS,
)
from baselines.metrics_callback import build_ablation_suffix
from c2g_env import C2GFastEnv
from c2g_env import C2GMacroEnv

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as exc:
    raise ImportError(
        "matplotlib is required for visualization. Install with: pip install matplotlib"
    ) from exc

def _columns_for_agent_type(agent_type: str) -> tuple[list[str], list[str]]:
    if "hardware" in agent_type:
        return STATE_COLUMNS, REWARD_COLUMNS
    if agent_type == "macro":
        return STATE_COLUMNS_MACRO, REWARD_COLUMNS_MACRO
    raise ValueError(
        f"Invalid agent_type '{agent_type}'. Expected one of ['hardware', 'macro', 'hardware_ha']."
    )


def _ha_safety_bounds_for_agent_type(
    agent_type: str,
    env_bounds: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    # Use HA thresholds from run_ha_benchmark.py and fill one-sided limits
    # with finite environment bounds so both demarcation lines can be drawn.
    if "hardware" in agent_type:
        return {
            "s_temp_A_norm": (env_bounds["s_temp_A_norm"][0], 1.0),
            "s_temp_B_norm": (env_bounds["s_temp_B_norm"][0], 1.0),
            "s_bess_soc": (0.10, 0.95),
            "s_freq_dev_norm": (-1.0, 1.0),
            "s_v_pcc_pu": (0.90, env_bounds["s_v_pcc_pu"][1]),
        }
    if agent_type == "macro":
        return {}
    raise ValueError(
        f"Invalid agent_type '{agent_type}'. Expected one of ['hardware', 'macro', 'hardware_ha']."
    )


def _env_bounds_for_agent_type(agent_type: str, state_columns: list[str]) -> dict[str, tuple[float, float]]:
    """
    Derive observation-space bounds from the appropriate environment.
    Returns a map from state column name to (low, high) tuple.
    """
    if "hardware" in agent_type:
        env = C2GFastEnv()
    elif agent_type == "macro":
        env = C2GMacroEnv()
    else:
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. Expected one of ['hardware', 'macro', 'hardware_ha']."
        )
    
    try:
        obs_space = env.observation_space
        low = np.array(obs_space.low)
        high = np.array(obs_space.high)
        
        bounds = {}
        for idx, col_name in enumerate(state_columns):
            if 0 <= idx < len(low):
                bounds[col_name] = (float(low[idx]), float(high[idx]))
        return bounds
    finally:
        env.close()


def _grid_rows(
    state_columns: list[str],
    reward_columns: list[str],
) -> list[list[str]]:
    # Top two rows are states (split in half); final row is rewards.
    n_cols = int(math.ceil(len(state_columns) / 2))
    row1 = state_columns[:n_cols]
    row2 = state_columns[n_cols:]
    row3 = reward_columns[:]

    return [row1, row2, row3]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _episode_number(path: Path) -> int:
    stem = path.stem
    match = re.match(r"episode(\d+)", stem)
    if match is None:
        return 10**9
    return int(match.group(1))


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
        if action_name not in ACTION_SHORT_NAMES:
            valid = ", ".join(VALID_ACTIONS)
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
        fixed_action_values[action_name] = value
    return fixed_action_values


def _build_ablation_suffix(
    fixed_action_values: dict[str, float],
) -> str:
    return build_ablation_suffix(fixed_action_values)


def _find_episode_csvs(run_dir: Path, ablation_suffix: str = "") -> list[Path]:
    
    episode_csvs = sorted(run_dir.glob("episode*.csv"), key=_episode_number)
    if not ablation_suffix:
        # Return only files matching episode<number>.csv (no ablation suffix)
        return [path for path in episode_csvs if "__" not in path.stem]
    
    return [path for path in episode_csvs if path.stem.endswith(ablation_suffix)]


def _load_episodes(run_dir: Path, ablation_suffix: str = "") -> list[pd.DataFrame]:
    """Load all episodes from a run directory as raw DataFrames."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    
    episode_csvs = _find_episode_csvs(run_dir, ablation_suffix=ablation_suffix)
    if not episode_csvs:
        raise FileNotFoundError(f"No episode CSV files matching experiment config found in {run_dir}")
    
    episodes = []
    for ep_csv in episode_csvs:
        print(f"Loading episode CSV: {ep_csv.name}")
        episodes.append(pd.read_csv(ep_csv))
    return episodes


def _compute_aggregated_stats(
    episodes: list[pd.DataFrame],
    columns: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute mean and 99% confidence interval for each column across episodes.
    Returns: {column_name: {"x": array, "mean": array, "ci_lower": array, "ci_upper": array}}
    """
    all_columns = columns
    
    # Align episodes to same length (pad with NaN if needed)
    max_len = max(len(ep) for ep in episodes)
    aligned_data = {}
    
    for col in all_columns:
        col_data = []
        for ep in episodes:
            # Determine which CSV column to read
            if col == "total_reward":
                csv_col = "r"
            else:
                csv_col = col
            
            # Check if the CSV column exists
            if csv_col not in ep.columns:
                continue
            
            values = ep[csv_col].values
            
            # Cumsum for all reward columns (total_reward and r_*)
            if col == "total_reward" or col.startswith("r_"):
                values = values.cumsum()
            
            # Pad to max length
            if len(values) < max_len:
                values = np.concatenate([values, np.full(max_len - len(values), np.nan)])
            col_data.append(values)
        
        if col_data:
            col_array = np.array(col_data)
            mean = np.nanmean(col_array, axis=0)
            sem = stats.sem(col_array, axis=0, nan_policy="omit")
            ci = sem * stats.t.ppf(0.995, len(episodes) - 1)  # 99% CI
            ci_lower = mean - ci
            ci_upper = mean + ci
            
            aligned_data[col] = {
                "x": np.arange(max_len),
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
    
    return aligned_data


def _trace_positions(grid_rows: list[list[str]]) -> list[tuple[int, int, str]]:
    positions: list[tuple[int, int, str]] = []
    for row_idx, row in enumerate(grid_rows, start=1):
        for col_idx, column in enumerate(row, start=1):
            positions.append((row_idx, col_idx, column))
    return positions


def _set_axis_padding(
    ax,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    env_bounds: tuple[float, float] | None = None,
) -> None:
    """Set y-axis limits using environment bounds if available, otherwise use data-driven padding."""
    if env_bounds is not None:
        # Use environment observation-space bounds directly.
        ax.set_ylim(env_bounds[0], env_bounds[1])
    else:
        # Fall back to data-driven padding.
        finite_lower = ci_lower[np.isfinite(ci_lower)]
        finite_upper = ci_upper[np.isfinite(ci_upper)]
        if finite_lower.size == 0 or finite_upper.size == 0:
            return

        y_min = float(np.min(finite_lower))
        y_max = float(np.max(finite_upper))
        y_span = max(y_max - y_min, 1e-6)

        lower_pad = 0.04 * y_span
        upper_pad = 0.10 * y_span
        ax.set_ylim(y_min - lower_pad, y_max + upper_pad)


def _build_figure_with_stats(
    run_name: str,
    aggregated_stats: dict[str, dict[str, np.ndarray]],
    state_columns: list[str],
    reward_columns: list[str],
    state_safety_bounds: dict[str, tuple[float, float]],
    env_bounds: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    """
    Build figure with mean lines, 99% CI shaded bands, and state reference lines.
    Returns matplotlib figure object.
    """
    grid_rows = _grid_rows(state_columns, reward_columns)
    max_cols = max(len(row) for row in grid_rows)
    n_rows = len(grid_rows)
    figsize_width = max_cols * 2.67  # ~2.67 inches per column

    # Use GridSpec and create axes only where traces exist.
    fig = plt.figure(figsize=(figsize_width, 11.3), constrained_layout=False)
    gs = fig.add_gridspec(n_rows, max_cols)
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.04, right=0.99, hspace=0.42, wspace=0.28)
    fig.suptitle(f"{run_name} — Mean ± 99% CI across all episodes", fontsize=16, y=0.97)
    
    positions = _trace_positions(grid_rows)
    state_columns_set = set(state_columns)
    
    for _, (row_idx, col_idx, column) in enumerate(positions):
        # Convert to 0-indexed for subplot access
        ax = fig.add_subplot(gs[row_idx - 1, col_idx - 1])
        
        is_state = column in state_columns_set
        color = "#1f77b4" if is_state else "#d62728"
        
        if column not in aggregated_stats:
            ax.set_title(column, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        stats_data = aggregated_stats[column]
        x = stats_data["x"]
        mean = stats_data["mean"]
        ci_lower = stats_data["ci_lower"]
        ci_upper = stats_data["ci_upper"]
        
        # Plot confidence band
        ax.fill_between(x, ci_lower, ci_upper, alpha=0.25, color=color, label="99% CI")
        
        # Plot mean line
        ax.plot(x, mean, color=color, linewidth=1.5, label="Mean")

        # Use environment bounds for state variables, data-driven padding for rewards.
        env_bound = None
        if is_state and env_bounds and column in env_bounds:
            env_bound = env_bounds[column]
        _set_axis_padding(ax, ci_lower, ci_upper, env_bounds=env_bound)
        
        # Add HA safety bounds only for variables with explicit audit thresholds.
        if is_state and column in state_safety_bounds:
            lower_bound, upper_bound = state_safety_bounds[column]
            ax.axhline(y=lower_bound, color="#444444", linestyle="--", linewidth=0.8, alpha=0.55)
            ax.axhline(y=upper_bound, color="#444444", linestyle="--", linewidth=0.8, alpha=0.55)
        
        ax.set_title(column, fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
    
    return fig


def generate_episode_plots(
    algoname: str,
    scenario: str,
    agent_type: str = "hardware",
    fixed_action_values: dict[str, float] | None = None,
    runs_dir: Path | str = "runs",
) -> None:
    """Generate aggregated episode statistics plot for a specific algorithm and scenario."""
    agent_type = agent_type.lower()
    state_columns, reward_columns = _columns_for_agent_type(agent_type)
    env_bounds = _env_bounds_for_agent_type(agent_type, state_columns)
    state_safety_bounds = _ha_safety_bounds_for_agent_type(agent_type, env_bounds)

    fixed_action_values = fixed_action_values or {}
    ablation_suffix = _build_ablation_suffix(fixed_action_values)
    
    runs_dir = Path(__file__).resolve().parent.parent / runs_dir
    # Construct the expected run directory
    run_name = f"{algoname}_{scenario}_{agent_type}"
    run_dir = runs_dir / run_name
    if not run_dir.exists():
        # Backward compatibility for directories without agent type suffix.
        fallback_name = f"{algoname}_{scenario}"
        fallback_dir = runs_dir / fallback_name
        if fallback_dir.exists():
            run_name = fallback_name
            run_dir = fallback_dir
    
    print(f"Loading episodes from {run_dir}")
    episodes = _load_episodes(run_dir, ablation_suffix=ablation_suffix)
    print(f"Loaded {len(episodes)} episodes")
    if ablation_suffix:
        print(f"Using ablation suffix filter: {ablation_suffix}")
    
    print("Computing statistics (mean ± 99% CI)...")
    aggregated_stats = _compute_aggregated_stats(episodes, state_columns + reward_columns)
    
    print("Building figure...")
    run_name_display = run_name + ablation_suffix if ablation_suffix else run_name
    fig = _build_figure_with_stats(
        run_name_display,
        aggregated_stats,
        state_columns,
        reward_columns,
        state_safety_bounds,
        env_bounds=env_bounds,
    )
    
    # save images in project/figures
    output_dir = Path(__file__).resolve().parent.parent / "figures"
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory could not be created: {output_dir}")
    
    jpeg_path = output_dir / f"{run_name_display}.jpeg"
    pdf_path = output_dir / f"{run_name_display}.pdf"
    
    # Save JPEG and PDF
    print(f"Saving JPEG: {jpeg_path}")
    fig.savefig(str(jpeg_path), format="jpeg", dpi=100, bbox_inches="tight")
    
    print(f"Saving PDF: {pdf_path}")
    fig.savefig(str(pdf_path), format="pdf", dpi=100, bbox_inches="tight")
    
    plt.close(fig)
    
    # Save HTML with links to image files
    print(f"Done. Generated files:")
    print(f"  - JPEG: {jpeg_path.resolve()}")
    print(f"  - PDF: {pdf_path.resolve()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate episode statistics visualization (JPEG, PDF, HTML) for algorithms."
    )
    parser.add_argument(
        "--algoname",
        required=True,
        help="Algorithm name. Will load from runs/algoname_scenario/",
    )
    parser.add_argument(
        "--scenario",
        default="default",
        help="Scenario name (default: 'default')",
    )
    parser.add_argument(
        "--agent-type",
        default="hardware",
        choices=["hardware", "macro", "hardware_ha"],
        help="Agent type suffix used in runs/<algo>_<scenario>_<agent_type>/ (default: hardware)",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Project Root directory containing algorithm_scenario subdirectories (default: runs)",
    )
    parser.add_argument(
        "--fixed-action",
        action="append",
        default=[],
        help="Fixed value for action in ablation filter, e.g. --fixed-action hvac_effort=0.8",
    )

    args = parser.parse_args()

    try:
        fixed_action_values = _parse_fixed_action_args(args.fixed_action)
    except ValueError as exc:
        parser.error(str(exc))
    
    try:
        generate_episode_plots(
            algoname=args.algoname,
            scenario=args.scenario,
            agent_type=args.agent_type,
            runs_dir=args.runs_dir,
            fixed_action_values=fixed_action_values,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        parser.exit(1)
