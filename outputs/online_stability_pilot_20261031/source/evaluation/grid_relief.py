"""
Grid-stress-relief metrics for hierarchical controllers (camera-ready analysis).

Runs ``rule_macro + <hw>`` episodes and a paired counterfactual baseline
(``rule_macro`` with no inner controller: levers fixed at 1/0.7/0.7/0) with
identical seeds, logs per-step facility power, grid load, frequency and
regulation response, and computes:

  Load relief vs counterfactual  ΔP_t = B_t - A_t  (MW)
    dP_stress_mw     mean ΔP over stress ticks S (top-decile grid load)
    LFR_pct          Σ_S ΔP / Σ_S B                     (load flexibility ratio)
    SPDR_pct         mean_S ΔP / max_t(L_t + B_t)        (system peak demand relief)
    LWSR_pct         Σ w ΔP / Σ w (L+B), w = max(0, L - L_median)
                     (stress-weighted; weight ∝ LMP scarcity adder, following
                      the LOLP-weighting method of Madaeni et al. 2013)
    LWCF_pct         (Σ w ΔP / Σ w) / committed_mw_max   (ELCC proxy)
    corr_P_L         corr(A_t, L_t)
  Counterfactual price relief (post hoc; LMP is exogenous in the simulator)
    dLMP_stress      mean_S [LMP(L+B) - LMP(L+A)]        ($/MWh)
    avoided_cost_usd Σ_t [LMP(L+B) - LMP(L+A)] · L_t · Δt
  Frequency support (coupled through the swing equation)
    f_rms_mhz, f_max_dev_mhz, f_outside_db_pct (|Δf| > 0.2 Hz)
  Regulation service delivered
    reg_energy_mwh   Σ |ΔP_act| Δt
    reg_mileage_mw   Σ |ΔP_act_t - ΔP_act_{t-1}|
    perf_ratio       1 - Σ|err| / Σ|ΔP_dem|
  Guardrails
    agent_ticks, baseline_ticks, paired_ticks (episode and comparison lengths)
    ramp_max_mw_min  max |dA/dt|
    rebound_pct      Σ_{not S} (A - B)+ / Σ_S (B - A)+
    throughput_ratio, backlog_mean_mw

Pass ``--trace-dir`` to retain compressed hardware and macro traces, including
observed versus applied commitments, actuator state, and settlement components.

Usage
-----
    .venv/bin/python evaluation/grid_relief.py \
        --hw rule_based pid bang_bang sac \
        --hw-model-dir trained_models/BestCheckpoints/sac_default_s100 \
        --scenarios default scenario_a scenario_b scenario_c \
        --markets nyiso_nyc pjm_dom caiso_pgae ercot_north entso_de aemo_nsw \
        --n_episodes 5 --seed 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from c2g_env.experiments.identity import file_fingerprint, source_id
from c2g_env.semantics import SIMULATION_MODES

STRESS_QUANTILE = 0.90
F_DEADBAND_HZ = 0.2
T_CRIT_ONE_SIDED_95 = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}
RELIEF_KEYS = ("dP_stress_mw", "LFR_pct", "SPDR_pct", "LWSR_pct", "LWCF_pct")
TRACE_INFO_FIELDS = (
    "tick", "bess_actual_kw", "bess_soc", "flex_reduction_kw", "cool_delta_kw",
    "temp_A", "temp_B", "p_cooling_mw", "p_flex_nom_kw", "p_flex_served_kw",
    "pump_speed_A", "hvac_effort", "bess_dispatch", "thermal_warning",
    "thermal_fault", "freq_fault", "voltage_fault", "terminated", "truncated",
    "reward_throughput", "reward_tracking", "reward_thermal", "reward_soc",
    "reward_freq", "reward_volt", "reward_backlog",
)
METRIC_TRACE_COLUMNS = (
    "p_fac_mw", "load_norm", "f_dev_hz", "dp_act_kw", "dp_dem_kw",
    "err_kw", "throttle", "backlog_kw", "lmp", "load_mw",
)
PHYSICAL_TRACE_COLUMNS = (
    "timestamp", "p_grid_import_mw", "p_reference_mw", "facility_relief_kw",
    "work_arrived_kwh", "work_served_kwh", "backlog_work_kwh", "throughput_ratio",
    "p_it_A_mw", "p_it_B_mw", "p_chiller_mw", "p_conversion_loss_mw",
    "regd_observed",
)


def _market_config(market: str, directory: str, simulation_mode: str) -> str:
    # The grid simulator reads grid_market/nyiso_zone from the *global* block, not the scenario.
    cfg = yaml.safe_load((ROOT / "c2g_env" / "config.yaml").read_text())
    cfg["global"]["grid_market"] = market
    cfg["global"]["nyiso_zone"] = None
    cfg["global"]["simulation_mode"] = simulation_mode
    for block in cfg.values():
        if simulation_mode == "physical_v1" and isinstance(block, dict) and "weather_market" in block:
            block["weather_market"] = market
            block["grid_market"] = market
    path = Path(directory) / f"{market}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _align_calendar(fast_env, start_day: int) -> np.ndarray:
    """Replay calendar ``start_day`` of the load file at real time (the env advances one 5-min row per 5-s tick)."""
    g, n_ticks = fast_env._grid, int(fast_env._episode_ticks)
    steps_per_row = max(1, round(300.0 / float(fast_env._dt)))
    ts = (pd.read_csv(ROOT / "data" / "processed" / "energy" / f"{g._market.zone_csv}.csv",
                      usecols=["Time Stamp", "Load"], parse_dates=["Time Stamp"])
          .dropna(subset=["Load"]).sort_values("Time Stamp")["Time Stamp"].to_numpy())
    r0 = int(np.searchsorted(ts, ts[0] + np.timedelta64(int(start_day), "D")))
    n_rows = n_ticks // steps_per_row + 1
    day = np.repeat(g._load_mw[r0:r0 + n_rows], steps_per_row)[:n_ticks]
    # Keep the original series as a tail so load_norm stays normalised by the annual max.
    g._load_mw = np.concatenate([day, g._load_mw])
    g._n = len(g._load_mw)
    w = fast_env._weather
    if getattr(w, "_temps_c", None) is not None:
        w._temps_c = np.roll(w._temps_c, -int(start_day) * 24)
        w._dewpoints_c = np.roll(w._dewpoints_c, -int(start_day) * 24)
    return day


def _bias_regd(grid, day_load: np.ndarray, beta: float) -> None:
    """Add beta * s_t to RegD, s_t = day-ahead load anomaly in [-1, 1] (+ = reduce draw at peak)."""
    mean, peak = day_load.mean(), day_load.max()
    stress = np.clip((day_load - mean) / max(peak - mean, 1e-9), -1.0, 1.0)
    step_regd = grid._step_regd

    def biased() -> float:
        return float(np.clip(step_regd() + beta * stress[grid._tick % len(stress)], -1.0, 1.0))

    grid._step_regd = biased


def _run_episode(hw: str, scenario: str, seed: int, model_dir: str | None,
                 macro: str = "rule_macro", macro_model_dir: str | None = None,
                 config_path: str | None = None, start_day: int | None = None,
                 regd_bias: float = 0.0, trace_dir: str | None = None,
                 simulation_mode: str = "physical_v1",
                 allow_semantic_mismatch: bool = False) -> pd.DataFrame:
    import torch
    torch.set_num_threads(1)
    import run_benchmark as rb

    from baselines.rule_based_macro import RuleBasedMacroController
    from c2g_env.obs_indices import Fast as _F

    inner_fn = None
    ctrl = None
    if hw != "none":
        inner_env = rb._make_env(scenario=scenario, config_path=config_path,
                                 simulation_mode=simulation_mode)
        inner_env.reset(seed=0)
        ctrl = rb._make_inner_controller(hw, env=inner_env, scenario=scenario,
                                         seed=seed, model_dir=model_dir,
                                         allow_semantic_mismatch=allow_semantic_mismatch)
        inner_fn = lambda obs, _a, c=ctrl: c.predict(obs)[0]  # noqa: E731
        inner_env.close()

    rows: list[dict] = []

    def cb(pre_obs, low_action, next_obs, rew, done, info):
        rows.append({
            "p_fac_mw": float(info["p_facility_mw"]),
            "load_norm": float(next_obs[8]),
            "f_dev_hz": float(info["freq_dev_hz"]),
            "dp_act_kw": float(info["delta_p_actual_kw"]),
            "dp_dem_kw": float(info["delta_p_demanded_kw"]),
            "err_kw": float(info["tracking_err_kw"]),
            "throttle": float(info["throttle_batch"]),
            "backlog_kw": float(info["backlog_kw"]),
            "lmp": float(info["lmp"]),
            "regd_observed": float(pre_obs[_F.REGD]),
            "committed_observed_mw": float(pre_obs[_F.COMMITTED] * env._committed_max_mw),
            "committed_actual_mw": float(env._fast_env.committed_mw),
            **{key: info[key] for key in TRACE_INFO_FIELDS},
            **{key: info[key] for key in PHYSICAL_TRACE_COLUMNS if key in info},
            **({"load_mw": info["grid_load_mw"]} if "grid_load_mw" in info else {}),
        })

    env = rb._make_macro_env(scenario=scenario, inner_action_fn=inner_fn, sub_step_callback=cb,
                             config_path=config_path, simulation_mode=simulation_mode,
                             start_day=(start_day or 0) if simulation_mode == "physical_v1" else 0,
                             regd_load_bias=regd_bias if simulation_mode == "physical_v1" else 0)
    obs, _ = env.reset(seed=seed)
    initial_hardware_obs = env._last_hw_obs.tolist()
    if simulation_mode == "legacy_v0" and start_day is not None:
        day_load = _align_calendar(env._fast_env, start_day)
        if regd_bias:
            _bias_regd(env._fast_env._grid, day_load, regd_bias)
    elif simulation_mode == "legacy_v0" and regd_bias:
        raise ValueError("--regd-load-bias requires --start-days (calendar-aligned load)")
    if macro == "rule_macro":
        agent = RuleBasedMacroController()
    else:
        agent = rb.load_sb3_agent(
            macro, scenario, seed, macro_model_dir, simulation_mode=simulation_mode,
            allow_semantic_mismatch=allow_semantic_mismatch,
        )
    revenue, done = 0.0, False
    macro_rows = []
    while not done:
        a, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        revenue += float(info.get("regulation_revenue", 0.0))
        macro_rows.append({
            **{key: value for key, value in info.items() if key != "last_inner_info"},
            "duration_hours": info["sub_steps_run"] * env._fast_env._dt / 3600.0,
            "settlement_hours": info["settlement_hours"],
            "terminated": term,
            "truncated": trunc,
        })
        done = term or trunc

    g = env._fast_env._grid
    df = pd.DataFrame(rows)
    macro_df = pd.DataFrame(macro_rows)
    if simulation_mode == "legacy_v0":
        df["load_mw"] = df["load_norm"] * float(np.max(g._load_mw))
    df.attrs.update(
        load_median=float(g._load_median), lmp_base=float(g.lmp_base_usd),
        lmp_slope=float(g.lmp_slope), dt_s=float(env._fast_env._dt),
        c_max=float(getattr(env, "_committed_max_mw", 30.0)),
        revenue_usd=revenue,
        simulation_mode=simulation_mode,
        expected_ticks=env._fast_env._episode_ticks,
        terminated=bool(term),
        truncated=bool(trunc),
        weather_source=env._fast_env._weather.source,
        diagnostics=_episode_diagnostics(df, macro_df, float(env._fast_env._dt)),
    )
    if simulation_mode == "physical_v1":
        df.attrs["stress_threshold_mw"] = float(np.quantile(
            g._calendar_load_mw[:env._fast_env._episode_ticks], STRESS_QUANTILE,
        ))
    if trace_dir is not None:
        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=False)
        df.to_csv(directory / "hardware.csv.gz", index=False, compression="gzip")
        macro_df.to_csv(directory / "macro.csv.gz", index=False, compression="gzip")
        context = {
            "scenario": scenario, "seed": seed, "start_day": start_day,
            "regd_bias": regd_bias, "macro": macro, "hardware": hw,
            "initial_hardware_observation": initial_hardware_obs,
            "hardware_checkpoint": getattr(ctrl, "checkpoint", None),
            "macro_checkpoint": getattr(agent, "checkpoint", None),
            "hardware_steps": len(df), "macro_steps": len(macro_df),
            "terminated": bool(term), "truncated": bool(trunc),
            "metric_context": {key: value for key, value in df.attrs.items() if key != "diagnostics"},
            "diagnostics": df.attrs["diagnostics"],
            "simulation_mode": simulation_mode,
            "source_id": source_id(ROOT),
            "config": {"global": env._fast_env._gcfg, "scenario": env._fast_env._scfg,
                       "reward": env._fast_env._rcfg},
            "data_files": {
                name: file_fingerprint(path)
                for name, path in {
                    "load": ROOT / "data/processed/energy" / f"{g._market.zone_csv}.csv",
                    "batch": ROOT / "data/processed/workload_traces/batch_v2023.csv",
                    "rigid": ROOT / "data/processed/workload_traces/genai_v2026.csv",
                    "inference": ROOT / "data/processed/workload_traces/dlrm_v2025.csv",
                    "weather": ROOT / "data/processed/weather" / f"{env._fast_env._weather._preset.zone_file}.csv",
                }.items()
            },
        }
        (directory / "episode.json").write_text(
            json.dumps(context, indent=2, allow_nan=False), encoding="utf-8",
        )
    env.close()
    extra = [key for key in PHYSICAL_TRACE_COLUMNS if key in df]
    return df.loc[:, list(METRIC_TRACE_COLUMNS) + extra]


def _episode_diagnostics(hardware, macro, dt_s):
    hours = macro["duration_hours"].to_numpy()
    before_discount = float(
        (macro["rate"] * macro["committed_mw"] * macro["settlement_hours"]).sum()
    )
    return {
        "mean_committed_mw": float(np.average(macro["committed_mw"], weights=hours)),
        "committed_mwh": float((macro["committed_mw"] * hours).sum()),
        "mean_paid_perf_score": float(np.average(macro["perf_score"], weights=hours)),
        "bid_acceptance_fraction": float(macro["bid_accepted"].mean()),
        "revenue_before_discount_usd": before_discount,
        "tracking_discount_usd": before_discount - float(macro["regulation_revenue"].sum()),
        "bess_discharge_mwh": float(hardware["bess_actual_kw"].clip(lower=0).sum() * dt_s / 3.6e6),
        "bess_charge_mwh": float(-hardware["bess_actual_kw"].clip(upper=0).sum() * dt_s / 3.6e6),
        "soc_min": float(hardware["bess_soc"].min()),
        "soc_final": float(hardware["bess_soc"].iloc[-1]),
        "temp_A_max": float(hardware["temp_A"].max()),
        "temp_B_max": float(hardware["temp_B"].max()),
        "thermal_warning_fraction": float(hardware["thermal_warning"].mean()),
        "regd_clipping_fraction": float(hardware["regd_observed"].abs().ge(1.0).mean()),
        "commitment_observation_mae_mw": float(
            (hardware["committed_actual_mw"] - hardware["committed_observed_mw"]).abs().mean()
        ),
    }


def _lmp(load_mw, a) -> np.ndarray:
    excess_gw = np.maximum(0.0, load_mw - a["load_median"]) / 1000.0
    return np.clip(a["lmp_base"] + a["lmp_slope"] * excess_gw, 0.0, 500.0)


def _legacy_metrics(A: pd.DataFrame, B: pd.DataFrame) -> dict[str, float]:
    agent_ticks, baseline_ticks = len(A), len(B)
    n = min(agent_ticks, baseline_ticks)
    A, B, at = A.iloc[:n], B.iloc[:n], A.attrs
    L = A["load_mw"].to_numpy()
    a, b = A["p_fac_mw"].to_numpy(), B["p_fac_mw"].to_numpy()
    dP = b - a
    dt_h = at["dt_s"] / 3600.0
    S = L >= at.get("stress_threshold_mw", np.quantile(L, STRESS_QUANTILE))
    w = np.maximum(0.0, L - at["load_median"])
    dlmp = _lmp(L + b, at) - _lmp(L + a, at)
    f = A["f_dev_hz"].to_numpy()
    fb = B["f_dev_hz"].to_numpy()
    act, dem, err = A["dp_act_kw"].to_numpy(), A["dp_dem_kw"].to_numpy(), A["err_kw"].to_numpy()
    lw = (w * dP).sum() / w.sum() if w.sum() > 0 else float("nan")
    return {
        "agent_ticks": agent_ticks,
        "baseline_ticks": baseline_ticks,
        "paired_ticks": n,
        "dP_mean_mw": dP.mean(),
        "dP_stress_mw": dP[S].mean(),
        "LFR_pct": 100 * dP[S].sum() / b[S].sum(),
        "SPDR_pct": 100 * dP[S].mean() / np.max(L + b),
        "LWSR_pct": 100 * (w * dP).sum() / (w * (L + b)).sum() if w.sum() > 0 else float("nan"),
        "LWCF_pct": 100 * lw / at["c_max"],
        "corr_P_L": float(np.corrcoef(a, L)[0, 1]),
        "corr_P_L_base": float(np.corrcoef(b, L)[0, 1]),
        "dLMP_stress_usd_mwh": dlmp[S].mean(),
        "avoided_cost_usd": float((dlmp * L * dt_h).sum()),
        "f_rms_mhz": 1e3 * np.sqrt((f ** 2).mean()),
        "f_rms_base_mhz": 1e3 * np.sqrt((fb ** 2).mean()),
        "f_max_dev_mhz": 1e3 * np.abs(f).max(),
        "f_max_dev_base_mhz": 1e3 * np.abs(fb).max(),
        "f_outside_db_pct": 100 * (np.abs(f) > F_DEADBAND_HZ).mean(),
        "f_outside_db_base_pct": 100 * (np.abs(fb) > F_DEADBAND_HZ).mean(),
        "reg_energy_mwh": float(np.abs(act).sum() / 1e3 * dt_h),
        "reg_mileage_mw": float(np.abs(np.diff(act)).sum() / 1e3),
        "perf_ratio": 1 - err.sum() / max(np.abs(dem).sum(), 1e-9),
        "ramp_max_mw_min": float(np.abs(np.diff(a)).max() / (at["dt_s"] / 60.0)),
        "rebound_pct": 100 * np.clip(a - b, 0, None)[~S].sum() / max(np.clip(dP, 0, None)[S].sum(), 1e-9),
        "throughput_ratio": A["throttle"].mean(),
        "backlog_mean_mw": A["backlog_kw"].mean() / 1e3,
        "rmse_kw": float(np.sqrt((err ** 2).mean())),
        "revenue_usd": at["revenue_usd"],
        "revenue_base_usd": B.attrs["revenue_usd"],
        "dP_dem_stress_mw": dem[S].mean() / 1e3,
        "dP_act_stress_mw": act[S].mean() / 1e3,
    }


def _metrics(A: pd.DataFrame, B: pd.DataFrame) -> dict:
    mode = A.attrs.get("simulation_mode", "legacy_v0")
    if mode != B.attrs.get("simulation_mode", "legacy_v0"):
        raise ValueError("Cannot compare trajectories with different simulation semantics")
    expected = A.attrs.get("expected_ticks", len(A))
    if expected != B.attrs.get("expected_ticks", len(B)):
        raise ValueError("Paired trajectories must have the same planned horizon")
    n = min(len(A), len(B))
    qualified = len(A) == len(B) == expected and not (
        A.attrs.get("terminated", False) or B.attrs.get("terminated", False)
    )
    audit = {
        "agent_ticks": len(A), "baseline_ticks": len(B), "paired_ticks": n,
        "expected_ticks": expected, "qualified": qualified,
        "comparison_status": "complete" if qualified else "incomplete_or_terminated",
        "observed_revenue_usd": A.attrs["revenue_usd"],
    }
    if mode == "physical_v1":
        for key in ("timestamp", "load_mw", "regd_observed", "work_arrived_kwh"):
            if not np.array_equal(A[key].to_numpy()[:n], B[key].to_numpy()[:n]):
                raise ValueError(f"Paired exogenous trajectories differ: {key}")
        if not qualified:
            warnings.warn(
                f"Unqualified grid-relief comparison: agent {len(A)}, baseline {len(B)}, "
                f"planned {expected} ticks; no common-prefix performance score",
                RuntimeWarning, stacklevel=2,
            )
            unavailable = (*RELIEF_KEYS, "rmse_kw", "revenue_usd", "f_rms_mhz",
                           "f_rms_base_mhz", "dP_dem_stress_mw", "dP_act_stress_mw",
                           "throughput_ratio", "pcc_relief_stress_mw")
            return {**dict.fromkeys(unavailable, float("nan")), **audit}
    result = {**_legacy_metrics(A, B), **audit}
    if mode == "physical_v1":
        stress = A["load_mw"].to_numpy() >= A.attrs["stress_threshold_mw"]
        result.update(
            pcc_relief_stress_mw=float(
                (B["p_grid_import_mw"].to_numpy() - A["p_grid_import_mw"].to_numpy())[stress].mean()
            ),
            throughput_ratio=float(A["throughput_ratio"].mean()),
            work_arrived_kwh=float(A["work_arrived_kwh"].sum()),
            work_served_kwh=float(A["work_served_kwh"].sum()),
            backlog_final_kwh=float(A["backlog_work_kwh"].iloc[-1]),
            facility_relief_mwh=float(
                (B["p_fac_mw"] - A["p_fac_mw"]).sum() * A.attrs["dt_s"] / 3600
            ),
        )
    return result


def _aggregate_results(per_episode: pd.DataFrame, keys: list[str], *, strict: bool) -> pd.DataFrame:
    numeric = per_episode.select_dtypes(include=["number", "bool"]).columns.difference(keys + ["seed"])
    grouped = per_episode.groupby(keys, sort=False)
    aggregate = grouped[list(numeric)].agg(["mean", "std"])
    aggregate.columns = [f"{c}_std" if stat == "std" else c for c, stat in aggregate.columns]
    aggregate["attempts"] = grouped.size()
    aggregate["qualified_count"] = grouped["qualified"].sum()
    if strict:
        incomplete = aggregate["qualified_count"] != aggregate["attempts"]
        audit_columns = {"attempts", "qualified_count", "qualified", "qualified_std",
                         "expected_ticks", "agent_ticks", "baseline_ticks", "paired_ticks",
                         "expected_ticks_std", "agent_ticks_std", "baseline_ticks_std", "paired_ticks_std"}
        aggregate.loc[incomplete, [key for key in aggregate.columns if key not in audit_columns]] = np.nan
    return aggregate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", nargs="+", default=["rule_based", "pid", "bang_bang", "sac"],
                   help="Hardware controllers; prefix with 'sac_macro+' to use the SAC macro.")
    p.add_argument("--macro-model-dir", default="trained_models/BestCheckpoints/sac_macro_default_s100")
    p.add_argument("--hw-model-dir", default="trained_models/BestCheckpoints/sac_default_s100")
    p.add_argument("--scenarios", nargs="+", default=["default"])
    p.add_argument("--markets", nargs="+", default=["nyiso_nyc"],
                   help="Grid market presets; nyiso_nyc reproduces the shipped config.")
    p.add_argument("--start-days", nargs="+", default=["0"],
                   help="Calendar day-of-year of the load file to replay at real time; 'legacy' = shipped indexing.")
    p.add_argument("--regd-load-bias", nargs="+", type=float, default=[0.0],
                   help="Add beta * (day-ahead load anomaly) to RegD; requires calendar start days.")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--simulation-mode", choices=SIMULATION_MODES, default="physical_v1")
    p.add_argument("--allow-semantic-mismatch", action="store_true")
    p.add_argument("--output", default="evaluation/results/grid_relief")
    p.add_argument("--trace-dir", help="Save per-rollout hardware/macro CSVs and checkpoint identities.")
    args = p.parse_args()

    ctrls = ["none"] + args.hw
    seeds = [args.seed + i for i in range(args.n_episodes)]
    if args.n_episodes < 1 or args.workers < 1:
        p.error("--n_episodes and --workers must be positive")
    if args.simulation_mode == "physical_v1" and "legacy" in args.start_days:
        p.error("'legacy' start days require --simulation-mode legacy_v0")
    days = [None if d == "legacy" else int(d) for d in args.start_days]
    combos = [(sc, m, d, b) for sc in args.scenarios for m in args.markets
              for d in days for b in args.regd_load_bias]

    def _job(sc, m, d, b, h, s):
        macro, hw = h.split("+", 1) if "+" in h else ("rule_macro", h)
        trace_dir = None if args.trace_dir is None else str(
            Path(args.trace_dir) / f"{sc}_{m}_d{d}_b{b:g}_{h}_s{s}"
        )
        return (hw, sc, s, args.hw_model_dir, macro, args.macro_model_dir, cfgs[m], d, b,
                trace_dir, args.simulation_mode, args.allow_semantic_mismatch)

    with tempfile.TemporaryDirectory(prefix="c2g-grid-relief-") as directory:
        cfgs = {m: _market_config(m, directory, args.simulation_mode) for m in args.markets}
        jobs = {(*c, h, s): _job(*c, h, s) for c in combos for h in ctrls for s in seeds}
        print(f"Running {len(jobs)} episodes on {args.workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers)) as ex:
            futs = {k: ex.submit(_run_episode, *v) for k, v in jobs.items()}
            traces = {k: f.result() for k, f in futs.items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_ep = []
    for sc, m, d, b in combos:
        for h in args.hw:
            for s in seeds:
                per_ep.append({"scenario": sc, "market": m, "day": "legacy" if d is None else d,
                               "simulation_mode": args.simulation_mode,
                               "regd_bias": b, "hw": h, "seed": s,
                               **_metrics(traces[(sc, m, d, b, h, s)], traces[(sc, m, d, b, "none", s)]),
                               **traces[(sc, m, d, b, h, s)].attrs["diagnostics"]})
    per_ep_df = pd.DataFrame(per_ep)
    keys = ["scenario", "market", "day", "regd_bias", "hw"]
    agg = _aggregate_results(per_ep_df, keys, strict=args.simulation_mode == "physical_v1")

    from scipy.stats import t as student_t
    t_crit = float(student_t.ppf(0.95, len(seeds) - 1)) if len(seeds) > 1 else float("nan")
    for k in RELIEF_KEYS:
        t = agg[k] / (agg[f"{k}_std"] / np.sqrt(len(seeds)))
        agg[f"{k}_sig_pos"] = t > t_crit
    agg["f_rms_reduction_pct"] = 100 * (1 - agg["f_rms_mhz"] / agg["f_rms_base_mhz"])
    agg["n_relief_sig_pos"] = agg[[f"{k}_sig_pos" for k in RELIEF_KEYS]].sum(axis=1)

    tag = (args.scenarios[0] if len(args.scenarios) == 1 else "sweep") + f"_{args.simulation_mode}"
    if args.markets != ["nyiso_nyc"]:
        tag += "_markets"
    if args.start_days != ["legacy"]:
        tag += "_days" + "-".join(args.start_days)
    if args.regd_load_bias != [0.0]:
        tag += "_regdbias" + "-".join(f"{b:g}" for b in args.regd_load_bias)
    per_ep_df.to_csv(f"{out}_{tag}_per_episode.csv", index=False)
    agg.to_csv(f"{out}_{tag}.csv")
    meta = {"scenarios": args.scenarios, "markets": args.markets, "seeds": seeds,
            "start_days": args.start_days, "regd_load_bias": args.regd_load_bias,
            "regd_bias_definition": "r_t + beta * clip((L_t - mean_day) / (max_day - mean_day), -1, 1); "
                                    "physical_v1 applies before reset observation; legacy_v0 evaluation-only patch",
            "simulation_mode": args.simulation_mode,
            "allow_semantic_mismatch": args.allow_semantic_mismatch,
            "source_id": source_id(ROOT),
            "weather_alignment": (
                "calendar climatology from source year, not coincident historical weather"
                if args.simulation_mode == "physical_v1" else "historical row-index replay"
            ),
            "comparison_policy": "all attempts retained; physical_v1 incomplete groups have no performance estimate",
            "hw_model_dir": args.hw_model_dir, "macro_model_dir": args.macro_model_dir,
            "trace_dir": args.trace_dir,
            "diagnostic_fields": list(next(iter(traces.values())).attrs["diagnostics"]),
            "stress_quantile": STRESS_QUANTILE, "sig_test": f"one-sided t > {t_crit}",
            "baseline": "rule_macro, no inner controller",
            "market_override": "global.grid_market (scenario grid_market is ignored by the env)"}
    Path(f"{out}_{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    cols = ["attempts", "qualified_count", "dP_stress_mw", "dP_stress_mw_std", "LFR_pct", "SPDR_pct", "LWCF_pct", "n_relief_sig_pos",
            "dP_dem_stress_mw", "dP_act_stress_mw", "rmse_kw", "revenue_usd", "f_rms_reduction_pct",
            "throughput_ratio"]
    with pd.option_context("display.width", 250, "display.max_rows", 500,
                           "display.float_format", "{:.4g}".format):
        print(agg[cols])
    print(f"\nSaved -> {out}_{tag}.csv (+ _per_episode.csv, _meta.json)")


if __name__ == "__main__":
    main()
