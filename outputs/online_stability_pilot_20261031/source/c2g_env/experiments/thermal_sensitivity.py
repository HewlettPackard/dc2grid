"""
c2g_env/experiments/thermal_sensitivity.py
==========================================
Thermal-parameter sensitivity harness for reviewer Ask 1
("are the safety results an artifact of permissive thermal assumptions?").

Two complementary instruments, both driving the *real* ``ThermalTwin`` plant so
there is no reliance on a surrogate model:

  (#1) Open-loop plant characterization — no controller in the loop. Runs the
       true plant under fixed cooling policies (worst-case / nominal / maximal)
       at worst-case IT load and reports where the temperature settles relative
       to the FIXED safety line T_safe = 35 degC, and how long it takes to cross
       it. This answers the permissiveness question with zero controller
       anchoring.

  (#3) Closed-loop analytic controllers — the frozen bang-bang / PID /
       rule-based hardware controllers (baselines/) run against each perturbed
       plant. Their internal danger references stay fixed at T_warn = 33 /
       T_safe = 35 (never re-tuned), so this measures how close realistic
       feedback gets to the open-loop safety floor.

The safety anchors T_warn = 33 and T_safe = 35 are FIXED invariants and are
never swept. Only the uncalibrated PLANT parameters are swept, read from
``conf/experiments.yaml`` (single source of truth).

Usage
-----
  # from the repo root, with the venv active
  python -m c2g_env.experiments.thermal_sensitivity
  python -m c2g_env.experiments.thermal_sensitivity --horizon-ticks 4320
  python -m c2g_env.experiments.thermal_sensitivity \
      --controllers pid rule_based --closed-loop-load adversarial \
      --out copilot/artifacts/thermal_sensitivity_custom.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from c2g_env.obs_indices import Fast as _F
from c2g_env.physics.thermal import ThermalTwin
from c2g_env.physics.workload import WorkloadOrchestrator
from c2g_env.thermal_limits import T_SAFE, T_WARN
from baselines.bang_bang import BangBangController
from baselines.pid_controller import PIDController
from baselines.rule_based_mpc import RuleBasedController

# --- Fixed safety anchors (physical scoring rubric — never swept) ------------
# T_SAFE and T_WARN are the single source of truth from config.yaml.

# ThermalTwin attributes that the sweep is permitted to override.
_ALLOWED_KEYS: frozenset[str] = frozenset({
    "C_A", "C_B", "K_liq", "K_air", "K_env_A", "K_env_B",
    "T_supply_A", "T_supply_B", "COP_base", "fault_factor", "T_amb",
})

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "conf" / "experiments.yaml"
_DEFAULT_TRACE_DIR = _REPO_ROOT / "data" / "processed" / "workload_traces"
_SCENARIO_DIR = _REPO_ROOT / "conf" / "scenario"

_CONTROLLER_FACTORIES: dict[str, Callable[[], Any]] = {
    "bang_bang": BangBangController,
    "pid": PIDController,
    "rule_based": RuleBasedController,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read the ``thermal_sensitivity`` block from conf/experiments.yaml."""
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        full = yaml.safe_load(fh)
    if "thermal_sensitivity" not in full:
        raise KeyError(f"'thermal_sensitivity' block not found in {path}")
    return full["thermal_sensitivity"]


def build_configurations(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand nominal, OAT sweep, and named coupled cases into flat configs.

    Each configuration is a dict with:
      ``id``          — unique row label
    ``param``       — the swept parameter name ('nominal'/'coupled' for cases)
    ``value``       — the swept value (None for nominal/coupled cases)
      ``overrides``   — full ThermalTwin attribute override dict
    """
    nominal: dict[str, float] = dict(cfg["nominal"])
    _validate_keys(nominal, where="nominal")

    configs: list[dict[str, Any]] = [
        {"id": "nominal", "param": "nominal", "value": None, "overrides": dict(nominal)}
    ]

    for param, values in cfg.get("sweep", {}).items():
        if param not in _ALLOWED_KEYS:
            raise KeyError(f"sweep key '{param}' is not an overridable ThermalTwin attribute")
        for val in values:
            overrides = dict(nominal)
            overrides[param] = float(val)
            configs.append({
                "id": f"{param}={val}",
                "param": param,
                "value": float(val),
                "overrides": overrides,
            })

    for case_id, case_overrides in cfg.get("coupled_cases", {}).items():
        _validate_keys(case_overrides, where=f"coupled_cases.{case_id}")
        overrides = dict(nominal)
        overrides.update({k: float(v) for k, v in case_overrides.items()})
        configs.append({
            "id": str(case_id),
            "param": "coupled",
            "value": None,
            "overrides": overrides,
        })

    return configs


def _validate_keys(d: dict[str, Any], where: str) -> None:
    bad = set(d) - _ALLOWED_KEYS
    if bad:
        raise KeyError(f"Unknown ThermalTwin override key(s) in '{where}': {sorted(bad)}")


# ---------------------------------------------------------------------------
# Plant
# ---------------------------------------------------------------------------
def build_plant(overrides: dict[str, float], dt_seconds: float) -> ThermalTwin:
    """Construct a ThermalTwin with the given attribute overrides applied.

    Safety anchors (T_safe / T_warn) are left at their fixed defaults. The
    zone temperatures are warm-started to T_amb + 5 degC, matching the reset
    warm-start used by C2GFastEnv.
    """
    plant = ThermalTwin(dt_seconds=dt_seconds)
    for key, val in overrides.items():
        setattr(plant, key, float(val))
    # fault_factor override implies an active fault when < 1.0.
    plant.fault_active = bool(plant.fault_factor < 1.0)
    # Warm-start temperatures at the scenario ambient (env convention).
    warm = min(plant.T_amb + 5.0, plant.T_safe - 1.0)
    plant.temp_A = warm
    plant.temp_B = warm
    return plant


# ---------------------------------------------------------------------------
# IT-load drivers  ->  (p_it_A_mw, p_it_B_mw) given (tick, throttle)
# ---------------------------------------------------------------------------
def make_load_driver(
    kind: str, cfg: dict[str, Any], dt_seconds: float, seed: int
) -> Callable[[int, float], tuple[float, float]]:
    """Return a callable yielding zone IT powers (MW) per tick.

    ``adversarial`` — constant worst-case nameplate load; ignores throttle
                      (no shed authority; the hardest safety load).
    ``workload``    — real Alibaba traces via WorkloadOrchestrator; throttle
                      reduces served Zone A flex load (genuine shed lever).
    """
    run = cfg["run"]
    if kind == "adversarial":
        p_a = float(run.get("adversarial_p_it_A_mw", 150.0))
        p_b = float(run.get("adversarial_p_it_B_mw", 100.0))

        def _adversarial(_tick: int, _throttle: float) -> tuple[float, float]:
            return p_a, p_b

        return _adversarial

    if kind == "workload":
        wl = WorkloadOrchestrator(
            trace_dir=str(_DEFAULT_TRACE_DIR), dt_seconds=dt_seconds, seed=seed
        )
        wl.reset(seed=seed)

        def _workload(_tick: int, throttle: float) -> tuple[float, float]:
            w = wl.step(throttle)
            p_it_A = (w.p_base_a_kw + w.p_flex_kw) / 1_000.0
            p_it_B = w.p_base_b_kw / 1_000.0
            return p_it_A, p_it_B

        return _workload

    raise ValueError(f"Unknown load driver: {kind!r}")


# ---------------------------------------------------------------------------
# Observation builder (18-D Fast layout) for the closed-loop controllers
# ---------------------------------------------------------------------------
def _build_obs(temp_A: float, temp_B: float, prev_throttle: float,
               prev_pump: float, t_amb: float) -> np.ndarray:
    """Minimal Fast-layout obs. Non-thermal fields are set to benign values;
    regd = 0 keeps the BESS branch idle so the controllers act as pure thermal
    regulators (isolating the cooling response)."""
    obs = np.zeros(_F.DIM, dtype=np.float32)
    obs[_F.TEMP_A]   = temp_A / T_SAFE
    obs[_F.TEMP_B]   = temp_B / T_SAFE
    obs[_F.SOC]      = 0.5
    obs[_F.P_BASE]   = 0.3
    obs[_F.P_FLEX]   = 0.2
    obs[_F.P_FAC]    = 0.5
    obs[_F.REGD]     = 0.0
    obs[_F.LMP]      = 0.3
    obs[_F.GRID_LOAD] = 0.5
    obs[_F.IS_SPIKE] = 0.0
    obs[_F.PREV_THR] = prev_throttle
    obs[_F.PREV_PMP] = prev_pump
    obs[_F.PUE]      = 0.5
    obs[_F.T_AMB]    = float(np.clip((t_amb + 20.0) / 65.0, 0.0, 1.0))
    obs[_F.FREQ_DEV] = 0.0
    obs[_F.VPCC]     = 1.0
    obs[_F.BACKLOG]  = 0.0
    obs[_F.COMMITTED] = 0.0
    return obs


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def _fresh_metrics() -> dict[str, Any]:
    return {
        "T_A_max": -np.inf, "T_B_max": -np.inf,
        "viol_ticks": 0, "fail": False, "time_to_tsafe_s": None,
        "termination_zone": None,
        "pump_sum": 0.0, "hvac_sum": 0.0, "throttle_sum": 0.0, "n": 0,
    }


def _accumulate(m: dict[str, Any], temp_A: float, temp_B: float,
                pump: float, hvac: float, throttle: float,
                tick: int, dt: float) -> None:
    m["T_A_max"] = max(m["T_A_max"], temp_A)
    m["T_B_max"] = max(m["T_B_max"], temp_B)
    t_hot = max(temp_A, temp_B)
    if t_hot > T_WARN:
        m["viol_ticks"] += 1
    if t_hot > T_SAFE:
        m["fail"] = True
        if m["time_to_tsafe_s"] is None:
            m["time_to_tsafe_s"] = tick * dt
            hot_A = temp_A > T_SAFE
            hot_B = temp_B > T_SAFE
            m["termination_zone"] = "A+B" if hot_A and hot_B else ("A" if hot_A else "B")
    m["pump_sum"] += pump
    m["hvac_sum"] += hvac
    m["throttle_sum"] += throttle
    m["n"] += 1


def _finalize(m: dict[str, Any], temp_A: float, temp_B: float) -> dict[str, Any]:
    n = max(m["n"], 1)
    return {
        "T_A_final": round(temp_A, 3),
        "T_B_final": round(temp_B, 3),
        "T_A_max": round(m["T_A_max"], 3),
        "T_B_max": round(m["T_B_max"], 3),
        "headroom_A": round(T_SAFE - m["T_A_max"], 3),
        "headroom_B": round(T_SAFE - m["T_B_max"], 3),
        "viol_rate": round(m["viol_ticks"] / n, 4),
        "crossed_tsafe": int(m["fail"]),
        "termination_reason": "thermal" if m["fail"] else None,
        "termination_zone": m["termination_zone"],
        "time_to_tsafe_s": (None if m["time_to_tsafe_s"] is None
                            else round(m["time_to_tsafe_s"], 1)),
        "mean_pump": round(m["pump_sum"] / n, 3),
        "mean_hvac": round(m["hvac_sum"] / n, 3),
        "mean_throttle": round(m["throttle_sum"] / n, 3),
    }


def run_open_loop(plant: ThermalTwin, driver: Callable[[int, float], tuple[float, float]],
                  pump: float, hvac: float, throttle: float,
                  horizon: int, dt: float) -> dict[str, Any]:
    """#1 — fixed cooling policy, no controller."""
    m = _fresh_metrics()
    temp_A, temp_B = plant.temp_A, plant.temp_B
    for tick in range(horizon):
        p_a, p_b = driver(tick, throttle)
        (temp_A, temp_B), _ = plant.step(p_a, p_b, hvac_effort=hvac, pump_speed=pump)
        _accumulate(m, temp_A, temp_B, pump, hvac, throttle, tick, dt)
    return _finalize(m, temp_A, temp_B)


def run_closed_loop(plant: ThermalTwin, controller: Any,
                    driver: Callable[[int, float], tuple[float, float]],
                    horizon: int, dt: float) -> dict[str, Any]:
    """#3 — frozen analytic controller in the loop."""
    if hasattr(controller, "reset"):
        controller.reset()
    m = _fresh_metrics()
    temp_A, temp_B = plant.temp_A, plant.temp_B
    prev_throttle, prev_pump = 1.0, 0.7
    for tick in range(horizon):
        obs = _build_obs(temp_A, temp_B, prev_throttle, prev_pump, plant.T_amb)
        action, _ = controller.predict(obs)
        throttle = float(np.clip(action[0], 0.0, 1.0))
        pump = float(np.clip(action[1], 0.0, 1.0))
        hvac = float(np.clip(action[2], 0.0, 1.0))
        p_a, p_b = driver(tick, throttle)
        (temp_A, temp_B), _ = plant.step(p_a, p_b, hvac_effort=hvac, pump_speed=pump)
        _accumulate(m, temp_A, temp_B, pump, hvac, throttle, tick, dt)
        prev_throttle, prev_pump = throttle, pump
    return _finalize(m, temp_A, temp_B)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_sweep(cfg: dict[str, Any], horizon: int | None = None,
              controllers: list[str] | None = None,
              closed_loop_load: str | None = None) -> list[dict[str, Any]]:
    run = cfg["run"]
    dt = float(run["dt_seconds"])
    seed = int(run["seed"])
    horizon = int(horizon if horizon is not None else run["horizon_ticks"])
    open_load = str(run.get("open_loop_load", "adversarial"))
    closed_load = str(closed_loop_load if closed_loop_load is not None
                      else run.get("closed_loop_load", "workload"))
    ctrl_names = controllers if controllers is not None else list(run["controllers"])

    configs = build_configurations(cfg)
    rows: list[dict[str, Any]] = []

    # Fixed open-loop cooling policies (throttle held at full load = 1.0).
    open_policies = {
        "open_worstcool": (ThermalTwin.PUMP_MIN, 0.0),
        "open_nominal":   (0.7, 0.7),
        "open_maxcool":   (1.0, 1.0),
    }

    for c in configs:
        # -- #1 open-loop probes (worst-case load) -----------------------
        for run_type, (pump, hvac) in open_policies.items():
            plant = build_plant(c["overrides"], dt)
            driver = make_load_driver(open_load, cfg, dt, seed)
            res = run_open_loop(plant, driver, pump, hvac, 1.0, horizon, dt)
            rows.append(_row(c, run_type, open_load, res))

        # -- #3 closed-loop frozen controllers ---------------------------
        for name in ctrl_names:
            plant = build_plant(c["overrides"], dt)
            driver = make_load_driver(closed_load, cfg, dt, seed)
            controller = _CONTROLLER_FACTORIES[name]()
            res = run_closed_loop(plant, controller, driver, horizon, dt)
            rows.append(_row(c, f"closed_{name}", closed_load, res))

    return rows


def _row(c: dict[str, Any], run_type: str, load: str,
         res: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": c["id"], "param": c["param"], "value": c["value"],
        "run_type": run_type, "load_driver": load, **res,
    }


_CSV_FIELDS = [
    "config_id", "param", "value", "run_type", "load_driver",
    "T_A_final", "T_B_final", "T_A_max", "T_B_max",
    "headroom_A", "headroom_B", "viol_rate", "crossed_tsafe",
    "termination_reason", "termination_zone", "time_to_tsafe_s",
    "mean_pump", "mean_hvac", "mean_throttle",
]


def write_csv(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="Path to experiments.yaml (default: conf/experiments.yaml)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: copilot/artifacts/thermal_sensitivity_<ts>.csv)")
    parser.add_argument("--horizon-ticks", type=int, default=None,
                        help="Override run horizon in ticks (default from config)")
    parser.add_argument("--controllers", nargs="+", default=None,
                        choices=sorted(_CONTROLLER_FACTORIES),
                        help="Subset of closed-loop controllers to run")
    parser.add_argument("--closed-loop-load", default=None,
                        choices=["workload", "adversarial"],
                        help="Load driver for closed-loop runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()
    rows = run_sweep(cfg, horizon=args.horizon_ticks,
                     controllers=args.controllers,
                     closed_loop_load=args.closed_loop_load)

    if args.out is not None:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = _REPO_ROOT / "copilot" / "artifacts" / f"thermal_sensitivity_{ts}.csv"
    write_csv(rows, out_path)

    elapsed = time.time() - t0
    n_fail = sum(int(r["crossed_tsafe"]) for r in rows if r["run_type"] == "open_maxcool")
    print(f"[thermal_sensitivity] {len(rows)} rows in {elapsed:.1f}s -> {out_path}")
    print(f"[thermal_sensitivity] configs where even max cooling crosses T_safe: {n_fail}")


if __name__ == "__main__":
    main()
