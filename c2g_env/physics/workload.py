"""
Step 1.1 — Workload Orchestrator
=================================
Ingests all four Alibaba trace datasets and converts raw workload metrics
into facility-level IT power demand at 5-minute (300 s) timesteps.

Trace → Power mapping
---------------------
Traces are converted to rack-level GPU utilisation fractions, then to power via
the non-linear server model from DatacenterElectrical (duplicated here as
constants to keep simulators independent and avoid circular imports):

    P_server(u) = N_racks × [P_idle + (P_max - P_idle) × u^alpha]   (kW)

Zone assignment and flex/base classification
--------------------------------------------
+--------------+--------+--------+---------+----------------------------+
| Trace        | Zone   | Class  | Racks   | Notes                      |
+--------------+--------+--------+---------+----------------------------+
| batch_v2023  |   A    | P_flex |  1 200  | Schedulable; DVFS applies  |
| genai_v2026  |   A    | P_base |    800  | Rigid; SLA-protected       |
| dlrm_v2025   |   B    | P_base |  2 500  | Rigid inference serving    |
+--------------+--------+--------+---------+----------------------------+

spot_v2026 is excluded from this release (arrival-based trace requires a
separate queue simulation; reserved for a future extension).

Timestep
--------
All traces are at 5-minute (300 s) resolution which matches ThermalTwin and
BESSModel defaults.  The low-level RL env runs at dt=300 s for Phase 1.
References
----------
[1] Weng, Q., Xiao, W., Yu, Y., et al. (2022) "MLaaS in the Wild: Workload
    Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters,"
    USENIX NSDI 2022.
    https://www.usenix.org/conference/nsdi22/presentation/weng
    — Alibaba production GPU trace (batch_v2023 & genai_v2026 modelled here).
[2] Guo, J., Chang, Z., Wang, S., et al. (2019) "Who Limits the Resource
    Efficiency of My Datacenter: An Analysis of Alibaba Datacenter Traces,"
    ACM IWQoS 2019.  DOI: 10.1145/3326285.3329074
    — GPU utilization distributions and flex/rigid workload calibration.
[3] Fan, X., Weber, W., Barroso, L.A. (2007) "Power Provisioning for a
    Warehouse-sized Computer," ACM ISCA 2007, pp. 13–23.
    DOI: 10.1145/1273440.1250665
    — P_server(u) = P_idle + (P_max-P_idle)×u^α; α=1.4 GPU superlinear.
[4] Wierman, A., Liu, Z., Liu, I., Mohsenian-Rad, H. (2014) "Opportunities
    and Challenges for Data Center Demand Response," IEEE IGCC 2014.
    DOI: 10.1109/IGCC.2014.7039172
    — deferrable (P_flex) vs. rigid (P_base) workload classification.
[5] Narayanan, D., Shoeybi, M., Casper, J., et al. (2021) "Efficient
    Large-Scale Language Model Training on GPU Clusters Using Megatron-LM,"
    SC 2021.  DOI: 10.1145/3458817.3476209
    — GPU cluster power (rack-level P_max, burst) for genai_v2026 trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Server power model constants — mirrors DatacenterElectrical to avoid
# circular imports.  Keep in sync with electrical.py if parameters change.
# ---------------------------------------------------------------------------
_ZONE_A_N_RACKS_FLEX = 1_200         # Batch: 60% of Zone A (2000 racks)
_ZONE_A_N_RACKS_BASE = 800           # GenAI: 40% of Zone A
_ZONE_B_N_RACKS      = 2_500         # DLRM:  all of Zone B
_P_IDLE_A_KW   = 8.0                 # Idle power per rack, Zone A (kW)
_P_MAX_A_KW    = 75.0                # Peak power per rack, Zone A (kW)
_ALPHA_A       = 1.4                 # GPU superlinear exponent
_P_IDLE_B_KW   = 4.0                 # Idle power per rack, Zone B (kW)
_P_MAX_B_KW    = 40.0                # Peak power per rack, Zone B (kW)
_ALPHA_B       = 1.2                 # CPU/inference exponent

# Normalisation denominators — 99th-percentile values from trace stats to
# cap utilisation at 1.0 while preserving the shape of heavy-tail bursts.
_BATCH_GPU_MILLI_MAX = 12_620.0      # batch: max observed gpu_milli_request
_DLRM_GPU_MAX        = 227.0         # dlrm:  max observed active_gpu_count
_GENAI_DUTY_MAX      = 100.0         # genai: duty_cycle already in [0, 100]
_GENAI_SPIKE_PCT75   = 12.19         # genai: P75 duty cycle → spike threshold


def _rack_power_kw(
    util: np.ndarray | float,
    n_racks: int,
    p_idle_kw: float,
    p_max_kw: float,
    alpha: float,
) -> np.ndarray | float:
    """Non-linear server power model (vectorised)."""
    u = np.clip(util, 0.0, 1.0)
    return n_racks * (p_idle_kw + (p_max_kw - p_idle_kw) * u ** alpha)


@dataclass(frozen=True)
class WorkloadState:
    """IT power snapshot at a single 300 s timestep."""
    p_base_kw:      float   # Rigid IT load: GenAI (Zone A) + DLRM (Zone B)  [kW]
    p_base_a_kw:    float   # Zone A rigid load: GenAI inference racks         [kW]
    p_base_b_kw:    float   # Zone B rigid load: DLRM serving racks            [kW]
    p_flex_nom_kw:  float   # Schedulable batch arrivals this tick             [kW]
    p_flex_kw:      float   # Batch load actually served (from queue)          [kW]
    p_total_it_kw:  float   # p_base + p_flex                                  [kW]
    throttle:       float   # DVFS factor applied                           [0, 1]
    is_spike_active: bool   # True when GenAI duty cycle > P75 threshold
    tick:           int     # Current trace tick index
    backlog_kw:     float   # Deferred batch work remaining in queue           [kW]
    avg_delay_steps: float  # Little's Law average delay (steps)
    work_arrived_kwh: float = 0.0
    work_served_kwh: float = 0.0
    backlog_work_kwh: float = 0.0


class WorkloadOrchestrator:
    """
    Manages IT power demand for the C2G-Bench 250 MW hyperscale facility.

    Fuses three Alibaba trace datasets (batch 2023, DLRM 2025, GenAI 2026)
    into synchronised P_base and P_flex time-series at 300 s resolution.

    Parameters
    ----------
    trace_dir:
        Directory containing the preprocessed trace CSVs.  Must contain:
        ``batch_v2023.csv``, ``dlrm_v2025.csv``, ``genai_v2026.csv``.
        Defaults to ``data/processed/workload_traces`` relative to cwd.
    dt_seconds:
        Simulation timestep in seconds.  Must match trace resolution (300 s).
    seed:
        Random seed for stochastic fallback (used only if trace files are
        unavailable).
    n_racks_a_flex, n_racks_a_base, n_racks_b:
        Optional rack-count overrides for re-instantiating the facility at a
        different nameplate capacity.  Traces are capacity-agnostic utilisation
        profiles, so only the rack counts (and per-rack power) set absolute MW.
        Defaults to the 250 MW module constants when None.
    """

    _TRACE_FILES = {
        "batch": "batch_v2023.csv",
        "dlrm":  "dlrm_v2025.csv",
        "genai": "genai_v2026.csv",
    }

    def __init__(
        self,
        trace_dir: str | Path = "data/processed/workload_traces",
        dt_seconds: float = 300.0,
        seed: int = 42,
        n_racks_a_flex: int | None = None,
        n_racks_a_base: int | None = None,
        n_racks_b: int | None = None,
        physical_accounting: bool = False,
        start_day: int = 0,
    ) -> None:
        self.dt = dt_seconds
        self.physical_accounting = physical_accounting
        self._start_tick = int(start_day * 86400 / dt_seconds)
        self._rng = np.random.default_rng(seed)
        self._n_racks_a_flex = int(n_racks_a_flex) if n_racks_a_flex is not None else _ZONE_A_N_RACKS_FLEX
        self._n_racks_a_base = int(n_racks_a_base) if n_racks_a_base is not None else _ZONE_A_N_RACKS_BASE
        self._n_racks_b      = int(n_racks_b)      if n_racks_b      is not None else _ZONE_B_N_RACKS
        self._tick: int = self._start_tick
        # Traces are at 300-s resolution; this many env steps share each tick.
        self._steps_per_trace_tick: int = max(1, round(300.0 / dt_seconds))

        # Queue state — initialised here, reset in reset()
        self._backlog_kw:           float = 0.0  # unsatisfied deferred work  [kW]
        self._delay_accum_kw_steps: float = 0.0  # Little's Law numerator
        self._total_served_kw:      float = 0.0  # denominator for avg delay

        trace_dir = Path(trace_dir)
        traces = self._load_traces(trace_dir)

        # Build power series aligned to the shortest trace (dlrm = 8640 ticks)
        n = min(len(traces["batch"]), len(traces["dlrm"]))

        batch = traces["batch"].reindex(range(n), fill_value=0)
        dlrm  = traces["dlrm"].reindex(range(n),  fill_value=traces["dlrm"].iloc[-1])

        # GenAI has only 288 ticks (1 day) → tile across full horizon
        genai_raw = traces["genai"]
        repeats   = int(np.ceil(n / len(genai_raw)))
        genai     = pd.concat([genai_raw] * repeats, ignore_index=True).iloc[:n]

        # --- Utilisation fractions ----------------------------------------
        util_batch = (batch["gpu_milli_request"].values / _BATCH_GPU_MILLI_MAX
                     ).clip(0.0, 1.0)
        util_dlrm  = (dlrm["active_gpu_count"].values  / _DLRM_GPU_MAX
                     ).clip(0.0, 1.0)
        util_genai = (genai["avg_gpu_duty_cycle"].values / _GENAI_DUTY_MAX
                     ).clip(0.0, 1.0)

        # --- Convert to power (kW) ----------------------------------------
        self._p_flex_arr   = _rack_power_kw(util_batch, self._n_racks_a_flex,
                                             _P_IDLE_A_KW, _P_MAX_A_KW, _ALPHA_A)
        self._p_base_a_arr = _rack_power_kw(util_genai, self._n_racks_a_base,
                                             _P_IDLE_A_KW, _P_MAX_A_KW, _ALPHA_A)
        self._p_base_b_arr = _rack_power_kw(util_dlrm,  self._n_racks_b,
                                             _P_IDLE_B_KW, _P_MAX_B_KW, _ALPHA_B)
        self._p_base_arr   = self._p_base_a_arr + self._p_base_b_arr

        # Spike flag: GenAI duty cycle above P75
        self._spike_arr = (genai["avg_gpu_duty_cycle"].values > _GENAI_SPIKE_PCT75)

        self._n = n
        # Hardware ceiling — precomputed once
        self._p_flex_max_kw: float = float(_rack_power_kw(
            1.0, self._n_racks_a_flex, _P_IDLE_A_KW, _P_MAX_A_KW, _ALPHA_A
        ))
        self.p_flex_idle_kw = self._n_racks_a_flex * _P_IDLE_A_KW
        self.work_capacity_kw = self._p_flex_max_kw - self.p_flex_idle_kw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, throttle_batch: float) -> WorkloadState:
        """
        Return the IT power state for the current tick and advance by one step.

        Queue model
        -----------
        Batch work that cannot be served is not dropped — it accumulates in
        ``_backlog_kw`` and is served in future steps when capacity allows.
        Service capacity per step = p_flex_max_kw × throttle_batch.

        Average delay is estimated via Little's Law:
            avg_delay_steps = Σ Q(t) / total_served_kw

        Parameters
        ----------
        throttle_batch:
            DVFS factor for schedulable batch jobs in [0, 1].
            1.0 = full capacity; 0.0 = fully suspended (all work deferred).
        """
        throttle_batch = float(np.clip(throttle_batch, 0.0, 1.0))
        # Trace is at 300-s resolution; hold constant across sub-ticks.
        trace_tick = self._tick // self._steps_per_trace_tick
        idx = trace_tick % self._n

        p_base_a = float(self._p_base_a_arr[idx])
        p_base_b = float(self._p_base_b_arr[idx])
        p_base   = p_base_a + p_base_b
        nominal_kw = float(self._p_flex_arr[idx])
        idle_kw = self.p_flex_idle_kw if self.physical_accounting else 0.0
        arrived = max(0.0, nominal_kw - idle_kw)
        is_spike = bool(self._spike_arr[idx])

        # --- Queue model -------------------------------------------------
        # Accumulate pre-service backlog for Little's Law numerator
        self._delay_accum_kw_steps += self._backlog_kw

        # Enqueue new arrivals
        self._backlog_kw += arrived

        # Serve up to hardware capacity (DVFS throttle limits throughput)
        capacity_kw = (self._p_flex_max_kw - idle_kw) * throttle_batch
        served_kw   = min(self._backlog_kw, capacity_kw)
        self._backlog_kw  -= served_kw
        self._total_served_kw += served_kw

        # Little's Law: avg_delay (steps) = Σ Q(t) / total_served_kw
        avg_delay_steps = (
            self._delay_accum_kw_steps / self._total_served_kw
            if self._total_served_kw > 0.0 else 0.0
        )

        self._tick += 1
        return WorkloadState(
            p_base_kw       = p_base,
            p_base_a_kw     = p_base_a,
            p_base_b_kw     = p_base_b,
            p_flex_nom_kw   = nominal_kw,
            p_flex_kw       = idle_kw + served_kw,
            p_total_it_kw   = p_base + idle_kw + served_kw,
            throttle        = throttle_batch,
            is_spike_active = is_spike,
            tick            = idx,
            backlog_kw      = self._backlog_kw,
            avg_delay_steps = avg_delay_steps,
            work_arrived_kwh = arrived * self.dt / 3600.0,
            work_served_kwh = served_kw * self.dt / 3600.0,
            backlog_work_kwh = self._backlog_kw * self.dt / 3600.0,
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset trace pointer and queue state to the beginning."""
        self._tick = self._start_tick
        self._backlog_kw           = 0.0
        self._delay_accum_kw_steps = 0.0
        self._total_served_kw      = 0.0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @property
    def horizon_ticks(self) -> int:
        """Total number of ticks before the trace loops."""
        return self._n

    @property
    def p_flex_max_kw(self) -> float:
        """Theoretical maximum flexible load (all batch racks at peak)."""
        return self._p_flex_max_kw

    @property
    def p_base_range_kw(self) -> tuple[float, float]:
        """(min, max) of P_base across the full trace horizon."""
        return float(self._p_base_arr.min()), float(self._p_base_arr.max())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_traces(self, trace_dir: Path) -> dict[str, pd.DataFrame]:
        """Load and validate the three trace CSVs."""
        required_cols = {
            "batch": {"tick", "gpu_milli_request"},
            "dlrm":  {"tick", "active_gpu_count"},
            "genai": {"tick", "avg_gpu_duty_cycle"},
        }
        frames: dict[str, pd.DataFrame] = {}
        for key, fname in self._TRACE_FILES.items():
            path = trace_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Trace file not found: {path}\n"
                    f"Run preprocessing/workload_traces/process_{key}*.py first."
                )
            df = pd.read_csv(path).set_index("tick").sort_index()
            missing = required_cols[key] - (set(df.columns) | {"tick"})
            if missing:
                raise ValueError(f"{fname} missing columns: {missing}")
            frames[key] = df
        return frames
