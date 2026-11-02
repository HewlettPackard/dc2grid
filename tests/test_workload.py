"""
Tests for c2g_env/simulators/workload.py
=========================================
Covers:
  - Normal operation: trace loading, step output types and ranges
  - DVFS throttle: full, zero, and mid-point; effect on p_flex
  - P_base immutability under throttle changes
  - Spike detection: at least some ticks are marked as spike
  - Trace looping: tick resets to 0 after horizon
  - Reset: tick pointer and RNG reset
  - Power conservation: p_total = p_base + p_flex
  - Edge cases: throttle clamping below 0 and above 1
  - horizon_ticks and p_flex_max_kw / p_base_range_kw properties
  - Monotonicity: p_flex scales linearly with throttle
  - Missing trace directory raises FileNotFoundError
  - Missing required columns raises ValueError
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.physics.workload import WorkloadOrchestrator, WorkloadState

TRACE_DIR = Path("data/processed/workload_traces")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def orch():
    return WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=0)


@pytest.fixture
def fresh_orch():
    return WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=42)


# ---------------------------------------------------------------------------
# 1. Instantiation and properties
# ---------------------------------------------------------------------------

def test_instantiation(orch):
    assert orch.horizon_ticks == 8640   # 30-day trace at 5-min = 8640 ticks


def test_p_flex_max_kw_positive(orch):
    assert orch.p_flex_max_kw > 0.0


def test_p_flex_max_kw_magnitude(orch):
    # 1200 racks × 75 kW = 90 MW at u=1; idle floor = 1200×8 = 9.6 MW
    assert 9_600.0 <= orch.p_flex_max_kw <= 90_000.0


def test_p_base_range_ordered(orch):
    lo, hi = orch.p_base_range_kw
    assert lo < hi
    assert lo > 0.0


def test_p_base_range_magnitude(orch):
    # 250 MW facility: base should be between 10 MW (idle) and 210 MW (full)
    lo, hi = orch.p_base_range_kw
    assert lo >= 5_000.0
    assert hi <= 210_000.0


# ---------------------------------------------------------------------------
# 2. Step return type and field presence
# ---------------------------------------------------------------------------

def test_step_returns_workload_state(fresh_orch):
    s = fresh_orch.step(1.0)
    assert isinstance(s, WorkloadState)


def test_step_fields_present(fresh_orch):
    s = fresh_orch.step(1.0)
    assert hasattr(s, "p_base_kw")
    assert hasattr(s, "p_flex_nom_kw")
    assert hasattr(s, "p_flex_kw")
    assert hasattr(s, "p_total_it_kw")
    assert hasattr(s, "throttle")
    assert hasattr(s, "is_spike_active")
    assert hasattr(s, "tick")
    assert hasattr(s, "backlog_kw")
    assert hasattr(s, "avg_delay_steps")


# ---------------------------------------------------------------------------
# 3. Value ranges
# ---------------------------------------------------------------------------

def test_p_base_non_negative(fresh_orch):
    for _ in range(100):
        s = fresh_orch.step(1.0)
        assert s.p_base_kw >= 0.0


def test_p_flex_nom_non_negative(fresh_orch):
    for _ in range(100):
        s = fresh_orch.step(1.0)
        assert s.p_flex_nom_kw >= 0.0


def test_p_total_non_negative(fresh_orch):
    for _ in range(100):
        s = fresh_orch.step(1.0)
        assert s.p_total_it_kw >= 0.0


def test_tick_in_range(fresh_orch):
    for _ in range(50):
        s = fresh_orch.step(1.0)
        assert 0 <= s.tick < fresh_orch.horizon_ticks


def test_is_spike_active_is_bool(fresh_orch):
    s = fresh_orch.step(1.0)
    assert isinstance(s.is_spike_active, bool)


# ---------------------------------------------------------------------------
# 4. Power conservation: p_total = p_base + p_flex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("throttle", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_power_conservation(throttle):
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=7)
    for _ in range(20):
        s = w.step(throttle)
        assert abs(s.p_total_it_kw - (s.p_base_kw + s.p_flex_kw)) < 1e-6, (
            f"Conservation failed: total={s.p_total_it_kw}, "
            f"base={s.p_base_kw}, flex={s.p_flex_kw}"
        )


# ---------------------------------------------------------------------------
# 5. DVFS throttle behaviour
# ---------------------------------------------------------------------------

def test_throttle_zero_zeroes_flex():
    """At throttle=0, p_flex must be exactly zero."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    for _ in range(50):
        s = w.step(0.0)
        assert s.p_flex_kw == pytest.approx(0.0, abs=1e-9)


def test_throttle_one_flex_equals_nom():
    """At throttle=1, p_flex must equal p_flex_nom."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    for _ in range(50):
        s = w.step(1.0)
        assert s.p_flex_kw == pytest.approx(s.p_flex_nom_kw, rel=1e-9)


def test_throttle_half_limits_capacity():
    """throttle=0.5 limits service capacity to 50% of p_flex_max_kw.
    p_flex_kw must never exceed that capacity regardless of backlog.
    """
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    p_flex_max = w.p_flex_max_kw
    for _ in range(50):
        s = w.step(0.5)
        # served work is bounded by hardware capacity
        assert s.p_flex_kw <= p_flex_max * 0.5 + 1e-6
        # p_flex_kw equals served work (never negative)
        assert s.p_flex_kw >= 0.0


def test_throttle_does_not_affect_p_base():
    """Throttle must never change p_base (rigid load is SLA-protected)."""
    w0 = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=5)
    w1 = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=5)
    for _ in range(100):
        s0 = w0.step(0.0)   # fully throttled
        s1 = w1.step(1.0)   # full power
        assert s0.p_base_kw == pytest.approx(s1.p_base_kw, rel=1e-9), (
            "p_base must be independent of throttle"
        )


def test_throttle_monotonic():
    """Higher throttle → higher or equal service for the same tick.
    With the queue model, service is min(backlog, capacity). When
    trace demand exceeds capacity at both throttle levels, the higher
    throttle serves more. When demand is below both capacities, both
    serve the full demand (equal).
    """
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=3)
    p_flex_max = w.p_flex_max_kw
    s = w.step(1.0)   # advance to tick 0
    nom = s.p_flex_nom_kw

    if nom > 0.0:
        w.reset()
        s_low = w.step(0.0)   # zero capacity → served=0
        w.reset()
        s_hi  = w.step(1.0)   # full capacity → served=nom
        # throttle=1 always serves at least as much as throttle=0
        assert s_hi.p_flex_kw >= s_low.p_flex_kw
        assert s_low.p_flex_kw == pytest.approx(0.0, abs=1e-9)
        assert s_hi.p_flex_kw  == pytest.approx(nom, rel=1e-9)


# ---------------------------------------------------------------------------
# 6. Throttle clamping (edge cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_throttle", [-1.0, -0.01, -100.0])
def test_throttle_clamp_below_zero(bad_throttle):
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=0)
    s = w.step(bad_throttle)
    assert s.throttle == pytest.approx(0.0)
    assert s.p_flex_kw == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad_throttle", [1.01, 2.0, 1e6])
def test_throttle_clamp_above_one(bad_throttle):
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=0)
    s_clamped = w.step(bad_throttle)
    w.reset()
    s_one = w.step(1.0)
    assert s_clamped.p_flex_kw == pytest.approx(s_one.p_flex_kw, rel=1e-9)


# ---------------------------------------------------------------------------
# 6b. Backlog queue model
# ---------------------------------------------------------------------------

def test_backlog_zero_at_start():
    """Backlog starts at zero after construction."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    assert w._backlog_kw == 0.0


def test_backlog_accumulates_on_zero_throttle():
    """throttle=0 → no work served; backlog grows each step."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    states = [w.step(0.0) for _ in range(10)]
    # p_flex_kw must be zero every step
    assert all(s.p_flex_kw == pytest.approx(0.0, abs=1e-9) for s in states)
    # backlog must be strictly positive after the first non-trivial tick
    assert states[-1].backlog_kw > 0.0


def test_backlog_drains_on_full_throttle():
    """After building backlog with throttle=0, switching to throttle=1 drains it."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    for _ in range(10):
        w.step(0.0)
    backlog_before = w.step(0.0).backlog_kw
    assert backlog_before > 0.0   # sanity check

    # Now drain
    s_after = w.step(1.0)
    assert s_after.backlog_kw < backlog_before


def test_avg_delay_positive_after_throttle_zero():
    """avg_delay_steps > 0 once work has been deferred and then served."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    for _ in range(5):
        w.step(0.0)   # build backlog, no service
    s = w.step(1.0)   # serve from backlog
    assert s.avg_delay_steps >= 0.0
    if s.p_flex_kw > 0.0:   # only meaningful if work was actually served
        assert s.avg_delay_steps > 0.0


def test_backlog_reset_clears_queue():
    """reset() must clear backlog and delay accumulators."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=1)
    for _ in range(20):
        w.step(0.0)
    w.reset()
    assert w._backlog_kw == 0.0
    assert w._delay_accum_kw_steps == 0.0
    assert w._total_served_kw == 0.0
    s = w.step(1.0)
    assert s.backlog_kw == pytest.approx(0.0, abs=1e-9)


def test_power_conservation_with_backlog():
    """p_total = p_base + p_flex (served) even mid-backlog."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=9)
    for _ in range(5):
        w.step(0.0)   # build backlog
    for t in [0.0, 0.3, 0.7, 1.0]:
        s = w.step(t)
        assert abs(s.p_total_it_kw - (s.p_base_kw + s.p_flex_kw)) < 1e-6


# ---------------------------------------------------------------------------
# 7. Trace looping
# ---------------------------------------------------------------------------

def test_trace_loops_at_horizon():
    """After horizon_ticks steps, the state should repeat."""
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=0)
    h = w.horizon_ticks
    s_first = w.step(1.0)
    for _ in range(h - 1):
        w.step(1.0)
    s_loop = w.step(1.0)   # should be tick 0 again
    assert s_loop.tick == s_first.tick
    assert s_loop.p_base_kw == pytest.approx(s_first.p_base_kw, rel=1e-9)
    assert s_loop.p_flex_nom_kw == pytest.approx(s_first.p_flex_nom_kw, rel=1e-9)


# ---------------------------------------------------------------------------
# 8. Reset
# ---------------------------------------------------------------------------

def test_reset_restarts_tick():
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=0)
    for _ in range(200):
        w.step(1.0)
    w.reset()
    s = w.step(1.0)
    assert s.tick == 0


def test_reset_reproducible():
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=42)
    states_a = [w.step(1.0) for _ in range(10)]
    w.reset(seed=42)
    states_b = [w.step(1.0) for _ in range(10)]
    for a, b in zip(states_a, states_b):
        assert a.p_base_kw == pytest.approx(b.p_base_kw)
        assert a.p_flex_nom_kw == pytest.approx(b.p_flex_nom_kw)


# ---------------------------------------------------------------------------
# 9. Spike detection: at least one spike over the full horizon
# ---------------------------------------------------------------------------

def test_spike_occurs_in_full_trace():
    w = WorkloadOrchestrator(trace_dir=TRACE_DIR, seed=0)
    spikes = sum(w.step(1.0).is_spike_active for _ in range(w.horizon_ticks))
    assert spikes > 0, "No GenAI spikes detected in full 30-day trace"


# ---------------------------------------------------------------------------
# 10. Error handling
# ---------------------------------------------------------------------------

def test_missing_trace_dir_raises():
    with pytest.raises(FileNotFoundError):
        WorkloadOrchestrator(trace_dir=Path("/nonexistent/path/to/traces"))


def test_missing_column_raises(tmp_path):
    """Trace with wrong columns should raise ValueError."""
    bad = tmp_path / "batch_v2023.csv"
    bad.write_text("tick,wrong_col\n0,1.0\n1,2.0\n")
    dlrm = tmp_path / "dlrm_v2025.csv"
    dlrm.write_text("tick,active_gpu_count\n0,10.0\n1,11.0\n")
    genai = tmp_path / "genai_v2026.csv"
    genai.write_text("tick,avg_gpu_duty_cycle\n0,5.0\n1,6.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        WorkloadOrchestrator(trace_dir=tmp_path)
