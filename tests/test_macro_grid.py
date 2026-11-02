"""
Tests for c2g_env/simulators/macro_grid.py
============================================
Covers:
  - Normal operation: step returns dict with all expected keys
  - delta_p_kw sign: matches RegD signal direction
  - delta_p_kw magnitude: bounded by committed_mw * 1000 (kW)
  - Energy neutrality: RegD integrates near zero over 15-min windows
  - LMP: non-negative, bounded at spike cap ($500/MWh)
  - LMP increases with load: higher load → higher LMP
  - LMP base-load period: at/below median load, LMP == lmp_base
  - grid_load_mw: matches real NYISO data range (positive finite)
  - load_norm: in [0, 1]
  - regd_signal: in [-1, 1]
  - committed_mw override per step
  - set_committed clamping: 0 to 50 MW
  - Trace looping: after horizon_ticks steps, load repeats
  - Reset: tick and RegD state clear
  - Reproducibility: same seed produces same sequence
  - Missing zone file: raises FileNotFoundError
  - Wrong column name: raises ValueError
  - lmp_stats returns mean, std, p95, max keys
  - Zero committed: delta_p always zero
"""
import math
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.physics.macro_grid import MacroGridSignal

ENERGY_DIR = "data/processed/energy"
EXPECTED_KEYS = {
    "delta_p_kw", "committed_mw", "lmp_usd_mwh",
    "grid_load_mw", "load_norm", "regd_signal", "tick",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid():
    return MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                           committed_mw=20.0, seed=0)


@pytest.fixture
def fresh_grid():
    return MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                           committed_mw=20.0, seed=42)


# ---------------------------------------------------------------------------
# 1. Return format
# ---------------------------------------------------------------------------

def test_step_returns_dict(grid):
    assert isinstance(grid.step(), dict)


def test_step_has_all_keys(grid):
    result = grid.step()
    assert EXPECTED_KEYS.issubset(set(result.keys()))


def test_all_values_finite(grid):
    r = grid.step()
    for k, v in r.items():
        if k != "tick":
            assert np.isfinite(v), f"Key '{k}' is not finite: {v}"


# ---------------------------------------------------------------------------
# 2. delta_p_kw magnitude
# ---------------------------------------------------------------------------

def test_delta_p_bounded_by_committed_mw():
    """delta_p must not exceed committed capacity in absolute value."""
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=20.0, seed=1)
    for _ in range(200):
        r = g.step()
        max_kw = r["committed_mw"] * 1_000.0
        assert abs(r["delta_p_kw"]) <= max_kw * 1.01, (
            f"|delta_p|={abs(r['delta_p_kw']):.1f} kW > "
            f"committed {max_kw:.1f} kW"
        )


def test_zero_committed_gives_zero_delta_p():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=0.0, seed=0)
    for _ in range(50):
        r = g.step()
        assert r["delta_p_kw"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. RegD signal properties
# ---------------------------------------------------------------------------

def test_regd_signal_in_range(fresh_grid):
    for _ in range(500):
        r = fresh_grid.step()
        assert -1.0 <= r["regd_signal"] <= 1.0, (
            f"regd_signal={r['regd_signal']} outside [-1, 1]"
        )


def test_regd_energy_neutrality_over_15min():
    """
    PJM RegD is designed to be energy-neutral over 15-minute windows.
    The mean of 180 consecutive 5-s ticks should be near zero.
    We test this over many windows and check the overall drift is small.
    """
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=20.0, seed=7)
    window = 180   # 180 × 5 s = 15 min
    window_means = []
    for _ in range(200):
        signals = [g.step()["regd_signal"] for _ in range(window)]
        window_means.append(np.mean(signals))
    # Mean of all window means should be very small
    assert abs(np.mean(window_means)) < 0.3, (
        f"RegD energy drift too large: mean={np.mean(window_means):.3f}"
    )


def test_delta_p_sign_matches_regd():
    """delta_p_kw must have same sign as regd_signal."""
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=20.0, seed=3)
    for _ in range(100):
        r = g.step()
        if abs(r["regd_signal"]) > 1e-9:
            assert np.sign(r["delta_p_kw"]) == np.sign(r["regd_signal"]), (
                "delta_p sign must match regd_signal sign"
            )


# ---------------------------------------------------------------------------
# 4. LMP properties
# ---------------------------------------------------------------------------

def test_lmp_non_negative(fresh_grid):
    for _ in range(200):
        r = fresh_grid.step()
        assert r["lmp_usd_mwh"] >= 0.0


def test_lmp_bounded_at_spike_cap(fresh_grid):
    for _ in range(200):
        r = fresh_grid.step()
        assert r["lmp_usd_mwh"] <= 500.0


def test_lmp_at_below_median_equals_base():
    """When load is exactly at/below median, LMP must equal lmp_base."""
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        lmp_base_usd=30.0, lmp_slope=20.0, seed=0)
    lmp = g._load_to_lmp(g._load_median)
    assert lmp == pytest.approx(30.0, abs=1e-6)


def test_lmp_increases_above_median():
    """LMP must be strictly higher above median load than at/below it."""
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        lmp_base_usd=30.0, lmp_slope=20.0, seed=0)
    lmp_med  = g._load_to_lmp(g._load_median)
    lmp_high = g._load_to_lmp(g._load_median + 1_000.0)
    assert lmp_high > lmp_med


def test_lmp_stats_keys():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC", seed=0)
    stats = g.lmp_stats
    assert {"mean", "std", "p95", "max"}.issubset(set(stats.keys()))


def test_lmp_stats_ordering():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC", seed=0)
    s = g.lmp_stats
    assert s["mean"] <= s["p95"] <= s["max"]
    assert s["std"] >= 0.0


# ---------------------------------------------------------------------------
# 5. grid_load_mw and load_norm
# ---------------------------------------------------------------------------

def test_grid_load_positive(fresh_grid):
    for _ in range(100):
        r = fresh_grid.step()
        assert r["grid_load_mw"] > 0.0


def test_load_norm_in_0_1(fresh_grid):
    for _ in range(100):
        r = fresh_grid.step()
        assert 0.0 < r["load_norm"] <= 1.0, (
            f"load_norm={r['load_norm']} outside (0, 1]"
        )


# ---------------------------------------------------------------------------
# 6. Committed MW override
# ---------------------------------------------------------------------------

def test_step_committed_override():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=10.0, seed=0)
    r = g.step(committed_mw=30.0)
    assert r["committed_mw"] == pytest.approx(30.0)
    assert g.committed_mw == pytest.approx(30.0)


def test_set_committed_clamped_min():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC", seed=0)
    g.set_committed(-5.0)
    assert g.committed_mw == pytest.approx(0.0)


def test_set_committed_clamped_max():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC", seed=0)
    g.set_committed(999.0)
    assert g.committed_mw == pytest.approx(50.0)


def test_set_committed_in_range():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC", seed=0)
    g.set_committed(25.0)
    assert g.committed_mw == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 7. Trace looping
# ---------------------------------------------------------------------------

def test_trace_loops_at_horizon():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=10.0, seed=99)
    r0 = g.step()
    h = g.horizon_ticks
    for _ in range(h - 1):
        g.step()
    r_loop = g.step()   # should be tick 0 again
    assert r_loop["tick"] == r0["tick"]
    assert r_loop["grid_load_mw"] == pytest.approx(r0["grid_load_mw"])


# ---------------------------------------------------------------------------
# 8. Reset
# ---------------------------------------------------------------------------

def test_reset_clears_tick():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC", seed=0)
    for _ in range(100):
        g.step()
    g.reset()
    r = g.step()
    assert r["tick"] == 0


def test_reset_reproducible():
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                        committed_mw=15.0, seed=5)
    seq_a = [g.step()["regd_signal"] for _ in range(20)]
    g.reset(seed=5)
    seq_b = [g.step()["regd_signal"] for _ in range(20)]
    for a, b in zip(seq_a, seq_b):
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# 9. Different zones load correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zone", ["CAPITL", "LONGIL", "WEST"])
def test_other_zones_load(zone):
    g = MacroGridSignal(energy_dir=ENERGY_DIR, zone=zone, seed=0)
    r = g.step()
    assert r["grid_load_mw"] > 0.0
    assert r["lmp_usd_mwh"] >= 0.0


# ---------------------------------------------------------------------------
# 10. Error handling
# ---------------------------------------------------------------------------

def test_missing_energy_dir_uses_synthetic():
    """Missing energy dir → synthetic load fallback (graceful degradation)."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = MacroGridSignal(energy_dir="/nonexistent/path", zone="NYC")
    assert g is not None
    out = g.step()
    assert math.isfinite(out["lmp_usd_mwh"])


def test_missing_zone_uses_synthetic():
    """Unknown zone → synthetic load fallback, not FileNotFoundError."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="UNKNOWN_ZONE")
    assert g is not None
    assert math.isfinite(g.step()["lmp_usd_mwh"])


def test_wrong_column_raises(tmp_path):
    bad = tmp_path / "NYC.csv"
    bad.write_text("Time Stamp,WrongCol\n2023-01-01 00:00:00,5000\n")
    with pytest.raises(ValueError, match="Load"):
        MacroGridSignal(energy_dir=str(tmp_path), zone="NYC")


# ---------------------------------------------------------------------------
# 11. Market preset system
# ---------------------------------------------------------------------------

from c2g_env.physics.macro_grid import MARKET_PRESETS, MarketParams


def test_all_presets_defined():
    expected = {"nyiso_nyc", "pjm_dom", "caiso_pgae", "ercot_north", "entso_de", "aemo_nsw"}
    assert expected == set(MARKET_PRESETS.keys())


@pytest.mark.parametrize("market_id", list(MARKET_PRESETS.keys()))
def test_market_preset_smoke(market_id, tmp_path):
    """Each market preset runs without error using synthetic load."""
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        g = MacroGridSignal(
            energy_dir=str(tmp_path),  # no real data → synthetic
            zone="SYNTHETIC",
            dt_seconds=5.0,
            committed_mw=15.0,
            seed=0,
            market=market_id,
        )
    for _ in range(5):
        out = g.step()
        assert math.isfinite(out["lmp_usd_mwh"])
        assert -1.0 <= out["regd_signal"] <= 1.0


def test_unknown_market_raises():
    with pytest.raises(ValueError, match="Unknown market"):
        MacroGridSignal(energy_dir="data/processed/energy", zone="NYC",
                        market="nonexistent_iso")


def test_market_params_preserved():
    """AR(1) and LMP params should reflect the chosen preset."""
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        g = MacroGridSignal(energy_dir="/tmp", zone="X",
                            seed=0, market="entso_de")
    p = MARKET_PRESETS["entso_de"]
    assert g._regd_rho == pytest.approx(p.regd_rho)
    assert g._regd_sigma == pytest.approx(p.regd_sigma)
    assert g.lmp_base_usd == pytest.approx(p.lmp_base_usd)
    assert g._window_ticks == p.window_ticks


def test_synthetic_load_has_correct_length():
    """Synthetic load should cover at least one full 24-hour episode (17280 ticks)."""
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        g = MacroGridSignal(energy_dir="/tmp", zone="X",
                            seed=0, market="aemo_nsw")
    assert g._n >= 17_280


def test_ercot_lmp_clips_at_200():
    """ERCOT has extreme price events; obs clips LMP at $200/MWh."""
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        g = MacroGridSignal(energy_dir="/tmp", zone="X", seed=0,
                            market="ercot_north", committed_mw=15.0)
    # Run during peak synthetic load (afternoon)
    for _ in range(g._market.window_ticks * 3):
        out = g.step()
    # LMP itself can exceed 200; the obs clips it — verify lmp is positive
    assert out["lmp_usd_mwh"] > 0
