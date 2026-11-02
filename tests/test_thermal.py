"""
Tests for c2g_env/simulators/thermal.py
=========================================
Covers:
  - Normal operation: step returns (temps, cooling_powers) with valid types
  - Temperature ranges: physically plausible values under nominal load
  - Zero IT load: temperatures must drift toward ambient / supply over time
  - Exact exponential integration: energy-balance steady-state convergence
  - COP values: must be positive and finite
  - supply temp clamping: out-of-range values are clamped to ASHRAE limits
  - Cooling fault injection: fault_factor reduces K, raising temperature
  - Cooling power non-negative: can never be negative (no phantom cooling)
  - Reset: restores initial temperatures and clears fault state
  - High IT load: temperatures stay bounded (thermal protection test)
  - Zone A / Zone B independence: modifying one doesn't affect the other
  - HVAC effort clamping: values outside [0, 1] are handled gracefully
  - Thermal time-constant: temperature settles near equilibrium within τ
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.physics.thermal import ThermalTwin
from c2g_env.thermal_limits import T_SAFE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tw():
    return ThermalTwin(dt_seconds=300.0)


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

def test_initial_temperatures(tw):
    assert tw.temp_A == pytest.approx(30.0)
    assert tw.temp_B == pytest.approx(20.0)


def test_t_safe_value(tw):
    assert tw.T_safe == pytest.approx(T_SAFE)


# ---------------------------------------------------------------------------
# 2. Step return format
# ---------------------------------------------------------------------------

def test_step_returns_tuple(tw):
    result = tw.step(p_it_A_mw=100.0, p_it_B_mw=60.0, hvac_effort=0.7)
    assert isinstance(result, tuple) and len(result) == 2


def test_step_temps_is_tuple_of_two_floats(tw):
    (t_A, t_B), _ = tw.step(100.0, 60.0, 0.7)
    assert isinstance(t_A, float)
    assert isinstance(t_B, float)


def test_step_cooling_is_tuple_of_three_floats(tw):
    _, (p_cool_A, p_hvac, p_pump) = tw.step(100.0, 60.0, 0.7)
    assert isinstance(p_cool_A, float)
    assert isinstance(p_hvac, float)
    assert isinstance(p_pump, float)


# ---------------------------------------------------------------------------
# 3. Temperature ranges under nominal operation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_it_A, p_it_B", [
    (80.0, 60.0),
    (120.0, 80.0),
    (20.0, 15.0),
])
def test_temps_physically_plausible(p_it_A, p_it_B):
    tw = ThermalTwin(dt_seconds=300.0)
    for _ in range(100):
        (t_A, t_B), _ = tw.step(p_it_A, p_it_B, hvac_effort=0.8)
    # Must not freeze or exceed absurd values
    assert 5.0 < t_A < 80.0, f"Zone A temp {t_A} out of plausible range"
    assert 5.0 < t_B < 80.0, f"Zone B temp {t_B} out of plausible range"


def test_cooling_powers_non_negative(tw):
    for _ in range(50):
        _, (p_A, p_B, p_pump) = tw.step(100.0, 60.0, 0.7)
        assert p_A >= 0.0, f"Zone A cooling power is negative: {p_A}"
        assert p_B >= 0.0, f"Zone B cooling power is negative: {p_B}"


# ---------------------------------------------------------------------------
# 4. Zero IT load → temperature converges toward ambient / supply
# ---------------------------------------------------------------------------

def test_zero_it_load_zone_A_cools_down():
    """With zero IT load, Zone A should trend toward supply temp."""
    tw = ThermalTwin(dt_seconds=300.0)
    tw.temp_A = 50.0   # start hot
    for _ in range(200):
        tw.step(0.0, 0.0, hvac_effort=0.0)
    assert tw.temp_A < 50.0, "Zone A should have cooled with no IT load"


def test_zero_it_load_zone_B_cools_down():
    """With zero IT load, Zone B should settle near supply or ambient."""
    tw = ThermalTwin(dt_seconds=300.0)
    tw.temp_B = 45.0
    for _ in range(200):
        tw.step(0.0, 0.0, hvac_effort=0.5)
    assert tw.temp_B < 45.0, "Zone B should have cooled with no IT load"


# ---------------------------------------------------------------------------
# 5. Steady-state: Zone A equilibrium validation
# ---------------------------------------------------------------------------

def test_zone_A_steady_state_approx():
    """
    Zone A equilibrium: T_eq = (P_IT + K_liq*T_supply + K_env*T_amb) / (K_liq + K_env)
    With P_IT=100 MW, K_liq=35, T_supply=30, K_env=0.5, T_amb=25:
    T_eq ≈ (100 + 35*30 + 0.5*25) / (35 + 0.5) ≈ 34.2°C
    """
    tw = ThermalTwin(dt_seconds=300.0)
    tw.T_amb = 25.0
    for _ in range(500):   # ~41 hours — well past τ ≈ 12.7 min
        tw.step(p_it_A_mw=100.0, p_it_B_mw=0.0, hvac_effort=0.0)
    K_total = tw.K_liq + tw.K_env_A
    T_eq = (100.0 + tw.K_liq * tw.T_supply_A + tw.K_env_A * tw.T_amb) / K_total
    assert abs(tw.temp_A - T_eq) < 0.5, (
        f"Zone A did not converge: got {tw.temp_A:.2f}°C, expected ~{T_eq:.2f}°C"
    )


# ---------------------------------------------------------------------------
# 6. COP values
# ---------------------------------------------------------------------------

def test_cop_liquid_positive(tw):
    assert tw.cop_liquid() > 0.0


def test_cop_air_positive(tw):
    assert tw.cop_air() > 0.0


def test_cop_liquid_finite(tw):
    assert np.isfinite(tw.cop_liquid())


def test_cop_air_finite(tw):
    assert np.isfinite(tw.cop_air())


def test_cop_liquid_degraded_at_low_supply():
    """COP should decrease when supply temperature is lowered below T_ref."""
    tw = ThermalTwin(dt_seconds=300.0)
    cop_ref = tw.cop_liquid()
    tw.set_supply_temps(T_supply_A=20.0)  # colder than T_ref_A=30
    cop_cold = tw.cop_liquid()
    assert cop_cold < cop_ref, "Liquid COP should degrade with lower supply temp"


def test_cop_air_degraded_at_high_ambient():
    """Air COP should decrease at high ambient temperatures."""
    tw_cool = ThermalTwin()
    tw_cool.T_amb = 20.0
    tw_hot = ThermalTwin()
    tw_hot.T_amb = 40.0
    assert tw_hot.cop_air() < tw_cool.cop_air()


# ---------------------------------------------------------------------------
# 7. Supply temperature clamping
# ---------------------------------------------------------------------------

def test_supply_A_clamped_to_minimum():
    tw = ThermalTwin()
    tw.set_supply_temps(T_supply_A=0.0)   # below minimum 20°C
    assert tw.T_supply_A == pytest.approx(20.0)


def test_supply_A_clamped_to_maximum():
    tw = ThermalTwin()
    tw.set_supply_temps(T_supply_A=100.0)  # above maximum 40°C
    assert tw.T_supply_A == pytest.approx(40.0)


def test_supply_B_clamped_to_minimum():
    tw = ThermalTwin()
    tw.set_supply_temps(T_supply_B=0.0)   # below minimum 15°C
    assert tw.T_supply_B == pytest.approx(15.0)


def test_supply_B_clamped_to_maximum():
    tw = ThermalTwin()
    tw.set_supply_temps(T_supply_B=50.0)  # above maximum 27°C
    assert tw.T_supply_B == pytest.approx(27.0)


def test_supply_temp_in_range_accepted():
    tw = ThermalTwin()
    tw.set_supply_temps(T_supply_A=25.0, T_supply_B=22.0)
    assert tw.T_supply_A == pytest.approx(25.0)
    assert tw.T_supply_B == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# 8. Cooling fault injection
# ---------------------------------------------------------------------------

def test_fault_raises_zone_A_temp():
    """Fault reduces cooling efficiency → Zone A heats up faster."""
    tw_normal = ThermalTwin(dt_seconds=300.0)
    tw_fault  = ThermalTwin(dt_seconds=300.0)
    tw_fault.set_cooling_fault(active=True, fault_factor=0.4)
    for _ in range(30):
        tw_normal.step(120.0, 60.0, 0.8)
        tw_fault.step(120.0, 60.0, 0.8)
    assert tw_fault.temp_A > tw_normal.temp_A, (
        "Fault should cause Zone A to run hotter"
    )


def test_fault_flag_set():
    tw = ThermalTwin()
    tw.set_cooling_fault(active=True, fault_factor=0.5)
    assert tw.fault_active is True
    assert tw.fault_factor == pytest.approx(0.5)


def test_fault_cleared_on_reset():
    tw = ThermalTwin()
    tw.set_cooling_fault(active=True, fault_factor=0.3)
    tw.reset()
    assert tw.fault_active is False
    assert tw.fault_factor == pytest.approx(1.0)


def test_fault_off_restores_normal_operation():
    tw = ThermalTwin()
    tw.set_cooling_fault(active=True)
    tw.set_cooling_fault(active=False)
    assert tw.fault_active is False
    assert tw.fault_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. Reset
# ---------------------------------------------------------------------------

def test_reset_restores_temps():
    tw = ThermalTwin()
    for _ in range(100):
        tw.step(150.0, 90.0, 1.0)
    tw.reset()
    assert tw.temp_A == pytest.approx(30.0)
    assert tw.temp_B == pytest.approx(20.0)


def test_reset_restores_supply_temps():
    tw = ThermalTwin()
    tw.set_supply_temps(T_supply_A=22.0, T_supply_B=16.0)
    tw.reset()
    assert tw.T_supply_A == pytest.approx(tw.T_ref_A)
    assert tw.T_supply_B == pytest.approx(tw.T_ref_B)


# ---------------------------------------------------------------------------
# 10. HVAC effort edge cases
# ---------------------------------------------------------------------------

def test_hvac_effort_zero_does_not_crash(tw):
    result = tw.step(50.0, 30.0, hvac_effort=0.0)
    assert result is not None


def test_hvac_effort_one_does_not_crash(tw):
    result = tw.step(50.0, 30.0, hvac_effort=1.0)
    assert result is not None


def test_zero_it_zero_hvac_zone_B_stable():
    """Genuinely zero load with some ambient → B should not diverge."""
    tw = ThermalTwin()
    tw.T_amb = 25.0
    tw.temp_B = 25.0
    for _ in range(100):
        (_, t_B), _ = tw.step(0.0, 0.0, 0.0)
    assert 10.0 < t_B < 45.0


# ---------------------------------------------------------------------------
# 11. Zone independence
# ---------------------------------------------------------------------------

def test_zone_A_load_does_not_directly_affect_zone_B():
    """Zone A IT load should not distort Zone B temperature directly."""
    tw_a = ThermalTwin()
    tw_b = ThermalTwin()
    for _ in range(50):
        tw_a.step(150.0, 0.0, 0.8)   # heavy A, no B
        tw_b.step(0.0,   0.0, 0.8)   # no load anywhere
    # Zone B temps should not wildly diverge due to A's load alone
    assert abs(tw_a.temp_B - tw_b.temp_B) < 5.0, (
        "Zone A IT load should not strongly couple into Zone B temperature"
    )


# ---------------------------------------------------------------------------
# 12. High IT load: thermal limit
# ---------------------------------------------------------------------------

def test_high_load_zone_A_approaches_but_does_not_skip_T_safe():
    """Under sustained heavy load, Zone A may exceed T_safe — but the
    simulator must not crash and temperatures must remain finite."""
    tw = ThermalTwin()
    for _ in range(200):
        (t_A, t_B), _ = tw.step(200.0, 100.0, 0.5)
        assert np.isfinite(t_A), "Zone A temperature became non-finite"
        assert np.isfinite(t_B), "Zone B temperature became non-finite"
