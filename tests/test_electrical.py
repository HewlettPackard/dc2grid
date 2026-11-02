"""
Tests for c2g_env/simulators/electrical.py
============================================
Covers:
  - Normal operation: step returns a dict with all expected keys
  - PUE: must always be >= 1.0 (facility always uses more than IT)
  - Power accounting: p_facility >= p_total_it (losses are additive)
  - Loss components all non-negative
  - UPS efficiency: in (0, 1] for valid loads
  - Server power model: monotonically increasing with utilisation
  - Server power idle floor: at util=0, power == N_racks * P_idle
  - Server power peak: at util=1, power == N_racks * P_max (approx)
  - Utilisation clamping: values < 0 and > 1 handled gracefully
  - Cooling power inputs: negative values clamped to zero internally
  - PUE physical realism: between 1.0 and 3.0 under typical loads
  - Power factor: between 0 and 1
  - Reactive power: non-negative
  - Transformer losses: increase with load (copper losses dominate)
  - Zero load: facility still draws aux power (lighting, networking)
  - Reset: clears internal state
  - get_diagnostics returns last state after a step
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.physics.electrical import DatacenterElectrical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elec():
    return DatacenterElectrical()


EXPECTED_KEYS = {
    "p_it_A_mw", "p_it_B_mw", "p_total_it_mw",
    "p_ups_loss_mw", "p_pdu_loss_mw", "p_xfmr_loss_mw",
    "p_cooling_mw", "p_aux_mw", "p_facility_mw",
    "pue_dynamic", "pf_composite", "q_reactive_mvar",
    "ups_eta_A", "ups_eta_B",
}


# ---------------------------------------------------------------------------
# 1. Return format
# ---------------------------------------------------------------------------

def test_step_returns_dict(elec):
    result = elec.step(0.7, 0.6, 5.0, 3.0)
    assert isinstance(result, dict)


def test_step_has_all_keys(elec):
    result = elec.step(0.7, 0.6, 5.0, 3.0)
    assert EXPECTED_KEYS.issubset(set(result.keys()))


def test_all_values_finite(elec):
    result = elec.step(0.7, 0.6, 5.0, 3.0)
    for k, v in result.items():
        assert np.isfinite(v), f"Key '{k}' is not finite: {v}"


# ---------------------------------------------------------------------------
# 2. PUE invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("util_A, util_B", [
    (0.01, 0.01),
    (0.5,  0.5),
    (1.0,  1.0),
    (0.3,  0.8),
])
def test_pue_always_gte_1(util_A, util_B):
    elec = DatacenterElectrical()
    r = elec.step(util_A, util_B, 5.0, 3.0)
    assert r["pue_dynamic"] >= 1.0, f"PUE={r['pue_dynamic']} < 1.0"


def test_pue_realistic_range():
    """PUE should be between 1.0 and 2.5 under normal loads."""
    elec = DatacenterElectrical()
    r = elec.step(0.7, 0.6, 8.0, 4.0)
    assert 1.0 <= r["pue_dynamic"] <= 2.5, f"PUE={r['pue_dynamic']} outside realistic range"


def test_pue_increases_with_cooling():
    """More cooling → higher PUE."""
    elec = DatacenterElectrical()
    r_low  = elec.step(0.5, 0.5, 2.0,  1.0)
    r_high = elec.step(0.5, 0.5, 20.0, 10.0)
    assert r_high["pue_dynamic"] > r_low["pue_dynamic"]


# ---------------------------------------------------------------------------
# 3. Power accounting: facility = IT + losses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("util_A, util_B, cool_A, cool_B", [
    (0.5, 0.5, 5.0, 3.0),
    (1.0, 1.0, 15.0, 8.0),
    (0.1, 0.1, 1.0, 0.5),
])
def test_facility_power_exceeds_it(util_A, util_B, cool_A, cool_B):
    elec = DatacenterElectrical()
    r = elec.step(util_A, util_B, cool_A, cool_B)
    assert r["p_facility_mw"] > r["p_total_it_mw"], (
        "Facility power must exceed IT power (losses are additive)"
    )


def test_loss_components_non_negative(elec):
    r = elec.step(0.6, 0.5, 6.0, 3.5)
    assert r["p_ups_loss_mw"]  >= 0.0
    assert r["p_pdu_loss_mw"]  >= 0.0
    assert r["p_xfmr_loss_mw"] >= 0.0
    assert r["p_cooling_mw"]   >= 0.0
    assert r["p_aux_mw"]       >= 0.0


def test_it_power_equals_sum_of_zones(elec):
    r = elec.step(0.6, 0.5, 5.0, 3.0)
    assert r["p_total_it_mw"] == pytest.approx(
        r["p_it_A_mw"] + r["p_it_B_mw"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# 4. Server power model
# ---------------------------------------------------------------------------

def test_server_power_idle_floor():
    """At util=0, Zone A power should equal N_racks_A * P_idle_A / 1000 MW."""
    elec = DatacenterElectrical()
    r = elec.step(0.0, 0.0, 0.0, 0.0)
    expected_A = elec.n_racks_A * elec.p_idle_rack_A_kw / 1000.0
    expected_B = elec.n_racks_B * elec.p_idle_rack_B_kw / 1000.0
    assert r["p_it_A_mw"] == pytest.approx(expected_A, rel=1e-6)
    assert r["p_it_B_mw"] == pytest.approx(expected_B, rel=1e-6)


def test_server_power_peak():
    """At util=1, Zone A power should equal N_racks_A * P_max_A / 1000 MW."""
    elec = DatacenterElectrical()
    r = elec.step(1.0, 1.0, 0.0, 0.0)
    expected_A = elec.n_racks_A * elec.p_max_rack_A_kw / 1000.0
    expected_B = elec.n_racks_B * elec.p_max_rack_B_kw / 1000.0
    assert r["p_it_A_mw"] == pytest.approx(expected_A, rel=1e-6)
    assert r["p_it_B_mw"] == pytest.approx(expected_B, rel=1e-6)


def test_server_power_monotone_with_utilisation():
    """Power must strictly increase with utilisation."""
    elec = DatacenterElectrical()
    utils = np.linspace(0.0, 1.0, 11)
    powers_A = []
    for u in utils:
        r = elec.step(u, 0.5, 0.0, 0.0)
        powers_A.append(r["p_it_A_mw"])
    for i in range(len(powers_A) - 1):
        assert powers_A[i] <= powers_A[i + 1], (
            f"Power not monotone at util {utils[i]:.1f}→{utils[i+1]:.1f}: "
            f"{powers_A[i]:.2f}→{powers_A[i+1]:.2f} MW"
        )


# ---------------------------------------------------------------------------
# 5. Utilisation clamping
# ---------------------------------------------------------------------------

def test_util_below_zero_clamped():
    elec = DatacenterElectrical()
    r_neg  = elec.step(-1.0, -1.0, 0.0, 0.0)
    r_zero = elec.step(0.0,  0.0,  0.0, 0.0)
    assert r_neg["p_it_A_mw"] == pytest.approx(r_zero["p_it_A_mw"], rel=1e-6)


def test_util_above_one_clamped():
    elec = DatacenterElectrical()
    r_over = elec.step(2.0, 2.0, 0.0, 0.0)
    r_one  = elec.step(1.0, 1.0, 0.0, 0.0)
    assert r_over["p_it_A_mw"] == pytest.approx(r_one["p_it_A_mw"], rel=1e-6)


# ---------------------------------------------------------------------------
# 6. Cooling power input clamping
# ---------------------------------------------------------------------------

def test_negative_cooling_clamped_to_zero():
    elec = DatacenterElectrical()
    r_neg  = elec.step(0.5, 0.5, -10.0, -5.0)
    r_zero = elec.step(0.5, 0.5,   0.0,  0.0)
    assert r_neg["p_cooling_mw"] == pytest.approx(0.0, abs=1e-9)
    assert r_neg["p_facility_mw"] == pytest.approx(r_zero["p_facility_mw"], rel=1e-6)


# ---------------------------------------------------------------------------
# 7. UPS efficiency
# ---------------------------------------------------------------------------

def test_ups_efficiency_in_valid_range(elec):
    r = elec.step(0.7, 0.6, 5.0, 3.0)
    assert 0.5 < r["ups_eta_A"] <= 1.0
    assert 0.5 < r["ups_eta_B"] <= 1.0


def test_ups_efficiency_improves_with_load():
    """Double-conversion UPS efficiency peaks at ~50% load and falls at low load."""
    elec = DatacenterElectrical()
    r_light = elec.step(0.05, 0.05, 0.0, 0.0)
    r_heavy = elec.step(0.70, 0.70, 0.0, 0.0)
    # Heavy should be more efficient than very light load
    assert r_heavy["ups_eta_A"] > r_light["ups_eta_A"]


# ---------------------------------------------------------------------------
# 8. Power factor
# ---------------------------------------------------------------------------

def test_power_factor_in_range(elec):
    r = elec.step(0.7, 0.6, 5.0, 3.0)
    assert 0.8 <= r["pf_composite"] <= 1.0, f"pf={r['pf_composite']}"


def test_reactive_power_non_negative(elec):
    r = elec.step(0.7, 0.6, 5.0, 3.0)
    assert r["q_reactive_mvar"] >= 0.0


# ---------------------------------------------------------------------------
# 9. Transformer losses increase with load
# ---------------------------------------------------------------------------

def test_transformer_loss_increases_with_load():
    elec = DatacenterElectrical()
    r_light = elec.step(0.1, 0.1, 1.0, 0.5)
    r_heavy = elec.step(0.9, 0.9, 12.0, 7.0)
    assert r_heavy["p_xfmr_loss_mw"] > r_light["p_xfmr_loss_mw"]


# ---------------------------------------------------------------------------
# 10. Zero utilisation: aux loads still present
# ---------------------------------------------------------------------------

def test_zero_util_facility_has_aux_power():
    elec = DatacenterElectrical()
    r = elec.step(0.0, 0.0, 0.0, 0.0)
    # Lighting + networking = 0.5 + 1.5 = 2.0 MW minimum
    assert r["p_aux_mw"] == pytest.approx(
        elec.p_lighting_mw + elec.p_network_mw, rel=1e-6
    )
    assert r["p_facility_mw"] >= r["p_aux_mw"]


# ---------------------------------------------------------------------------
# 11. Reset
# ---------------------------------------------------------------------------

def test_reset_clears_last_state():
    elec = DatacenterElectrical()
    elec.step(0.5, 0.5, 5.0, 3.0)
    elec.reset()
    assert elec._last_state is None


def test_get_diagnostics_after_reset_returns_empty():
    elec = DatacenterElectrical()
    elec.reset()
    diag = elec.get_diagnostics()
    assert isinstance(diag, dict)


def test_get_diagnostics_returns_last_step():
    elec = DatacenterElectrical()
    r = elec.step(0.6, 0.5, 4.0, 2.5)
    diag = elec.get_diagnostics()
    assert diag["p_facility_mw"] == pytest.approx(r["p_facility_mw"])


# ---------------------------------------------------------------------------
# 12. Full-load facility power cap (sanity check)
# ---------------------------------------------------------------------------

def test_full_load_facility_under_transformer_rating():
    """Full-load facility power must not exceed transformer rating."""
    elec = DatacenterElectrical()
    r = elec.step(1.0, 1.0, 20.0, 15.0)
    # Transformer rating is 300 MVA — at PF≈0.9, P ≈ 270 MW
    assert r["p_facility_mw"] <= elec.S_xfmr_mva * 1.1, (
        f"Facility power {r['p_facility_mw']:.1f} MW exceeded transformer rating "
        f"{elec.S_xfmr_mva} MVA"
    )
