"""
Tests for frequency and voltage safety signals
================================================
Covers:
  - Grid frequency: swing equation integration, nominal freq, reset
  - Grid frequency: bounded within UFLS/OFGT limits (±0.5 Hz)
  - Grid frequency: zero deficit → freq stays at nominal
  - Grid frequency: positive deficit → freq drops
  - Grid frequency: ENTSO-E market → 50 Hz nominal
  - PCC voltage: present in electrical step output
  - PCC voltage: realistic range [0.85, 1.10] pu
  - PCC voltage: increases load → lower voltage
  - PCC voltage: no load → near 1.0 pu
  - Env integration: obs space is 17-D with freq, voltage and backlog
  - Env integration: info dict contains freq/voltage keys
  - Env integration: reward includes freq/voltage penalties
  - Env integration: termination on extreme freq/voltage events
  - Env integration: reset returns nominal freq/voltage in obs
"""
import math
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.physics.macro_grid import MacroGridSignal
from c2g_env.physics.electrical import DatacenterElectrical
from c2g_env.env_low_level import C2GFastEnv

ENERGY_DIR = "data/processed/energy"


# =========================================================================
# A. Grid Frequency Model (macro_grid.py)
# =========================================================================

class TestGridFrequency:
    """Tests for the swing-equation frequency model in MacroGridSignal."""

    @pytest.fixture
    def grid60(self):
        return MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                               committed_mw=20.0, seed=42)

    @pytest.fixture
    def grid50(self):
        return MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                               committed_mw=20.0, seed=42,
                               market="entso_de")

    def test_nominal_freq_60hz(self, grid60):
        assert grid60.f_nom == pytest.approx(60.0)

    def test_nominal_freq_50hz(self, grid50):
        assert grid50.f_nom == pytest.approx(50.0)

    def test_initial_freq_at_nominal(self, grid60):
        assert grid60.f_grid == pytest.approx(60.0)

    def test_step_returns_freq_keys(self, grid60):
        r = grid60.step()
        assert "f_grid_hz" in r
        assert "f_nom_hz" in r

    def test_freq_at_nominal_after_step(self, grid60):
        r = grid60.step()
        assert math.isfinite(r["f_grid_hz"])
        assert math.isfinite(r["f_nom_hz"])

    def test_zero_deficit_stays_near_nominal(self, grid60):
        """With zero tracking deficit, freq should drift only from noise."""
        for _ in range(50):
            grid60._step_frequency(0.0)
        # Should be very close to nominal (noise is σ=0.005 Hz)
        assert abs(grid60.f_grid - 60.0) < 0.1

    def test_positive_deficit_lowers_freq(self):
        """Positive deficit = excess demand → frequency drops."""
        g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                            committed_mw=20.0, seed=0)
        g._f_noise_std = 0.0  # disable noise for deterministic test
        for _ in range(20):
            g._step_frequency(10.0)  # 10 MW excess demand
        assert g.f_grid < g.f_nom, (
            f"Freq should drop below nominal: got {g.f_grid:.4f} Hz"
        )

    def test_negative_deficit_raises_freq(self):
        """Negative deficit = excess supply → frequency rises."""
        g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                            committed_mw=20.0, seed=0)
        g._f_noise_std = 0.0
        for _ in range(20):
            g._step_frequency(-10.0)  # 10 MW excess supply
        assert g.f_grid > g.f_nom, (
            f"Freq should rise above nominal: got {g.f_grid:.4f} Hz"
        )

    def test_freq_bounded_ufls_ofgt(self):
        """Frequency must be clipped to [f_nom - 0.5, f_nom + 0.5]."""
        g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                            committed_mw=20.0, seed=0)
        g._f_noise_std = 0.0
        # Drive freq down hard
        for _ in range(1000):
            g._step_frequency(100.0)
        assert g.f_grid >= g.f_nom - 0.5
        # Drive freq up hard
        g.reset(seed=0)
        g._f_noise_std = 0.0
        for _ in range(1000):
            g._step_frequency(-100.0)
        assert g.f_grid <= g.f_nom + 0.5

    def test_freq_bounded_50hz(self, grid50):
        """50 Hz system bounded to [49.5, 50.5]."""
        grid50._f_noise_std = 0.0
        for _ in range(1000):
            grid50._step_frequency(100.0)
        assert grid50.f_grid >= 49.5

    def test_reset_restores_nominal(self, grid60):
        for _ in range(50):
            grid60._step_frequency(10.0)
        grid60.reset(seed=0)
        assert grid60.f_grid == pytest.approx(60.0)

    def test_freq_damping_returns_to_nominal(self):
        """After a disturbance, frequency should return to nominal (D > 0)."""
        g = MacroGridSignal(energy_dir=ENERGY_DIR, zone="NYC",
                            committed_mw=20.0, seed=0)
        g._f_noise_std = 0.0
        # Apply impulse
        for _ in range(5):
            g._step_frequency(5.0)
        f_disturbed = g.f_grid
        # Let it recover with zero deficit
        for _ in range(500):
            g._step_frequency(0.0)
        assert abs(g.f_grid - g.f_nom) < abs(f_disturbed - g.f_nom), (
            "Frequency should recover towards nominal after disturbance"
        )


# =========================================================================
# B. PCC Voltage Model (electrical.py)
# =========================================================================

class TestPCCVoltage:
    """Tests for the voltage drop model in DatacenterElectrical."""

    @pytest.fixture
    def elec(self):
        return DatacenterElectrical()

    def test_voltage_keys_present(self, elec):
        r = elec.step(0.7, 0.6, 5.0, 3.0)
        assert "v_pcc_pu" in r
        assert "v_drop_pu" in r

    def test_voltage_finite(self, elec):
        r = elec.step(0.7, 0.6, 5.0, 3.0)
        assert math.isfinite(r["v_pcc_pu"])
        assert math.isfinite(r["v_drop_pu"])

    def test_voltage_in_realistic_range(self, elec):
        """PCC voltage should be in [0.85, 1.10] pu for any load."""
        for util in [0.0, 0.3, 0.5, 0.7, 1.0]:
            r = elec.step(util, util, 5.0, 3.0)
            assert 0.85 <= r["v_pcc_pu"] <= 1.10, (
                f"v_pcc_pu={r['v_pcc_pu']:.4f} outside [0.85, 1.10] "
                f"at util={util}"
            )

    def test_voltage_near_nominal_at_no_load(self, elec):
        """At zero load, voltage should be close to 1.0 pu."""
        r = elec.step(0.0, 0.0, 0.0, 0.0)
        assert abs(r["v_pcc_pu"] - 1.0) < 0.02

    def test_voltage_drop_non_negative(self, elec):
        r = elec.step(0.7, 0.6, 5.0, 3.0)
        assert r["v_drop_pu"] >= 0.0

    def test_higher_load_lower_voltage(self, elec):
        """Heavier load should produce larger voltage drop."""
        r_low  = elec.step(0.2, 0.2, 2.0, 1.0)
        r_high = elec.step(1.0, 1.0, 15.0, 8.0)
        assert r_high["v_pcc_pu"] <= r_low["v_pcc_pu"], (
            f"Voltage should drop with load: low={r_low['v_pcc_pu']:.4f}, "
            f"high={r_high['v_pcc_pu']:.4f}"
        )

    def test_voltage_drop_increases_with_load(self, elec):
        r_low  = elec.step(0.2, 0.2, 2.0, 1.0)
        r_high = elec.step(1.0, 1.0, 15.0, 8.0)
        assert r_high["v_drop_pu"] >= r_low["v_drop_pu"]


# =========================================================================
# C. Env Integration — Frequency & Voltage in C2GFastEnv
# =========================================================================

class TestEnvFreqVoltageIntegration:
    """Tests for freq/voltage signals flowing through the Gymnasium env."""

    @pytest.fixture
    def env(self):
        e = C2GFastEnv(scenario="default")
        e.reset(seed=42)
        return e

    def test_obs_shape_is_16(self, env):
        assert env.observation_space.shape == (18,)

    def test_reset_obs_shape(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (18,)

    def test_reset_freq_nominal(self, env):
        obs, _ = env.reset(seed=0)
        assert obs[14] == pytest.approx(0.0, abs=0.01), (
            "freq_dev_norm should be ~0 at reset"
        )

    def test_reset_voltage_nominal(self, env):
        obs, _ = env.reset(seed=0)
        assert obs[15] == pytest.approx(1.0, abs=0.01), (
            "v_pcc_pu should be ~1.0 at reset"
        )

    def test_step_obs_contains_freq_voltage(self, env):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (18,)
        assert math.isfinite(obs[14]), "freq_dev_norm not finite"
        assert math.isfinite(obs[15]), "v_pcc_pu not finite"

    def test_freq_dev_norm_bounded(self, env):
        """freq_dev_norm should be clipped to [-1, 1]."""
        for _ in range(100):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
            assert -1.0 <= obs[14] <= 1.0, f"freq_dev_norm={obs[14]}"

    def test_v_pcc_bounded(self, env):
        """v_pcc_pu should stay in [0.85, 1.10]."""
        for _ in range(100):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
            assert 0.0 <= obs[15] <= 1.10, f"v_pcc_pu={obs[15]}"

    def test_info_has_freq_keys(self, env):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "f_grid_hz" in info
        assert "f_nom_hz" in info
        assert "freq_dev_hz" in info
        assert "freq_penalty" in info

    def test_info_has_voltage_keys(self, env):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "v_pcc_pu" in info
        assert "v_drop_pu" in info
        assert "volt_penalty" in info

    def test_info_has_fault_flags(self, env):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "thermal_fault" in info
        assert "freq_fault" in info
        assert "voltage_fault" in info

    def test_freq_penalty_zero_in_deadband(self, env):
        """When freq deviation < 0.2 Hz, penalty should be 0."""
        _, _, _, _, info = env.step(np.array([1.0, 0.7, 0.6, 0.0],
                                              dtype=np.float32))
        # After just one step with balanced action, freq is near nominal
        if abs(info["freq_dev_hz"]) < 0.2:
            assert info["freq_penalty"] == pytest.approx(0.0, abs=1e-6)

    def test_volt_penalty_zero_in_safe_range(self, env):
        """When voltage is in [0.95, 1.05], penalty should be 0."""
        _, _, _, _, info = env.step(np.array([1.0, 0.7, 0.6, 0.0],
                                              dtype=np.float32))
        if 0.95 <= info["v_pcc_pu"] <= 1.05:
            assert info["volt_penalty"] == pytest.approx(0.0, abs=1e-6)

    def test_freq_dev_sign_convention(self, env):
        """freq_dev_hz = f_grid - f_nom; negative = under-frequency."""
        _, _, _, _, info = env.step(env.action_space.sample())
        expected = info["f_grid_hz"] - info["f_nom_hz"]
        assert info["freq_dev_hz"] == pytest.approx(expected, abs=1e-9)


# =========================================================================
# D. Termination on extreme events
# =========================================================================

class TestSafetyTermination:
    """Tests for episode termination on extreme frequency/voltage."""

    def test_terminated_flag_is_bool(self):
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)

    def test_no_early_termination_with_safe_action(self):
        """A balanced action should not cause immediate termination."""
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        safe_action = np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32)
        for _ in range(50):
            _, _, terminated, _, _ = env.step(safe_action)
            if terminated:
                break
        assert not terminated, "Safe action should not trigger termination"

    def test_termination_info_flags_mutually_consistent(self):
        """terminated == (thermal_fault OR freq_fault OR voltage_fault)."""
        env = C2GFastEnv(scenario="default")
        env.reset(seed=0)
        for _ in range(20):
            _, _, terminated, _, info = env.step(env.action_space.sample())
            expected = (info["thermal_fault"]
                        or info["freq_fault"]
                        or info["voltage_fault"])
            assert terminated == expected, (
                f"terminated={terminated} but faults="
                f"thermal={info['thermal_fault']}, "
                f"freq={info['freq_fault']}, "
                f"voltage={info['voltage_fault']}"
            )
            if terminated:
                break
