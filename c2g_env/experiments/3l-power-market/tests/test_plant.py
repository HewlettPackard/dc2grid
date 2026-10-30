from dataclasses import asdict

import numpy as np
import pytest

from c2g_env.env_low_level import C2GFastEnv


def test_physical_parity_with_existing_env(plant):
    legacy = C2GFastEnv()
    legacy._scfg = {**legacy._scfg, "T_amb": 7.0, "weather_driven": False}
    legacy.reset(seed=0)
    for action in (
        [1.0, 1.0, 0.7, 0.0],
        [0.0, 0.4, 0.9, 0.3],
        [0.5, 0.8, 0.6, -0.8],
        [1.0, 0.9, 1.0, 0.0],
    ):
        action = np.array(action, dtype=np.float32)
        _, _, _, _, info = legacy.step(action)
        state = plant.step(action, 7.0)
        assert state.temp_a_c == pytest.approx(info["temp_A"], rel=0, abs=1e-10)
        assert state.temp_b_c == pytest.approx(info["temp_B"], rel=0, abs=1e-10)
        assert state.soc == pytest.approx(info["bess_soc"], rel=0, abs=1e-10)
        assert state.facility_mw == pytest.approx(
            info["p_facility_mw"], rel=0, abs=1e-10
        )
        assert state.battery_mw == pytest.approx(
            info["bess_actual_kw"] / 1000, rel=0, abs=1e-10
        )
        assert state.batch_served_mw == pytest.approx(
            info["p_flex_served_kw"] / 1000, rel=0, abs=1e-10
        )
        assert state.backlog_work_mwh == pytest.approx(
            info["backlog_kw"] * plant.dt_seconds / 3_600_000, rel=0, abs=1e-10
        )
    legacy.close()


def test_wrapper_never_constructs_c2g_grid_or_weather(modules, config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External C2G context must not be constructed")

    monkeypatch.setattr("c2g_env.env_low_level.MacroGridSignal.__init__", forbidden)
    monkeypatch.setattr("c2g_env.env_low_level.WeatherLoader.__init__", forbidden)
    plant = modules.plant.C2GPlant(config["c2g"])
    initial = plant.reset(2, 4.0)
    assert initial.temp_a_c == 9.0
    assert initial.ambient_c == 4.0
    state = plant.step(plant.nominal_action(), 8.0)
    assert state.ambient_c == 8.0
    assert not any("voltage" in key or "frequency" in key for key in asdict(state))


@pytest.mark.parametrize("action", ([0, 1], [float("nan"), 0, 0, 0]))
def test_invalid_actions_are_errors(plant, action):
    with pytest.raises(ValueError, match="four finite"):
        plant.step(np.array(action), 7.0)


def test_actual_battery_power_defines_meter(plant):
    plant.battery.set_initial_soc(plant.battery.SOC_MIN)
    state = plant.step(np.array([1.0, 1.0, 0.7, 1.0]), 7.0)
    assert state.battery_mw == pytest.approx(0.0)
    assert state.net_import_mw == state.facility_mw
    charged = plant.step(np.array([1.0, 1.0, 0.7, -1.0]), 7.0)
    assert charged.net_import_mw > charged.facility_mw


def test_thermal_overrides_compose_but_ambient_cannot(modules, config):
    config["c2g"]["thermal_overrides"] = {"C_A": 30_000}
    plant = modules.plant.C2GPlant(config["c2g"])
    plant.reset(0, 5.0)
    assert plant.thermal.C_A == 30_000
    config["c2g"]["thermal_overrides"]["T_amb"] = 25
    with pytest.raises(ValueError, match="T_amb"):
        modules.plant.C2GPlant(config["c2g"])


def test_zero_physical_overrides_and_fractional_racks(modules, config):
    config["c2g"]["plant_overrides"] = {"n_racks_a_base": 0, "n_racks_b": 0}
    config["c2g"]["thermal_overrides"] = {"K_env_A": 0}
    plant = modules.plant.C2GPlant(config["c2g"])
    plant.reset(0, 7.0)
    state = plant.step(plant.nominal_action(), 7.0)
    assert np.isfinite(state.facility_mw)
    assert plant.thermal.K_env_A == 0
    config["c2g"]["plant_overrides"]["n_racks_b"] = 0.5
    with pytest.raises(ValueError, match="integer"):
        modules.plant.C2GPlant(config["c2g"])
