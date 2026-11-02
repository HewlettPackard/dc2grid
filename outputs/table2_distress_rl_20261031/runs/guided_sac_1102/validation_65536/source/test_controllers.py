# isort: off
import numpy as np
import pytest

from controllers import (
    CoolingFirstController,
    FullResidualMacro,
    battery_bounds,
    facility_power,
    liquid_temperature,
    observed_liquid_heat,
    pretransformer_power,
)
from c2g_env.physics.bess import _SimpleBESSModel
from c2g_env.physics.thermal import ThermalTwin
# isort: on


@pytest.mark.parametrize("power", [30.0, 80.0, 200.0, 400.0])
def test_transformer_inverse(power):
    assert pretransformer_power(facility_power(power)) == pytest.approx(power)


@pytest.mark.parametrize("pump", [0.15, 0.7, 1.0])
@pytest.mark.parametrize("heat", [16.0, 60.0, 150.0])
def test_liquid_model_and_observer(pump, heat):
    twin = ThermalTwin(dt_seconds=5)
    twin.temp_A, twin.T_amb = 32.2, 21.0
    actual = twin.step(heat, 30, 0.7, pump)[0][0]
    predicted = liquid_temperature(32.2, heat, pump, 21)
    assert actual == pytest.approx(predicted, abs=1e-12)
    recovered = observed_liquid_heat(32.2, predicted, pump, 21)
    assert recovered == pytest.approx(heat, abs=1e-8)


@pytest.mark.parametrize("soc", [0.30, 0.30001, 0.5, 0.79999, 0.8])
def test_battery_reserve_after_real_step(soc):
    for dispatch in battery_bounds(soc):
        bess = _SimpleBESSModel(dt_seconds=5)
        bess.set_initial_soc(soc)
        result = bess.step(dispatch)
        assert 0.30 - 1e-12 <= result["soc_fraction"] <= 0.80 + 1e-12


def test_macro_rounds_inside_true_offer():
    controller = FullResidualMacro()
    for residual in np.linspace(5.01, 40, 1000):
        obs = np.zeros(19, dtype=np.float32)
        obs[18] = residual / 30
        action, _ = controller.predict(obs)
        assert 5 <= float(action[0]) * 30 <= min(30, residual)
        assert action[1] == 0


def test_no_action_without_measurement():
    with pytest.raises(RuntimeError, match="Observe"):
        CoolingFirstController().action()


def test_invalid_observation_is_explicit():
    with pytest.raises(ValueError, match="18D"):
        CoolingFirstController().features(np.zeros(17))
