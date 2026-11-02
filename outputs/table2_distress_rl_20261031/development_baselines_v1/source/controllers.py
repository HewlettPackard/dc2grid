"""Causal cooling-first control and bounded residual coordinates.

The controller receives only the original 18 observations and its own history.
Its nominal liquid-loop observer is reconstructed from measured temperatures;
it never reads the simulator's independent reference or future inputs.
"""

from __future__ import annotations

import math

import common  # noqa: F401
import numpy as np

from c2g_env.obs_indices import Fast
from c2g_env.physics.weather import WEATHER_PRESETS

NOMINAL = float(np.float32(0.7))
DT = 5.0
FEATURE_COUNT = 24


def pretransformer_power(facility_mw: float) -> float:
    if not math.isfinite(facility_mw) or facility_mw <= 0.15:
        raise ValueError("Invalid positive facility power")
    copper = 0.006 / 300.0
    net = facility_mw - 0.15
    return 2 * net / (1 + math.sqrt(1 + 4 * copper * net))


def facility_power(pretransformer_mw: float) -> float:
    return (
        pretransformer_mw
        + 0.15
        + 0.006 * 300 * min(pretransformer_mw / 300, 1.5) ** 2
    )


def liquid_temperature(
    temperature: float, heat_mw: float, pump: float, ambient: float
) -> float:
    conductance = 35 * max(0.15, pump)
    total = conductance + 0.5
    equilibrium = (heat_mw + conductance * 30 + 0.5 * ambient) / total
    return equilibrium + (temperature - equilibrium) * math.exp(
        -total * DT / 27000
    )


def observed_liquid_heat(
    before: float, after: float, pump: float, ambient: float
) -> float:
    conductance = 35 * max(0.15, pump)
    total = conductance + 0.5
    decay = math.exp(-total * DT / 27000)
    equilibrium = (after - before * decay) / (1 - decay)
    return equilibrium * total - conductance * 30 - 0.5 * ambient


def battery_bounds(soc: float) -> tuple[float, float]:
    if not math.isfinite(soc) or not 0.1 <= soc <= 0.95:
        raise ValueError("SOC outside the frozen battery support")
    hours = DT / 3600
    discharge = min(5.0, max(0.0, soc - 0.30) * 15 * 0.70 / hours)
    charge = min(5.0, max(0.0, 0.80 - soc) * 15 / hours)
    return -charge, discharge


class CoolingFirstController:
    """Observation-only controller; zero residual is the nonlearned ablation."""

    def __init__(self, *, transformer_compensation: bool = True):
        self.transformer_compensation = transformer_compensation
        self.reset()

    def reset(self):
        self.previous_observation = None
        self.observation = None
        self.last_action = np.array([1, NOMINAL, NOMINAL, 0], dtype=np.float32)
        self.reference_temperature_A = 30.0
        self.estimated_heat_A = 60.0
        self.filtered_request = 0.0
        self.previous_request = 0.0
        self.tick = 0
        self.last_diagnostics = {}

    @staticmethod
    def ambient(observation: np.ndarray) -> float:
        preset = WEATHER_PRESETS["nyiso_nyc"]
        value = float(observation[Fast.T_AMB])
        if not 0 < value < 1:
            raise ValueError("Clipped ambient observation cannot be inverted")
        return preset.norm_min_c + value * (
            preset.norm_max_c - preset.norm_min_c
        )

    def features(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (18,) or not np.isfinite(observation).all():
            raise ValueError("Expected one finite original 18D observation")
        temperature = float(observation[Fast.TEMP_A]) * 35
        if self.observation is None:
            self.reference_temperature_A = temperature
        else:
            before = float(self.observation[Fast.TEMP_A]) * 35
            heat = observed_liquid_heat(
                before,
                temperature,
                float(self.last_action[1]),
                self.ambient(self.observation),
            )
            if not -0.1 <= heat <= 150.1:
                raise ValueError(f"Liquid observer heat outside nameplate: {heat}")
            self.estimated_heat_A = min(max(heat, 0), 150)
            self.reference_temperature_A = liquid_temperature(
                self.reference_temperature_A,
                self.estimated_heat_A,
                NOMINAL,
                self.ambient(self.observation),
            )
            self.tick += 1
        request = (
            float(observation[Fast.COMMITTED])
            * 30
            * float(observation[Fast.REGD])
        )
        request_change = request - self.previous_request
        self.filtered_request += (1 - math.exp(-DT / 900)) * (
            request - self.filtered_request
        )
        self.previous_request = request
        self.previous_observation = self.observation
        self.observation = observation.copy()
        return np.r_[
            observation,
            (self.reference_temperature_A - temperature) / 5,
            self.estimated_heat_A / 150,
            self.last_action[2],
            self.filtered_request / 30,
            request_change / 60,
            max(0, 1 - self.tick / 17280),
        ].astype(np.float32)

    def action(self, residual: np.ndarray | None = None) -> np.ndarray:
        if self.observation is None:
            raise RuntimeError("Observe the current state before selecting an action")
        if residual is None:
            residual = np.zeros(3, dtype=np.float32)
        residual = np.asarray(residual)
        if (
            residual.shape != (3,)
            or not np.isfinite(residual).all()
            or (np.abs(residual) > 1).any()
        ):
            raise ValueError("Residual coordinates must be finite and in [-1,1]")
        observation = self.observation
        ta = float(observation[Fast.TEMP_A]) * 35
        tb = float(observation[Fast.TEMP_B]) * 35
        ambient = self.ambient(observation)
        soc = float(observation[Fast.SOC])
        request = float(observation[Fast.COMMITTED]) * 30 * float(
            observation[Fast.REGD]
        )
        pump = float(
            np.clip(
                NOMINAL + max(0, ta - 32.0) * 0.3 + 0.15 * residual[0],
                0.15,
                1,
            )
        )
        soc_target = 0.50 + 0.10 * float(residual[1])
        lower_battery, upper_battery = battery_bounds(soc)
        battery = float(np.clip(60 * (soc - soc_target), lower_battery, upper_battery))
        desired_relief = request - battery + float(residual[2])
        reference_next = liquid_temperature(
            self.reference_temperature_A, self.estimated_heat_A, NOMINAL, ambient
        )
        actual_next = liquid_temperature(ta, self.estimated_heat_A, pump, ambient)
        reference_cool = (
            35 * NOMINAL * max(0, reference_next - 30) / 30 + 1.5 * NOMINAL
        )
        actual_cool = 35 * pump * max(0, actual_next - 30) / 30 + 1.5 * pump
        previous_cool = (
            35 * float(self.last_action[1]) * max(0, ta - 30) / 30
            + 1.5 * float(self.last_action[1])
        )
        observed_power = float(observation[Fast.P_FAC]) * 250
        reference_pre = (
            pretransformer_power(observed_power)
            + reference_cool
            - previous_cool
            + 50 * (NOMINAL - float(self.last_action[2]))
        )
        if self.transformer_compensation:
            target_pre = pretransformer_power(
                facility_power(reference_pre) - desired_relief
            )
            cooling_change = target_pre - reference_pre
        else:
            cooling_change = -desired_relief
        unconstrained_hvac = NOMINAL + (
            cooling_change - (actual_cool - reference_cool)
        ) / 50
        thermal_floor = float(np.clip(2 * (tb - 32.0), 0, 1))
        hvac = float(np.clip(unconstrained_hvac, thermal_floor, 1))
        if hvac != unconstrained_hvac:
            before = reference_pre + actual_cool - reference_cool + 50 * (
                unconstrained_hvac - NOMINAL
            )
            after = reference_pre + actual_cool - reference_cool + 50 * (
                hvac - NOMINAL
            )
            battery += facility_power(after) - facility_power(before)
        battery = float(np.clip(battery, lower_battery, upper_battery))
        action = np.array([1.0, pump, hvac, battery / 5], dtype=np.float32)
        self.last_action = action.copy()
        self.last_diagnostics = {
            "estimated_heat_A_mw": self.estimated_heat_A,
            "observer_reference_temperature_A": self.reference_temperature_A,
            "requested_mw": request,
            "soc_target": soc_target,
            "thermal_hvac_floor": thermal_floor,
            "unconstrained_hvac": unconstrained_hvac,
            "residual_pump": float(residual[0]),
            "residual_reserve": float(residual[1]),
            "residual_tracking": float(residual[2]),
        }
        return action

    def predict(self, observation, deterministic=True):
        self.features(observation)
        return self.action(), None


class ResidualController:
    def __init__(self, model, *, transformer_compensation: bool = True):
        self.model = model
        self.base = CoolingFirstController(
            transformer_compensation=transformer_compensation
        )

    def reset(self):
        self.base.reset()

    def predict(self, observation, deterministic=True):
        features = self.base.features(observation)
        residual = self.model.predict(features, deterministic=deterministic)[0]
        return self.base.action(residual), None


class FullResidualMacro:
    def predict(self, observation, deterministic=True):
        residual_mw = float(observation[18]) * 30
        quantity = np.nextafter(
            np.float32(min(30.0, residual_mw) / 30), np.float32(0)
        )
        if float(quantity) * 30 < 5:
            raise ValueError("Posted residual is below the fixed 5MW lower bound")
        return np.array([quantity, 0], dtype=np.float32), None
