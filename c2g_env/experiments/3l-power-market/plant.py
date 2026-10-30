from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, replace

import numpy as np
import yaml

from c2g_env.env_low_level import (
    _PLANT_BESS_KEYS,
    _PLANT_ELEC_KEYS,
    _PLANT_RACK_KEYS,
    _PLANT_THERMAL_KEYS,
    _THERMAL_OVERRIDE_KEYS,
    C2GFastEnv,
)
from c2g_env.physics.bess import _SimpleBESSModel
from c2g_env.physics.electrical import DatacenterElectrical
from c2g_env.physics.thermal import ThermalTwin
from c2g_env.physics.workload import WorkloadOrchestrator

from .config import REPO_ROOT


@dataclass(frozen=True)
class PlantState:
    temp_a_c: float
    temp_b_c: float
    soc: float
    base_a_mw: float
    base_b_mw: float
    batch_arrived_mw: float
    batch_served_mw: float
    backlog_work_mwh: float
    facility_mw: float
    battery_mw: float
    net_import_mw: float
    cooling_mw: float
    pue: float
    ambient_c: float


class C2GPlant:
    def __init__(self, config: dict):
        self.config = config
        with (REPO_ROOT / config["config"]).open(encoding="utf-8") as handle:
            source = yaml.safe_load(handle)
        self.global_config = source["global"]
        self.scenario = source[config["scenario"]]
        self.reward_config = source["reward"]
        self.dt_seconds = float(self.global_config["dt_seconds"])
        if self.dt_seconds <= 0 or 900 % self.dt_seconds != 0:
            raise ValueError("The C2G physical timestep must divide 900 seconds.")
        self.plant_overrides = config["plant_overrides"] or self.scenario.get(
            "plant", {}
        )
        self.thermal_overrides = config["thermal_overrides"]
        self._validate_overrides()
        self._state: PlantState | None = None

    @property
    def state(self) -> PlantState:
        if self._state is None:
            raise RuntimeError("Reset the plant before accessing its state.")
        return self._state

    def _validate_overrides(self) -> None:
        plant_keys = (
            _PLANT_RACK_KEYS | _PLANT_BESS_KEYS | _PLANT_ELEC_KEYS | _PLANT_THERMAL_KEYS
        )
        unknown = self.plant_overrides.keys() - plant_keys
        unknown |= self.thermal_overrides.keys() - (_THERMAL_OVERRIDE_KEYS - {"T_amb"})
        if unknown:
            raise ValueError(f"Unsupported DC-only override(s): {sorted(unknown)}")
        positive = {
            "C_A",
            "C_B",
            "K_liq",
            "K_air",
            "COP_base",
            "E_NOM_MWH",
            "P_MAX_MW",
            "n_racks_a_flex",
            "S_xfmr_mva",
        }
        for key, value in {**self.plant_overrides, **self.thermal_overrides}.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Physical override {key} must be finite and nonnegative."
                )
            if key in positive and value == 0:
                raise ValueError(f"Physical override {key} must be positive.")
            if key in _PLANT_RACK_KEYS and int(value) != value:
                raise ValueError(f"Rack count {key} must be an integer.")

    def reset(self, seed: int, ambient_c: float) -> PlantState:
        self._check_ambient(ambient_c)
        trace_dir = REPO_ROOT / self.global_config["trace_dir"]
        for name in WorkloadOrchestrator._TRACE_FILES.values():
            if not (trace_dir / name).is_file():
                raise FileNotFoundError(
                    f"Required workload trace is missing: {trace_dir / name}"
                )
        racks = {
            key: self.plant_overrides[key]
            for key in _PLANT_RACK_KEYS
            if key in self.plant_overrides
        }
        self.workload = WorkloadOrchestrator(
            trace_dir=trace_dir, dt_seconds=self.dt_seconds, seed=seed, **racks
        )
        self.thermal = ThermalTwin(dt_seconds=self.dt_seconds)
        self.thermal.T_safe = float(self.global_config["T_safe"])
        initial = min(ambient_c + 5.0, self.thermal.T_safe - 1.0)
        self.thermal.reset(temp_A=initial, temp_B=initial)
        self.electrical = DatacenterElectrical()
        self.electrical.reset()
        # Optional PySAM installation must not change this experiment's battery.
        self.battery = _SimpleBESSModel(dt_seconds=self.dt_seconds)
        self.battery.reset()
        self._apply_overrides()
        self.battery.set_initial_soc(float(self.scenario["bess_soc_init"]))
        self.initial_soc = self.battery.soc_fraction
        self._state = None
        self.previous_action = self.nominal_action()
        preview = deepcopy(self)
        estimate = preview.step(self.previous_action, ambient_c)
        self.workload_state = replace(preview.workload_state, backlog_kw=0.0)
        self._state = replace(
            estimate,
            temp_a_c=initial,
            temp_b_c=initial,
            soc=self.initial_soc,
            batch_served_mw=0.0,
            backlog_work_mwh=0.0,
            battery_mw=0.0,
            net_import_mw=estimate.facility_mw,
        )
        return self.state

    def _apply_overrides(self) -> None:
        self.electrical.n_racks_A = (
            self.workload._n_racks_a_flex + self.workload._n_racks_a_base
        )
        self.electrical.n_racks_B = self.workload._n_racks_b
        for engine, keys in (
            (self.electrical, _PLANT_ELEC_KEYS),
            (self.thermal, _PLANT_THERMAL_KEYS),
            (self.battery, _PLANT_BESS_KEYS),
        ):
            for key in keys & self.plant_overrides.keys():
                setattr(engine, key, float(self.plant_overrides[key]))
        self.thermal.set_cooling_fault(
            active=self.scenario.get("cooling_fault", False),
            fault_factor=self.scenario.get("cooling_fault_factor", 0.4),
        )
        for key, value in self.thermal_overrides.items():
            setattr(self.thermal, key, float(value))
        if "fault_factor" in self.thermal_overrides:
            self.thermal.fault_active = self.thermal.fault_factor < 1.0

    @property
    def batch_capacity_mw(self) -> float:
        return self.workload.p_flex_max_kw / 1000.0

    def nominal_action(self) -> np.ndarray:
        return np.array(
            [1.0, self.thermal.PUMP_NOM_SPEED, self.thermal.HVAC_NOM_EFFORT, 0.0],
            dtype=np.float32,
        )

    @staticmethod
    def _check_ambient(ambient_c: float) -> None:
        if not np.isfinite(ambient_c):
            raise ValueError("A finite ambient temperature from 3L is required.")

    def step(self, action: np.ndarray, ambient_c: float) -> PlantState:
        self._check_ambient(ambient_c)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Hardware actions must contain four finite values.")
        action = np.clip(action, C2GFastEnv._ACT_LOW, C2GFastEnv._ACT_HIGH)
        work = self.workload.step(float(action[0]))
        self.workload_state = work
        self.previous_action = action.copy()
        battery = self.battery.step(float(action[3]) * self.battery.P_MAX_MW)
        electrical = self._facility(
            work.p_base_a_kw / 1000,
            work.p_base_b_kw / 1000,
            work.p_flex_kw / 1000,
            float(action[1]),
            float(action[2]),
            ambient_c,
        )
        self._state = PlantState(
            self.thermal.temp_A,
            self.thermal.temp_B,
            battery["soc_fraction"],
            work.p_base_a_kw / 1000,
            work.p_base_b_kw / 1000,
            work.p_flex_nom_kw / 1000,
            work.p_flex_kw / 1000,
            work.backlog_kw * self.dt_seconds / 3_600_000,
            electrical["p_facility_mw"],
            battery["actual_power_mw"],
            electrical["p_facility_mw"] - battery["actual_power_mw"],
            electrical["p_cooling_mw"],
            electrical["pue_dynamic"],
            ambient_c,
        )
        return self.state

    def _facility(self, base_a, base_b, batch_mw, pump, hvac, ambient_c) -> dict:
        self.thermal.T_amb = ambient_c
        _, (liquid, air, pump_mw) = self.thermal.step(
            p_it_A_mw=base_a + batch_mw,
            p_it_B_mw=base_b,
            hvac_effort=hvac,
            pump_speed=pump,
        )
        util_a = self.electrical._inverse_rack_util(
            (base_a + batch_mw) * 1000,
            self.electrical.n_racks_A,
            self.electrical.p_idle_rack_A_kw,
            self.electrical.p_max_rack_A_kw - self.electrical.p_idle_rack_A_kw,
            self.electrical.alpha_A,
        )
        util_b = self.electrical._inverse_rack_util(
            base_b * 1000,
            self.electrical.n_racks_B,
            self.electrical.p_idle_rack_B_kw,
            self.electrical.p_max_rack_B_kw - self.electrical.p_idle_rack_B_kw,
            self.electrical.alpha_B,
        )
        result = self.electrical.step(util_a, util_b, liquid, air + pump_mw)
        return {
            key: value
            for key, value in result.items()
            if key not in {"v_pcc_pu", "v_drop_pu"}
        }

    def copy_physics(self) -> C2GPlant:
        """Isolate facility dynamics; forecasts must not step workload or BESS."""
        preview = copy(self)
        preview.thermal = deepcopy(self.thermal)
        preview.electrical = deepcopy(self.electrical)
        return preview
