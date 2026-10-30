from __future__ import annotations

from importlib import import_module
from pathlib import Path
from time import perf_counter
from types import MethodType, SimpleNamespace

import gymnasium as gym
import numpy as np

from c2g_env.env_high_level import (
    _DEFAULT_BESS,
    _DEFAULT_HVAC,
    _DEFAULT_PUMP,
    _DEFAULT_THROTTLE,
)
from c2g_env.env_low_level import _FACILITY_CAP_KW, C2GFastEnv
from c2g_env.experiments.identity import file_fingerprint
from c2g_env.obs_indices import Fast
from c2g_env.physics.weather import WEATHER_PRESETS

from .config import REPO_ROOT
from .controllers import policy_identity
from .market import MARKETS, SLOT, hours, timestamp


class FixedActionController:
    def predict(self, observation, deterministic=True):
        return np.array(
            [_DEFAULT_THROTTLE, _DEFAULT_PUMP, _DEFAULT_HVAC, _DEFAULT_BESS],
            dtype=np.float32,
        ), None


class C2GHardwareAdapter:
    def __init__(self, env, seed: int):
        self.env = env
        self.plant, self.data, self.ledger = env.plant, env.data, env.ledger
        self.name = env.config["controllers"]["inner"]
        self.benchmark = import_module("evaluation.run_benchmark")
        self.observation_space = gym.spaces.Box(
            C2GFastEnv._OBS_LOW.copy(), C2GFastEnv._OBS_HIGH.copy(), dtype=np.float32
        )
        self.checkpoint = None
        self.normalization = None
        self.predict_calls = 0
        self.predict_seconds = 0.0
        self.max_predict_seconds = 0.0
        self.capped_steps = 0
        self.last_decision: dict = {}
        self._market_key = None
        self._market_context: dict = {}
        self._ambient_c = self.plant.state.ambient_c
        self._weather_preset = WEATHER_PRESETS[self.plant.scenario["weather_market"]]
        self.context = self._encoder_context()
        # The existing encoder needs only this state view, not a C2G environment.
        self._encode_state = MethodType(C2GFastEnv._build_obs, self.context)
        env.action_space.seed(seed)
        self.controller = self._load_controller(seed)

    def _encoder_context(self):
        electrical = self.plant.electrical
        capacity_kw = (
            electrical.n_racks_A * electrical.p_max_rack_A_kw
            + electrical.n_racks_B * electrical.p_max_rack_B_kw
        )
        reference_mw = (
            self.plant.scenario["committed_mw_max"] * capacity_kw / _FACILITY_CAP_KW
        )
        if not np.isfinite(reference_mw) or reference_mw <= 0:
            raise ValueError("The legacy tracking normalization must be positive.")
        return SimpleNamespace(
            action_space=self.env.action_space,
            _bess=self.plant.battery,
            _thermal=self.plant.thermal,
            _workload=self.plant.workload,
            _facility_cap_kw=capacity_kw,
            _committed_mw_max=reference_mw,
            _committed_mw=0.0,
            _prev_throttle=1.0,
            _prev_pump_speed=self.plant.thermal.PUMP_NOM_SPEED,
            _tick=0,
            _weather=SimpleNamespace(temp_norm=self._ambient_observation),
        )

    def _ambient_observation(self, tick) -> float:
        low = self._weather_preset.norm_min_c
        high = self._weather_preset.norm_max_c
        return float(np.clip((self._ambient_c - low) / (high - low), 0.0, 1.0))

    def _load_controller(self, seed: int):
        names = self.benchmark._INNER_CONTROLLERS | {"fixed"}
        if not isinstance(self.name, str) or self.name not in names:
            raise ValueError(
                f"Unknown inner controller {self.name!r}; choose from {sorted(names)}."
            )
        if self.name in {"ppo", "sac"}:
            return self._load_learned_controller(seed)
        if self.env.config["controllers"]["model_dir"] is not None:
            raise ValueError("model_dir is only valid for PPO or SAC controllers.")
        if self.name == "fixed":
            return FixedActionController()
        return self.benchmark._make_inner_controller(
            self.name,
            env=self.context,
            scenario=self.plant.config["scenario"],
            seed=seed,
        )

    def _load_learned_controller(self, seed: int):
        model_dir = self.env.config["controllers"]["model_dir"]
        scenario = self.plant.config["scenario"]
        directory = (
            REPO_ROOT / "trained_models" / f"{self.name}_{scenario}_s{seed}"
            if model_dir is None
            else (REPO_ROOT / Path(model_dir)).resolve()
        )
        checkpoint = directory / "final_model.zip"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required {self.name} checkpoint: {checkpoint}")
        cls, _, _ = self.benchmark._resolve_sb3_spec(self.name)
        model = cls.load(str(checkpoint), device="cpu")
        if model.observation_space != self.observation_space:
            raise ValueError("Checkpoint must use the C2G 18-field fast observation.")
        if model.action_space != self.env.action_space:
            raise ValueError("Checkpoint must use the unchanged four C2G actions.")
        normalizer = self._load_normalizer(directory / "vec_normalize.pkl")
        self.checkpoint = {
            **file_fingerprint(checkpoint),
            "saved_seed": model.seed,
            "saved_num_timesteps": model.num_timesteps,
        }
        return self.benchmark.SB3Agent(
            model,
            algo_name=self.name,
            obs_normalizer=normalizer,
            checkpoint=self.checkpoint,
        )

    def _load_normalizer(self, path: Path):
        if self.name == "ppo" and not path.is_file():
            raise FileNotFoundError(f"PPO requires its saved normalization: {path}")
        if not path.is_file():
            return None
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        observed = gym.wrappers.TransformObservation(
            self.env,
            lambda observation: self.observe(self.env.now)[0],
            observation_space=self.observation_space,
        )
        normalizer = VecNormalize.load(str(path), DummyVecEnv([lambda: observed]))
        normalizer.training = False
        normalizer.norm_reward = False
        self.normalization = file_fingerprint(path)
        return normalizer

    def market_context(self, when) -> dict:
        when = timestamp(when)
        slot = when.floor("15min")
        key = (slot, self.ledger.revision)
        if key != self._market_key:
            prices = {
                name: self.ledger.price(name, slot, slot + SLOT, when)
                for name in MARKETS
            }
            active = any(price is not None for price in prices.values())
            active |= any(
                trade.start < slot + SLOT and trade.end > slot
                for trade in self.ledger.trades.values()
            )
            target = self.ledger.energy(slot, slot + SLOT) / hours(slot, slot + SLOT)
            source, price = self._price_input(prices, when)
            self._market_context = {
                "target_import_mw": target if active else None,
                "price_eur_mwh": price,
                "price_source": source,
            }
            self._market_key = key
        return self._market_context

    def _price_input(self, prices, when):
        for name in reversed(MARKETS):
            if prices[name] is not None:
                return f"{name}_clearing", prices[name]
        if when >= self.env.score_start:
            raise RuntimeError("No live market price for scored hardware control.")
        return "supplied_IDM_forecast_warmup", self.data.forecast("IDM", when)

    def observe(self, when) -> tuple[np.ndarray, dict]:
        market = self.market_context(when)
        external = self.data.context(when)
        target = market["target_import_mw"]
        request = 0.0 if target is None else self.plant.state.facility_mw - target
        scale = min(
            self.context._committed_mw_max,
            max(self.plant.battery.P_MAX_MW, abs(request)),
        )
        signal = float(np.clip(request / scale, -1.0, 1.0))
        self.context._committed_mw = 0.0 if target is None else scale
        self.context._prev_throttle = float(self.plant.previous_action[0])
        self.context._prev_pump_speed = float(self.plant.previous_action[1])
        self._ambient_c = external["ambient_c"]
        observation = self._encode_state(
            self.plant.state.temp_a_c,
            self.plant.state.temp_b_c,
            {"soc_fraction": self.plant.state.soc},
            self.plant.workload_state,
            {
                "p_facility_mw": self.plant.state.facility_mw,
                "pue_dynamic": self.plant.state.pue,
            },
            {
                "regd_signal": signal,
                "lmp_usd_mwh": market["price_eur_mwh"],
                "load_norm": external["system_demand_mw"] / self.data.demand_scale_mw,
            },
            f_grid_hz=0.0,
            f_nom=0.0,
            v_pcc_pu=1.0,
        )
        if observation.shape != (Fast.DIM,) or not np.isfinite(observation).all():
            raise ValueError("The C2G adaptor must produce 18 finite observations.")
        return observation, {
            "timestamp": timestamp(when).isoformat(),
            **market,
            **external,
            "requested_reduction_mw": request,
            "represented_reduction_mw": signal * self.context._committed_mw,
            "request_clipped_mw": request - signal * self.context._committed_mw,
        }

    def act(self, when) -> np.ndarray:
        observation, diagnostics = self.observe(when)
        if self.name == "mpc_fast":
            self.controller.T_amb = diagnostics["ambient_c"]
        started = perf_counter()
        action, _ = self.controller.predict(observation, deterministic=True)
        elapsed = perf_counter() - started
        self.predict_calls += 1
        self.predict_seconds += elapsed
        self.max_predict_seconds = max(self.max_predict_seconds, elapsed)
        action = np.array(action, dtype=np.float32).reshape(-1)
        inner_throttle = float(action[0])
        cap = self.env.controllers.plan.cap(when)
        # The operating plan bounds batch service; thermal shedding can only lower it.
        action[0] = min(inner_throttle, cap)
        self.capped_steps += int(action[0] < inner_throttle)
        self.last_decision = {
            **diagnostics,
            "observation": observation,
            "predict_seconds": elapsed,
            "inner_throttle": inner_throttle,
            "schedule_cap": cap,
        }
        return action

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "implementation": policy_identity(self.controller),
            "observation_schema": "c2g-fast-18-market-tracking-v1",
            "action_interpretation": (
                "C2G actions; batch throttle capped by the L1/L2 operating plan"
            ),
            "tracking_request": (
                "gross facility minus position; tracking metrics use net import"
            ),
            "tracking_normalization_mw": self.context._committed_mw_max,
            "demand_normalization_mw": self.data.demand_scale_mw,
            "temperature_normalization_c": [
                self._weather_preset.norm_min_c,
                self._weather_preset.norm_max_c,
            ],
            "unmodeled_input_placeholders": {
                "frequency_deviation": 0.0,
                "voltage_pu": 1.0,
            },
            "price_currency": "EUR; legacy numeric scaling, no FX conversion",
            "checkpoint": self.checkpoint,
            "normalization": self.normalization,
            "learned_policy_interpretation": "transfer evaluation",
        }
