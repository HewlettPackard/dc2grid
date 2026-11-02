"""Finite 24-hour standalone learning task under the unchanged rule macro."""

from __future__ import annotations

from common import cap_cpu
from history_policy import (
    ACTION_HIGH,
    ACTION_LOW,
    FEATURE_COUNT,
    OwnHistory,
    hardware_action,
)

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from c2g_env import C2GFastEnv
from c2g_env.train_envs import RuleMacroWrappedEnv
from macro_task import Ledger


class StandaloneTask(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        protocol: dict,
        *,
        seed: int,
        rank: int = 0,
        fixed_cell: dict | None = None,
    ):
        super().__init__()
        self.protocol = protocol
        self.seed_key, self.rank, self.fixed_cell = seed, rank, fixed_cell
        self.episode_ordinal = 0
        self.fast = C2GFastEnv(
            scenario="default",
            simulation_mode="physical_v1",
            config_path=protocol["config_path"],
        )
        self.env = RuleMacroWrappedEnv(self.fast)
        self.controller = OwnHistory()
        self.action_space = spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(FEATURE_COUNT,), dtype=np.float32
        )
        self.observation_space.c2g_simulation_mode = "physical_v1"
        self.configured_ticks = self.fast._episode_ticks
        if self.configured_ticks != 17280 or self.fast._dt != 5:
            raise ValueError("Standalone learning requires the frozen service cadence")
        self.done = True
        self.last_episode = None

    def condition(self):
        if self.fixed_cell is not None:
            return dict(self.fixed_cell)
        days, biases = self.protocol["train_days"], self.protocol["train_biases"]
        slot = self.episode_ordinal * self.protocol["n_envs"] + self.rank
        cycle, position = divmod(slot, len(days))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed_key, 713, cycle]))
        day_index = int(rng.permutation(len(days))[position])
        bias = biases[(position + cycle) % len(biases)]
        seed = int(
            np.random.SeedSequence([self.seed_key, 157, cycle, position])
            .generate_state(1)[0]
        )
        forbidden_seeds = set(range(90001, 90005)) | {
            91001, 91002, 92001, 92002, 92003, 100, 101, 102, 103, 104
        }
        if seed in forbidden_seeds:
            raise RuntimeError("Training seed collided with an evaluation seed")
        return {
            "start_day": days[day_index],
            "regd_load_bias": bias,
            "seed": seed,
            "curriculum_cycle": cycle,
            "curriculum_position": position,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Episode conditions are controlled by the study protocol")
        self.cell = self.condition()
        permitted = set(self.protocol["train_days"])
        if self.fixed_cell is None and self.cell["start_day"] not in permitted:
            raise ValueError("Condition is outside the training allowlist")
        self.fast._episode_ticks = self.configured_ticks
        native, info = self.env.reset(
            seed=self.cell["seed"],
            options={key: self.cell[key] for key in ("start_day", "regd_load_bias")},
        )
        self.controller.reset()
        self.features = self.controller.features(native)
        self.ledger = Ledger()
        self.steps = 0
        self.reward_sum = 0.0
        self.episode_ordinal += 1
        self.done = False
        self.last_episode = None
        return self.features.copy(), {**info, "condition": self.cell}

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset before stepping a completed task")
        native = hardware_action(action)
        self.controller.last_action = native.copy()
        observation, reward, term, trunc, info = self.env.step(native)
        self.ledger.update(self.fast, info, self.protocol["normalization"])
        self.steps += 1
        self.reward_sum += reward
        self.features = self.controller.features(observation)
        self.done = bool(term or trunc)
        if trunc and not term and self.steps != self.configured_ticks:
            raise RuntimeError("Service horizon disagreement")
        if self.done:
            ledger = self.ledger.report()
            gate = self.protocol["service_gate"]
            complete = bool(trunc and not term and self.steps == self.configured_ticks)
            self.last_episode = {
                "condition": self.cell,
                "steps": self.steps,
                "complete": complete,
                "physical_terminated": bool(term),
                "mean_original_reward": self.reward_sum / self.steps,
                "service": ledger,
                "terminal_soc": self.fast._bess.soc_fraction,
                "service_qualified": bool(
                    complete
                    and self.fast._bess.soc_fraction >= gate["min_terminal_soc"]
                    and ledger["backlog_kwh"]
                    <= gate["max_backlog_fraction"] * ledger["work_arrived_kwh"] + 1e-6
                    and ledger["thermal_warning_fraction"]
                    <= gate["max_thermal_warning_fraction"]
                ),
                "recovery_not_performed_in_training": True,
            }
        return (
            self.features.copy(), float(reward), self.done, False,
            {
                "condition": self.cell,
                "native_action": native,
                "original_reward": float(reward),
                "physical_terminated": bool(term),
                "service_truncated": bool(trunc),
                "episode_result": self.last_episode,
                "tracking_err_kw": info["tracking_err_kw"],
                "bess_soc": info["bess_soc"],
            },
        )

    def training_prefix(self):
        return {
            "condition": self.cell,
            "steps": self.steps,
            "done": self.done,
            "service": self.ledger.report() if self.steps else None,
            "not_a_complete_episode": not self.done,
        }

    def close(self):
        self.env.close()


def make_training_worker(protocol, seed, rank, cpu):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(cpu)
    return StandaloneTask(protocol, seed=seed, rank=rank)
