"""Reuse the baseline task, adding reward diagnostics without changing its reward."""

from __future__ import annotations

from pathlib import Path

from study import cap_cpu

# isort: off
import gymnasium as gym
import numpy as np
from fast_task import StandaloneTask
# isort: on

COMPONENTS = (
    "reward_throughput",
    "reward_tracking",
    "reward_thermal",
    "reward_soc",
    "reward_freq",
    "reward_volt",
    "reward_backlog",
)


class CaptureStepInfo(gym.Wrapper):
    def step(self, action):
        transition = self.env.step(action)
        self.last_info = transition[-1]
        return transition


class DiagnosticTask(StandaloneTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = CaptureStepInfo(self.env)
        self.transition_records = []
        self.reward_alpha = float(self.fast._rcfg["alpha"])
        if self.reward_alpha <= 0:
            raise ValueError("Throughput diagnostics require positive reward alpha")

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.scale_constants = {
            "hvac_nameplate_kw": self.fast._thermal.max_hvac_mw * 1000,
            "committed_max_mw": self.fast._committed_mw_max,
            "tracking_floor_kw": self.fast._tracking_floor_kw,
            "tracking_reward_beta": self.fast._rcfg["beta"],
        }
        if self.scale_constants != {
            "hvac_nameplate_kw": 50000,
            "committed_max_mw": 30,
            "tracking_floor_kw": 100,
            "tracking_reward_beta": 2,
        }:
            raise ValueError("Registered prior-scale diagnostic constants changed")
        forbidden = {
            cell["seed"]
            for role in ("development_cells", "final_cells", "regression_cells")
            for cell in self.protocol.get(role, [])
        }
        if self.cell["seed"] in forbidden:
            raise ValueError("Training noise seed collides with an evaluation seed")
        self.component_sum = np.zeros(len(COMPONENTS), dtype=np.float64)
        self.component_squares = np.zeros(len(COMPONENTS), dtype=np.float64)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        components = np.array(
            [self.env.last_info[name] for name in COMPONENTS], dtype=np.float64
        )
        if not np.isclose(components.sum(), reward, atol=1e-12, rtol=1e-12):
            raise AssertionError("Diagnostic reward components changed the reward")
        self.component_sum += components
        self.component_squares += components**2
        raw = self.env.last_info
        backlog_ratio = raw["backlog_kw"] / self.fast._workload.work_capacity_kw
        self.transition_records.append(
            (
                self.episode_ordinal - 1,
                self.cell["start_day"],
                self.cell["regd_load_bias"],
                self.cell["seed"],
                self.steps,
                *components,
                backlog_ratio,
                float(backlog_ratio >= 1),
                float(components[0] >= self.reward_alpha - 1e-12),
                float(raw["thermal_fault"]),
                float(raw["freq_fault"]),
                float(raw["voltage_fault"]),
                float(terminated),
                float(truncated),
                float(info["physical_terminated"]),
                float(info["service_truncated"]),
                float(observation[-1]),
            )
        )
        if self.last_episode is not None:
            means = self.component_sum / self.steps
            variance = np.maximum(self.component_squares / self.steps - means**2, 0)
            self.last_episode["reward_components"] = {
                name: {
                    "mean": float(means[index]),
                    "sd": float(np.sqrt(variance[index])),
                }
                for index, name in enumerate(COMPONENTS)
            }
            self.last_episode["fault_channels"] = {
                name: bool(raw[name])
                for name in ("thermal_fault", "freq_fault", "voltage_fault")
            }
            self.last_episode["terminal_clock"] = float(observation[-1])
        return observation, reward, terminated, truncated, info

    def save_diagnostics(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("Refusing to overwrite transition diagnostics")
        columns = [
            "episode_ordinal",
            "start_day",
            "regd_load_bias",
            "env_seed",
            "step",
            *COMPONENTS,
            "post_backlog_over_work_capacity",
            "post_backlog_above_capacity",
            "throughput_reward_saturated",
            "thermal_fault",
            "freq_fault",
            "voltage_fault",
            "task_terminated",
            "task_truncated",
            "physical_terminated",
            "service_truncated",
            "next_clock",
        ]
        np.savez_compressed(
            path,
            columns=np.array(columns),
            values=np.asarray(self.transition_records, dtype=np.float64),
            reward_alpha=np.asarray(self.reward_alpha),
        )
        return {"path": str(path), "transitions": len(self.transition_records)}

    def diagnostic_constants(self):
        return {**self.scale_constants, "throughput_reward_alpha": self.reward_alpha}


def make_worker(protocol, seed, rank, cpu):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cap_cpu(cpu)
    return DiagnosticTask(protocol, seed=seed, rank=rank)
