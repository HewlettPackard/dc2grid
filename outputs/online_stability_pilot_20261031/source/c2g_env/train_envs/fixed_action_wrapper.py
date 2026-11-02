"""
c2g_env/train_envs/fixed_cooling_wrapper.py
============================================
Gymnasium wrapper that holds arbitrary actions at fixed values while
exposing only the **free** (learnable) dimensions to the RL agent.

Usage
-----
  from c2g_env import C2GFastEnv
  from c2g_env.train_envs import FixedActionWrapper

  # Fix pump_speed_A (idx 1) and hvac_effort (idx 2)
  env = FixedActionWrapper(
      C2GFastEnv(scenario="default"),
      fixed_actions={1: 1.0, 2: 0.7},
  )
  assert env.action_space.shape == (2,)     # [throttle, bess]
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FixedActionWrapper(gym.ActionWrapper):
    """
    Reduce an N-D Box action space by pinning selected dimensions to
    constant values.  Only the *free* dimensions are exposed to the agent.

    Parameters
    ----------
    env : gym.Env
        Inner environment with a 1-D Box action space.
    fixed_actions : dict[int, float]
        Mapping from action index → fixed value.  Indices not present
        here remain learnable.  Values are clipped to the inner env's
        action bounds.
    """

    def __init__(
        self,
        env: gym.Env,
        fixed_actions: dict[int, float] | None = None,
    ) -> None:
        super().__init__(env)
        inner_space: spaces.Box = env.action_space
        n = inner_space.shape[0]

        fixed_actions = fixed_actions or {}
        self._fixed_indices: list[int] = sorted(fixed_actions.keys())
        self._free_indices: list[int] = [
            i for i in range(n) if i not in fixed_actions
        ]

        # Pre-compute the full template with fixed values clipped to bounds
        self._template = np.zeros(n, dtype=np.float32)
        for idx, val in fixed_actions.items():
            self._template[idx] = np.clip(
                val, inner_space.low[idx], inner_space.high[idx]
            )

        # Exposed action space: only the free dimensions
        self.action_space = spaces.Box(
            low=inner_space.low[self._free_indices],
            high=inner_space.high[self._free_indices],
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Expand reduced agent action → full env action."""
        full = self._template.copy()
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        full[self._free_indices] = act
        return full
