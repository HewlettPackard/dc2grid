"""
c2g_env.experiments.action_ablation_env
========================================
ActionAblationFastEnv — a subclass of C2GFastEnv that supports
fixed-action ablation studies.

During ``step()``, analyst-chosen fixed setpoints can override
selected action dimensions regardless of what the agent requests.

Usage
-----
  from c2g_env.experiments import ActionAblationFastEnv

  env = ActionAblationFastEnv(
      scenario="default",
      fixed_action_values={"bess_dispatch": 0.0, "hvac_effort": 0.8},
  )
  obs, info = env.reset(seed=0)
  obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # fixed_action_values controls the applied setpoints per action name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from c2g_env.env_low_level import C2GFastEnv

# ---------------------------------------------------------------------------
# Action metadata (mirrors the structure that was previously inlined in
# C2GFastEnv but now lives exclusively here so the base env stays clean)
# ---------------------------------------------------------------------------

_ACTION_NAMES: tuple[str, ...] = (
    "throttle_batch",
    "pump_speed_A",
    "hvac_effort",
    "bess_dispatch",
)

_ACTION_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(_ACTION_NAMES)}

_ACTION_BOUNDS: dict[str, tuple[float, float]] = {
    "throttle_batch": (0.0, 1.0),
    "pump_speed_A":   (0.0, 1.0),
    "hvac_effort":    (0.0, 1.0),
    "bess_dispatch":  (-1.0, 1.0),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _normalise_fixed_action_values(
    fixed_action_values: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate and clip fixed action values to their declared bounds."""
    if fixed_action_values is None:
        return {}

    cleaned: dict[str, float] = {}
    invalid: list[str] = []
    for name, value in fixed_action_values.items():
        if name not in _ACTION_BOUNDS:
            invalid.append(name)
            continue
        low, high = _ACTION_BOUNDS[name]
        cleaned[name] = float(np.clip(float(value), low, high))

    if invalid:
        valid = ", ".join(_ACTION_NAMES)
        bad   = ", ".join(sorted(str(n) for n in invalid))
        raise ValueError(f"Unknown fixed_action_values keys: {bad}. Valid actions: {valid}")

    return cleaned


# ---------------------------------------------------------------------------
# Subclass
# ---------------------------------------------------------------------------

class ActionAblationFastEnv(C2GFastEnv):
    """
    C2GFastEnv with fixed-action ablation support.

    All physics, reward, observation, and termination logic is inherited
    unchanged from C2GFastEnv. This subclass only intercepts ``step()``
    to apply fixed-action overrides before forwarding to the parent.

    Parameters
    ----------
    scenario : str
        Passed through to C2GFastEnv.
    config_path : str or Path, optional
        Passed through to C2GFastEnv.
    fixed_action_values : mapping of str -> float, optional
        Fixed setpoints applied during ``step()``. Values are clipped
        to the action bounds.
    **kwargs
        Forwarded to C2GFastEnv.
    """

    # Class-level metadata exposed for introspection
    ACTION_NAMES     = _ACTION_NAMES
    ACTION_INDEX     = _ACTION_INDEX
    ACTION_BOUNDS    = _ACTION_BOUNDS

    def __init__(
        self,
        scenario: str = "default",
        config_path: str | Path | None = None,
        fixed_action_values: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scenario=scenario, config_path=config_path, **kwargs)
        self._fixed_action_values = _normalise_fixed_action_values(fixed_action_values)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fixed_action_values(self) -> dict[str, float]:
        """Copy of the configured fixed setpoints."""
        return dict(self._fixed_action_values)

    # ------------------------------------------------------------------
    # Overridden step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply ablation overrides then delegate to C2GFastEnv.step().

        Fixed setpoints override requested action dimensions before the
        physics simulation runs.
        """
        requested_action: dict[str, float] = {
            name: float(value)
            for name, value in zip(
                self.ACTION_NAMES,
                np.clip(action.astype(np.float32), self._ACT_LOW, self._ACT_HIGH),
                strict=False,
            )
        }

        clipped = np.fromiter(requested_action.values(), dtype=np.float32)

        applied_arr = clipped.copy()

        for action_name, applied_value in self._fixed_action_values.items():
            applied_arr[self.ACTION_INDEX[action_name]] = applied_value

        applied_action: dict[str, float] = {
            name: float(applied_arr[idx]) for idx, name in enumerate(self.ACTION_NAMES)
        }

        obs, reward, terminated, truncated, info = super().step(applied_arr)

        return obs, reward, terminated, truncated, info
