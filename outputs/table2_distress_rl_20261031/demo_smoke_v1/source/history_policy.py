"""Standalone learned actuator commands with causal observation history.

Unlike residual projection, the neural policy selects pump, HVAC and BESS
directly. Full work service is a fixed policy choice, not added plant capacity.
"""

from __future__ import annotations

import math

import numpy as np

ACTION_LOW = np.array([0.15, 0.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)
FEATURE_COUNT = 24


class OwnHistory:
    """Only the original observation, prior command, filtered request and clock."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_action = np.array([1, 0.7, 0.7, 0], dtype=np.float32)
        self.filtered_request = 0.0
        self.observations = 0

    def features(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (18,) or not np.isfinite(observation).all():
            raise ValueError("Expected one finite original 18D observation")
        request = float(observation[6]) * float(observation[17])
        self.filtered_request += (1 - math.exp(-5 / 900)) * (
            request - self.filtered_request
        )
        result = np.r_[
            observation,
            self.last_action,
            self.filtered_request,
            max(0, 1 - self.observations / 17280),
        ].astype(np.float32)
        self.observations += 1
        return result


def hardware_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if (
        action.shape != (3,)
        or not np.isfinite(action).all()
        or (action < ACTION_LOW).any()
        or (action > ACTION_HIGH).any()
    ):
        raise ValueError("Learned actuator command is outside its declared space")
    return np.r_[np.float32(1.0), action].astype(np.float32)


class HistorySACController:
    def __init__(self, model):
        self.model = model
        self.observer = OwnHistory()

    def reset(self):
        self.observer.reset()

    def predict(self, observation, deterministic=True):
        features = self.observer.features(observation)
        command = self.model.predict(features, deterministic=deterministic)[0]
        native = hardware_action(command)
        self.observer.last_action = native.copy()
        return native, None
