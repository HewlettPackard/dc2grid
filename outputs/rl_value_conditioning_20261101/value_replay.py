"""Terminal-correct frozen-potential conditioning of ordinary SAC replay rewards."""

from __future__ import annotations

import copy
import math

# isort: off
from ctx import FEATURE_CONTRACT  # noqa: F401
from mixed_core import MixedReferenceSAC
import torch
from stable_baselines3.common.buffers import ReplayBuffer
# isort: on


def potential(observations, config, gamma):
    if observations.shape[-1] != 24 or not 0 < gamma < 1:
        raise ValueError("Potential requires the frozen 24-D finite-horizon contract")
    remaining = torch.round(observations[:, 23:24].double() * 17280).clamp(
        min=0, max=17280
    )
    factor = -torch.expm1(remaining * math.log(gamma)) / (1 - gamma)
    if config["kind"] == "constant":
        mean = torch.full_like(factor, config["mean"])
    elif config["kind"] == "linear_causal":
        indices = [3, 4] if config["weights"][3] == 0 else [3, 4, 16]
        coefficients = config["weights"][: len(indices) + 1]
        features = torch.cat(
            (torch.ones_like(remaining), observations[:, indices].double()), dim=1
        )
        weights = torch.as_tensor(
            coefficients, dtype=torch.float64, device=observations.device
        )
        mean = (features @ weights).reshape(-1, 1).clamp(0, config["alpha"])
    else:
        raise ValueError("Unknown frozen potential kind")
    return factor * mean


def shaped_rewards(samples, config, gamma):
    current = potential(samples.observations, config, gamma)
    following = potential(samples.next_observations, config, gamma)
    transformed = (
        samples.rewards.double()
        + gamma * (1 - samples.dones.double()) * following
        - current
    )
    return transformed.to(samples.rewards.dtype)


class PotentialReplayBuffer(ReplayBuffer):
    def __init__(self, *args, potential_config, potential_gamma, **kwargs):
        super().__init__(*args, **kwargs)
        self.potential_config = copy.deepcopy(potential_config)
        self.potential_gamma = potential_gamma
        if not 0 < potential_gamma < 1:
            raise ValueError("Invalid potential discount")
        if potential_config["kind"] == "constant":
            if (
                not math.isfinite(potential_config["mean"])
                or potential_config["mean"] < 0
            ):
                raise ValueError("This study requires a finite nonnegative mean")
        elif potential_config["kind"] == "linear_causal":
            if (
                len(potential_config["weights"]) != 4
                or not all(
                    math.isfinite(value) for value in potential_config["weights"]
                )
                or not math.isfinite(potential_config["alpha"])
                or potential_config["alpha"] <= 0
            ):
                raise ValueError("Invalid causal potential coefficients")
        else:
            raise ValueError("Unknown potential configuration")

    def sample(self, batch_size, env=None):
        if env is not None:
            raise ValueError("Potential units require unnormalized replay rewards")
        data = super().sample(batch_size, env)
        return data._replace(
            rewards=shaped_rewards(data, self.potential_config, self.potential_gamma)
        )


class ConditionedSAC(MixedReferenceSAC):
    def _setup_model(self):
        super()._setup_model()
        if self._vec_normalize_env is not None:
            raise ValueError("This study does not normalize rewards or observations")
        if not isinstance(self.replay_buffer, PotentialReplayBuffer):
            raise ValueError(
                "Training requires the explicitly configured potential buffer"
            )
        if self.replay_buffer.potential_gamma != self.gamma:
            raise ValueError("Potential and Bellman discounts must agree")

    def _excluded_save_params(self):
        # Training-only replay transforms are not required by ordinary SAC inference.
        return super()._excluded_save_params() + [
            "replay_buffer_class",
            "replay_buffer_kwargs",
        ]
