"""Frozen-v1 SAC update with one new, equally weighted mixed-state prior branch."""

from __future__ import annotations

import math

# isort: off
from ctx import FEATURE_CONTRACT  # noqa: F401
from low_core import ReferenceSAC
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
# isort: on


class MixedReferenceSAC(ReferenceSAC):
    def configure_mixture(self, weight):
        self.configure_reference("replay_reference_kl", weight)
        self.reference_state_distribution = "equal_demo_replay_mixture"

    def train(self, gradient_steps, batch_size=64):
        if self.ent_coef_optimizer is not None or self.use_sde:
            raise ValueError("This ablation requires fixed-entropy Gaussian SAC")
        if self.reference_weight and self.reference_actor is None:
            raise RuntimeError("A frozen BC reference is required for training")
        if self.reference_state_distribution not in (
            "fixed",
            "replay",
            "equal_demo_replay_mixture",
        ):
            raise ValueError("Unknown reference-state distribution")
        if not math.isfinite(self.reference_weight) or self.reference_weight < 0:
            raise ValueError("Reference weight must be finite and nonnegative")
        if self.bc_weight != 0:
            raise ValueError("The old demonstration MSE regularizer must stay disabled")
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        critic_losses, actor_losses, kl_losses, q_scales, target_scales = (
            [],
            [],
            [],
            [],
            [],
        )
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discount = data.discounts if data.discounts is not None else self.gamma
            if not self.critic_only:
                action, log_prob = self.actor.action_log_prob(data.observations)
            with torch.no_grad():
                next_action, next_log_prob = self.actor.action_log_prob(
                    data.next_observations
                )
                next_q = (
                    torch.cat(
                        self.critic_target(data.next_observations, next_action), dim=1
                    )
                    .min(dim=1, keepdim=True)
                    .values
                )
                target = data.rewards + (1 - data.dones) * discount * (
                    next_q - self.ent_coef_tensor * next_log_prob.reshape(-1, 1)
                )
            current = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(value, target) for value in current)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("Nonfinite critic loss")
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))
            q_scales.append(float(current[0].detach().abs().mean().cpu()))
            target_scales.append(float(target.abs().mean().cpu()))
            if not self.critic_only:
                q_pi = (
                    torch.cat(self.critic(data.observations, action), dim=1)
                    .min(dim=1, keepdim=True)
                    .values
                )
                actor_loss = (
                    self.ent_coef_tensor * log_prob.reshape(-1, 1) - q_pi
                ).mean()
                if self.reference_weight:
                    if self.reference_state_distribution == "fixed":
                        states = self.demo_batch(batch_size)[0]
                    elif (
                        self.reference_state_distribution == "equal_demo_replay_mixture"
                    ):
                        states = torch.cat(
                            (
                                data.observations,
                                self.demo_batch(batch_size)[0],
                            ),
                            dim=0,
                        )
                        if states.shape[0] != 2 * batch_size:
                            raise AssertionError(
                                "The two reference sources must be equally weighted"
                            )
                    else:
                        states = data.observations
                    kl = self.reference_kl(states)
                    if not torch.isfinite(kl) or kl < -1e-5:
                        raise FloatingPointError("Invalid Gaussian reference KL")
                    actor_loss = actor_loss + self.reference_weight * kl
                    kl_losses.append(float(kl.detach().cpu()))
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Nonfinite actor loss")
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                self.actor_updates += 1
                actor_losses.append(float(actor_loss.detach().cpu()))
            if step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/actor_updates", self.actor_updates)
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/critic_abs_q", float(np.mean(q_scales)))
        self.logger.record("train/target_abs_q", float(np.mean(target_scales)))
        self.logger.record("train/reference_weight", self.reference_weight)
        if actor_losses:
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        if kl_losses:
            self.logger.record("train/reference_kl", float(np.mean(kl_losses)))
        with torch.no_grad():
            mean, log_std, _ = self.actor.get_action_dist_params(data.observations)
            span = torch.as_tensor(
                self.action_space.high - self.action_space.low, device=self.device
            )
            local_std = span / 2 * (1 - mean.tanh().square()) * log_std.exp()
            for index, name in enumerate(("pump", "hvac", "bess")):
                self.logger.record(
                    f"train/latent_std_{name}",
                    float(log_std[:, index].exp().mean().cpu()),
                )
                self.logger.record(
                    f"train/native_local_std_{name}",
                    float(local_std[:, index].mean().cpu()),
                )
        if not self.critic_only and self.num_timesteps > 0:
            if self.num_timesteps % 4096 == 0:
                self.logger.dump(step=self.num_timesteps)
                self.last_training_dump_step = self.num_timesteps
