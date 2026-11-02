"""SAC with an explicit, optional fixed-demonstration actor regularizer.

The Bellman, entropy and target updates follow SB3 2.7.1 SAC. The only actor
objective addition is the documented BC term; the environment reward is intact.
"""

from __future__ import annotations

# isort: off
import common  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from fit_bc import parameter_digest
# isort: on


class GuidedSAC(SAC):
    def __init__(self, *args, **kwargs):
        self.bc_weight = 0.0
        self.demo_observations = None
        self.demo_actions = None
        self.demo_rng = None
        self.critic_only = False
        self.actor_updates = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "demo_observations",
            "demo_actions",
            "demo_rng",
        ]

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            action, _ = self.predict(self._last_obs, deterministic=False)
            return action, self.policy.scale_action(action)
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def attach_demonstrations(self, observations, actions, seed):
        self.demo_observations = torch.as_tensor(observations, device=self.device)
        self.demo_actions = torch.as_tensor(
            self.policy.scale_action(actions), device=self.device
        )
        self.demo_rng = np.random.default_rng(np.random.SeedSequence([seed, 601]))

    def demo_batch(self, size):
        if (
            self.demo_observations is None
            or self.demo_actions is None
            or self.demo_rng is None
        ):
            raise RuntimeError("Fixed demonstration corpus is not attached")
        indices = self.demo_rng.integers(0, len(self.demo_observations), size=size)
        index = torch.as_tensor(indices, device=self.device)
        return self.demo_observations[index], self.demo_actions[index]

    def calibrate_regularization(self, ratio=1.0):
        observations, target = self.demo_batch(1024)
        deterministic = self.actor(observations, deterministic=True)
        q = (
            torch.cat(self.critic(observations, deterministic), dim=1)
            .min(dim=1)
            .values.mean()
        )
        bc = F.mse_loss(deterministic, target)
        parameters = [p for p in self.actor.parameters() if p.requires_grad]

        def norm(gradients):
            return torch.sqrt(
                sum(
                    gradient.square().sum()
                    for gradient in gradients
                    if gradient is not None
                )
            )

        q_norm = norm(
            torch.autograd.grad(q, parameters, retain_graph=True, allow_unused=True)
        )
        bc_norm = norm(torch.autograd.grad(bc, parameters, allow_unused=True))
        if not torch.isfinite(q_norm + bc_norm) or bc_norm <= 1e-12:
            raise ValueError("Cannot calibrate a finite nonzero BC gradient")
        weight = float((ratio * q_norm / bc_norm).detach().cpu())
        if not 0 < weight < 1e6:
            raise ValueError(f"Unusable BC gradient balance: {weight}")
        return {
            "method": "fixed_initial_actor_parameter_gradient_norm_ratio",
            "target_ratio": ratio,
            "q_gradient_norm": float(q_norm.detach().cpu()),
            "bc_gradient_norm": float(bc_norm.detach().cpu()),
            "bc_weight": weight,
            "gradient_calibration_uses_only_training_demonstrations": True,
            "fixed_entropy_coefficient": float(self.ent_coef_tensor.cpu()),
        }

    def train(self, gradient_steps, batch_size=64):
        if self.ent_coef_optimizer is not None or self.use_sde:
            raise ValueError(
                "This study requires fixed entropy and ordinary Gaussian SAC"
            )
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        critic_losses, actor_losses, bc_losses = [], [], []
        q_scales, target_scales = [], []
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
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
                target = data.rewards + (1 - data.dones) * discounts * (
                    next_q - self.ent_coef_tensor * next_log_prob.reshape(-1, 1)
                )
            current = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(value, target) for value in current)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("Nonfinite SAC critic loss")
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
                if self.bc_weight:
                    demo_obs, demo_action = self.demo_batch(batch_size)
                    bc_prediction = self.actor(demo_obs, deterministic=False)
                    bc_loss = F.mse_loss(bc_prediction, demo_action)
                    actor_loss = actor_loss + self.bc_weight * bc_loss
                    bc_losses.append(float(bc_loss.detach().cpu()))
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Nonfinite SAC actor loss")
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
        if actor_losses:
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        if bc_losses:
            self.logger.record("train/bc_loss", float(np.mean(bc_losses)))
        self.logger.record("train/bc_weight", self.bc_weight)

    def calibrate_critics(self, updates):
        before = parameter_digest(self.actor)
        if self.actor.optimizer.state or self.actor_updates:
            raise ValueError("Critic calibration must precede actor optimizer updates")
        self.critic_only = True
        try:
            self.train(updates, self.batch_size)
        finally:
            self.critic_only = False
        if parameter_digest(self.actor) != before or self.actor.optimizer.state:
            raise AssertionError("Critic-only calibration changed the actor")
        return {
            "critic_updates": updates,
            "actor_digest_unchanged": before,
            "actor_optimizer_empty": True,
        }
