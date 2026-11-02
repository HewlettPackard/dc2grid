"""Matched SAC policy-prior ablation: fixed-state versus replay-state Gaussian KL."""

from __future__ import annotations

import copy
import math

# isort: off
from study import FEATURE_CONTRACT
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from sac_core import GuidedSAC
from fit_bc import parameter_digest
# isort: on


def gaussian_kl(mean, log_std, reference_mean, reference_log_std):
    return (
        reference_log_std
        - log_std
        + 0.5
        * (
            (2 * (log_std - reference_log_std)).exp()
            + (mean - reference_mean).square() * (-2 * reference_log_std).exp()
            - 1
        )
    ).sum(dim=1)


class ReferenceSAC(GuidedSAC):
    """The prior is training-only; deployment retains the ordinary SAC actor."""

    def __init__(self, *args, **kwargs):
        self.reference_actor = None
        self.reference_digest = None
        self.reference_weight = 0.0
        self.reference_state_distribution = "replay"
        self.diagnostic_observations = None
        self.warmup_probe = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "reference_actor",
            "diagnostic_observations",
            "warmup_probe",
        ]

    def freeze_reference(self):
        if self.actor.optimizer.state or self.actor_updates:
            raise ValueError("Freeze the BC reference before any SAC actor update")
        self.reference_actor = copy.deepcopy(self.actor)
        self.reference_actor.set_training_mode(False)
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
        self.reference_digest = parameter_digest(self.reference_actor)
        if self.reference_digest != parameter_digest(self.actor):
            raise AssertionError("Reference copy differs from the initialized actor")

    def configure_reference(self, variant, weight):
        distributions = {
            "fixed_reference_kl": "fixed",
            "replay_reference_kl": "replay",
        }
        if variant not in distributions or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Invalid registered reference configuration")
        self.verify_reference()
        self.reference_state_distribution = distributions[variant]
        self.reference_weight = float(weight)
        if self.bc_weight != 0:
            raise ValueError("Legacy BC regularizer cannot be combined implicitly")

    def verify_reference(self):
        if self.reference_actor is None:
            raise RuntimeError("Reference policy is not attached")
        if parameter_digest(self.reference_actor) != self.reference_digest:
            raise AssertionError("Frozen reference parameters changed")
        return self.reference_digest

    def reference_kl(self, observations):
        if self.reference_actor is None:
            raise RuntimeError("Reference policy is required for KL regularization")
        mean, log_std, _ = self.actor.get_action_dist_params(observations)
        with torch.no_grad():
            ref_mean, ref_log_std, _ = self.reference_actor.get_action_dist_params(
                observations
            )
        return gaussian_kl(mean, log_std, ref_mean, ref_log_std).mean()

    def set_diagnostic_batch(self, seed):
        if self.demo_observations is None:
            raise RuntimeError("Attach the shared demonstration states first")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3121]))
        indices = rng.integers(0, len(self.demo_observations), size=1024)
        self.diagnostic_observations = self.demo_observations[
            torch.as_tensor(indices, device=self.device)
        ]

    def snapshot_warmup_probe(self):
        if self.replay_buffer.full or self.replay_buffer.pos == 0:
            raise ValueError("Expected a nonempty, unwrapped warmup buffer")
        states = self.replay_buffer.observations[: self.replay_buffer.pos].reshape(
            -1, self.observation_space.shape[0]
        )
        order = np.random.default_rng(
            np.random.SeedSequence([self.seed, 957])
        ).permutation(len(states))
        self.warmup_probe = torch.as_tensor(states[order].copy(), device=self.device)

    def gradient_probe(self, observations):
        mean_action = self.actor(observations, deterministic=True)
        q = (
            torch.cat(self.critic(observations, mean_action), dim=1)
            .min(dim=1)
            .values.mean()
        )
        kl = self.reference_kl(observations)
        parameters = list(self.actor.parameters())

        def norm(loss):
            gradients = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
            return float(
                torch.sqrt(
                    sum(
                        value.square().sum() for value in gradients if value is not None
                    )
                )
                .detach()
                .cpu()
            )

        result = {
            "q_actor_gradient_norm": norm(q),
            "weighted_kl_actor_gradient_norm": norm(self.reference_weight * kl),
        }
        normalized_steps = (2 * 0.01 / 0.85, 0.002, 0.01)
        with torch.no_grad():
            actions = self.actor(observations, deterministic=True)
            base_q = (
                torch.cat(self.critic(observations, actions), dim=1).min(dim=1).values
            )
            landscape = {}
            for index, name in enumerate(("pump", "hvac", "bess")):
                changes = []
                for sign in (-1, 1):
                    trial = actions.clone()
                    trial[:, index] = torch.clamp(
                        trial[:, index] + sign * normalized_steps[index], -1, 1
                    )
                    trial_q = (
                        torch.cat(self.critic(observations, trial), dim=1)
                        .min(dim=1)
                        .values
                    )
                    changes.append(float((trial_q - base_q).mean().cpu()))
                landscape[name] = {
                    "normalized_action_step": normalized_steps[index],
                    "mean_q_delta_minus_plus": changes,
                }
        result["q_landscape_not_a_physical_counterfactual"] = landscape
        return result

    def diagnostics(self):
        if self.warmup_probe is None:
            raise RuntimeError("Warmup diagnostic states were not frozen")
        count = (
            self.replay_buffer.buffer_size
            if self.replay_buffer.full
            else self.replay_buffer.pos
        )
        states = self.replay_buffer.observations[:count].reshape(-1, 24)
        actions = self.replay_buffer.actions[:count].reshape(-1, 3)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.num_timesteps, 184])
        )
        sample = rng.integers(0, len(states), size=min(1024, len(states)))
        current = torch.as_tensor(states[sample], device=self.device)
        native_actions = self.policy.unscale_action(actions[sample])
        return {
            "fixed_demonstration_states": {
                **self.policy_diagnostics(),
                **self.gradient_probe(self.diagnostic_observations[:128]),
            },
            "fixed_warmup_states": {
                **self.policy_diagnostics(self.warmup_probe),
                **self.gradient_probe(self.warmup_probe[:128]),
            },
            "current_replay_states": {
                **self.policy_diagnostics(current),
                **self.gradient_probe(current[:128]),
            },
            "stored_native_action_std": native_actions.std(axis=0).tolist(),
            "stored_native_action_quantiles": np.quantile(
                native_actions, [0, 0.05, 0.5, 0.95, 1], axis=0
            ).tolist(),
            "replay_clock_min_max": [
                float(states[:, 23].min()),
                float(states[:, 23].max()),
            ],
            "replay_clock_histogram": np.histogram(
                states[:, 23], bins=np.linspace(0, 1, 21)
            )[0].tolist(),
            "replay_clock_bin_edges": np.linspace(0, 1, 21).tolist(),
        }

    def prior_scale_probe(self):
        with torch.no_grad():
            observations = self.diagnostic_observations
            mean, log_std, _ = self.reference_actor.get_action_dist_params(observations)
            native_hvac = (mean[:, 1].tanh() + 1) / 2
            values = []
            for sign in (-1, 1):
                shifted = torch.clamp(native_hvac + sign * 0.001, 1e-6, 1 - 1e-6)
                new_mean = torch.atanh(2 * shifted - 1)
                penalty = (
                    0.5
                    * (new_mean - mean[:, 1]).square()
                    * (-2 * log_std[:, 1]).exp()
                    * self.reference_weight
                )
                reward_scale = (
                    2 * 50 / torch.clamp(observations[:, 17] * 30000, min=100)
                )
                values.append(
                    {
                        "sign": sign,
                        "median_weighted_kl": float(penalty.median().cpu()),
                        "median_immediate_tracking_reward_scale": float(
                            reward_scale.median().cpu()
                        ),
                        "median_ratio_versus_single_step_reward": float(
                            (penalty / reward_scale).median().cpu()
                        ),
                    }
                )
        return {
            "native_hvac_step": 0.001,
            "nominal_hvac_power_step_kw": 50.0,
            "not_a_return_or_improvement_estimate": True,
            "probes": values,
        }

    def policy_diagnostics(self, observations=None):
        observations = (
            self.diagnostic_observations if observations is None else observations
        )
        if observations is None:
            raise RuntimeError("No registered diagnostic states")
        with torch.no_grad():
            mean, log_std, _ = self.actor.get_action_dist_params(observations)
            ref_mean, ref_log_std, _ = self.reference_actor.get_action_dist_params(
                observations
            )
            span = torch.as_tensor(
                self.action_space.high - self.action_space.low, device=self.device
            )
            delta = span / 2 * (mean.tanh() - ref_mean.tanh())
            local_std = span / 2 * (1 - mean.tanh().square()) * log_std.exp()
            kl = gaussian_kl(mean, log_std, ref_mean, ref_log_std)
        return {
            "mean_kl": float(kl.mean().cpu()),
            "maximum_kl": float(kl.max().cpu()),
            "native_mean_action_rmse_from_reference": (
                delta.square().mean(dim=0).sqrt().cpu().tolist()
            ),
            "latent_std_mean": log_std.exp().mean(dim=0).cpu().tolist(),
            "native_local_std_mean": local_std.mean(dim=0).cpu().tolist(),
            "reference_digest": self.verify_reference(),
            "feature_contract": FEATURE_CONTRACT["id"],
        }

    def train(self, gradient_steps, batch_size=64):
        if self.ent_coef_optimizer is not None or self.use_sde:
            raise ValueError("This ablation requires fixed-entropy Gaussian SAC")
        if self.reference_state_distribution not in ("fixed", "replay"):
            raise ValueError("Unknown reference-state distribution")
        if not math.isfinite(self.reference_weight) or self.reference_weight < 0:
            raise ValueError("Reference weight must be finite and nonnegative")
        if self.bc_weight != 0:
            raise ValueError(
                "The old demonstration MSE regularizer must remain disabled"
            )
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
                    states = (
                        self.demo_batch(batch_size)[0]
                        if self.reference_state_distribution == "fixed"
                        else data.observations
                    )
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
