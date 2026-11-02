# isort: off
from study import baseline_protocol, FEATURE_CONTRACT
from low_core import ReferenceSAC, gaussian_kl
from low_task import DiagnosticTask
from fast_task import StandaloneTask
from fit_bc import SpaceOnlyEnv, parameter_digest
from history_policy import HistorySACController

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
# isort: on


def model_with_replay(kind=ReferenceSAC):
    model = kind(
        "MlpPolicy",
        SpaceOnlyEnv(24),
        device="cpu",
        seed=6100,
        ent_coef=1e-5,
        batch_size=32,
        buffer_size=1024,
        policy_kwargs={"net_arch": [32, 32]},
    )
    model.set_logger(configure(format_strings=[]))
    rng = np.random.default_rng(171)
    for _ in range(128):
        model.replay_buffer.add(
            rng.normal(size=(1, 24)).astype(np.float32),
            rng.normal(size=(1, 24)).astype(np.float32),
            rng.uniform(-1, 1, size=(1, 3)).astype(np.float32),
            rng.normal(size=1).astype(np.float32),
            np.array([rng.uniform() < 0.1]),
            [{}],
        )
    return model


def test_kl_matches_torch_distribution_and_direction():
    mean = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    log_std = torch.tensor([[-2.0, -3.0, -1.5]], requires_grad=True)
    ref_mean = torch.tensor([[0.1, 0.1, -0.1]])
    ref_log_std = torch.tensor([[-2.5, -2.5, -2.5]])
    actual = gaussian_kl(mean, log_std, ref_mean, ref_log_std)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean, log_std.exp()),
        torch.distributions.Normal(ref_mean, ref_log_std.exp()),
    ).sum(dim=1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(log_std.grad).all()


def test_equal_distributions_have_zero_kl_and_gradient():
    mean = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    log_std = torch.tensor([[-4.6, -4.6, -4.6]], requires_grad=True)
    value = gaussian_kl(mean, log_std, mean.detach(), log_std.detach()).sum()
    assert value.item() == 0
    value.backward()
    torch.testing.assert_close(mean.grad, torch.zeros_like(mean))
    torch.testing.assert_close(log_std.grad, torch.zeros_like(log_std))


def test_zero_reference_weight_matches_standard_sac_updates():
    reference, standard = model_with_replay(), model_with_replay(SAC)
    assert parameter_digest(reference.policy) == parameter_digest(standard.policy)
    for model in (reference, standard):
        np.random.seed(451)
        torch.manual_seed(451)
        model.train(3, 32)
    assert parameter_digest(reference.policy) == parameter_digest(standard.policy)
    assert reference.actor_updates == 3 and reference._n_updates == 3


@pytest.mark.parametrize("distribution", ["fixed", "replay"])
def test_reference_frozen_and_checkpoint_is_standard_inference(tmp_path, distribution):
    model = model_with_replay()
    model.freeze_reference()
    reference_digest = model.verify_reference()
    rng = np.random.default_rng(191)
    observations = rng.normal(size=(128, 24)).astype(np.float32)
    actions = np.tile(np.array([0.7, 0.7, 0], dtype=np.float32), (128, 1))
    model.attach_demonstrations(observations, actions, 771)
    model.set_diagnostic_batch(771)
    model.reference_state_distribution = distribution
    model.reference_weight = 1.0
    model.calibrate_critics(4)
    assert model.actor_updates == 0 and not model.actor.optimizer.state
    model.train(3, 32)
    assert model.verify_reference() == reference_digest
    assert all(p.grad is None for p in model.reference_actor.parameters())
    assert parameter_digest(model.actor) != reference_digest
    rng_before = torch.get_rng_state().clone()
    parameters_before = parameter_digest(model.policy)
    diagnostics = model.policy_diagnostics()
    assert np.isfinite(diagnostics["mean_kl"])
    assert torch.equal(rng_before, torch.get_rng_state())
    assert parameter_digest(model.policy) == parameters_before
    model.c2g_context = {
        "simulation_mode": "physical_v1",
        "feature_contract": FEATURE_CONTRACT,
        "kind": "state-local-policy-prior-SAC",
    }
    path = tmp_path / f"{distribution}.zip"
    model.save(path)
    loaded = SAC.load(path, device="cpu")
    assert parameter_digest(loaded.policy) == parameter_digest(model.policy)
    controller = HistorySACController(loaded)
    action, _ = controller.predict(np.zeros(18, dtype=np.float32))
    assert action.shape == (4,) and action[0] == 1
    restored = ReferenceSAC.load(path, device="cpu")
    assert restored.reference_actor is None
    with pytest.raises(RuntimeError, match="not attached"):
        restored.verify_reference()


def test_reward_diagnostics_preserve_baseline_transitions():
    protocol = baseline_protocol()
    cell = {"start_day": 16, "regd_load_bias": 0.5, "seed": 730001}
    baseline = StandaloneTask(protocol, seed=6100, fixed_cell=cell)
    diagnostic = DiagnosticTask(protocol, seed=6100, fixed_cell=cell)
    try:
        a, _ = baseline.reset()
        b, _ = diagnostic.reset()
        np.testing.assert_array_equal(a, b)
        for _ in range(360):
            action = np.array([0.7, 0.65, 0.1], dtype=np.float32)
            first, second = baseline.step(action), diagnostic.step(action)
            np.testing.assert_array_equal(first[0], second[0])
            assert first[1:4] == second[1:4]
        assert diagnostic.component_sum[0] > 0
        assert diagnostic.component_sum[1] < 0
    finally:
        baseline.close()
        diagnostic.close()


def test_state_distribution_does_not_perturb_sac_rng():
    states = []
    for distribution in ("fixed", "replay"):
        model = model_with_replay()
        model.freeze_reference()
        observations = np.ones((128, 24), dtype=np.float32)
        targets = np.tile(np.array([0.7, 0.7, 0], dtype=np.float32), (128, 1))
        model.attach_demonstrations(observations, targets, 171)
        model.reference_state_distribution = distribution
        model.reference_weight = 0.1
        model.lr_schedule = lambda progress: 0.0
        np.random.seed(213)
        torch.manual_seed(213)
        model.train(4, 32)
        states.append((np.random.get_state(), torch.get_rng_state().clone()))
    np.testing.assert_array_equal(states[0][0][1], states[1][0][1])
    assert states[0][0][0] == states[1][0][0]
    assert states[0][0][2:] == states[1][0][2:]
    assert torch.equal(states[0][1], states[1][1])


def test_collapsed_variance_has_gradient_toward_the_declared_prior():
    mean = torch.zeros((1, 3), requires_grad=True)
    log_std = torch.full((1, 3), np.log(1e-4), requires_grad=True)
    reference_log_std = torch.full((1, 3), np.log(0.01))
    value = gaussian_kl(mean, log_std, mean.detach(), reference_log_std).mean()
    value.backward()
    assert (log_std.grad < 0).all()


def test_full_diagnostics_preserve_optimizer_parameters_and_randomness():
    model = model_with_replay()
    model.freeze_reference()
    model.configure_reference("replay_reference_kl", 0.1)
    observations = np.ones((128, 24), dtype=np.float32)
    targets = np.tile(np.array([0.7, 0.7, 0], dtype=np.float32), (128, 1))
    model.attach_demonstrations(observations, targets, 171)
    model.set_diagnostic_batch(711)
    model.snapshot_warmup_probe()
    before = parameter_digest(model.policy)
    random_state = torch.get_rng_state().clone()
    diagnostic = model.diagnostics()
    assert parameter_digest(model.policy) == before
    assert torch.equal(random_state, torch.get_rng_state())
    assert diagnostic["fixed_warmup_states"]["q_actor_gradient_norm"] >= 0
    assert (
        diagnostic["fixed_demonstration_states"]["weighted_kl_actor_gradient_norm"] == 0
    )
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert model.prior_scale_probe()["nominal_hvac_power_step_kw"] == 50
