from __future__ import annotations

import copy

import pytest

# isort: off
from run_fraction import (
    MacroJournal, MacroLatentEnv, attach_update_counters,
    initialize_actor, warmup_and_calibrate,
)
from economy_context import PRIOR, cap_cpu, read_json
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
import torch
# isort: on


@pytest.mark.parametrize("algorithm", ("ppo", "sac"))
@pytest.mark.parametrize("variant", ("fraction", "potential"))
def test_native_fractional_training_integration(algorithm, variant, tmp_path):
    from fraction_macro import CreditPotentialEnv, FractionalDispatchTask

    cap_cpu()
    protocol = copy.deepcopy(read_json(PRIOR / "registration.json"))
    protocol["n_envs"] = 1
    cell = {"start_day": 16, "seed": 97000030, "regd_load_bias": 0.5}
    task = FractionalDispatchTask(protocol, seed=1200, fixed_cell=cell)
    environment = task
    if variant == "potential":
        environment = CreditPotentialEnv(environment, coefficient_usd_per_mwh=10.0)
    if algorithm == "ppo":
        environment = MacroLatentEnv(environment)
        model = PPO(
            "MlpPolicy", environment, n_steps=8, batch_size=4, n_epochs=2,
            gamma=1.0, ent_coef=0.0, learning_rate=1e-4,
            policy_kwargs={
                "net_arch": {"pi": [16, 16], "vf": [16, 16]},
                "activation_fn": torch.nn.ReLU, "ortho_init": False,
            },
            seed=1200, device="cpu",
        )
    else:
        model = SAC(
            "MlpPolicy", environment, buffer_size=128, batch_size=4,
            learning_starts=16, train_freq=1, gradient_steps=1,
            gamma=1.0, ent_coef=0.0001, learning_rate=1e-4,
            policy_kwargs={"net_arch": [16, 16]}, seed=1200, device="cpu",
        )
    initialize_actor(model, algorithm, 1200, 0.05, [16, 16])
    model.set_logger(configure(str(tmp_path / "metrics"), ["csv"]))
    handles = attach_update_counters(model, algorithm)
    journal = MacroJournal(tmp_path)
    try:
        warmup_and_calibrate(
            model, algorithm, warmup_steps=16, critic_updates=8, callback=journal,
        )
        model.learn(16, callback=journal, reset_num_timesteps=False)
        receipt = journal.save_transitions(model)
        assert receipt["transitions"] == 32
        assert model.c2g_update_counts["actor"] > 0
        assert model.c2g_update_counts["critic"] == model.c2g_update_counts["actor"] + 8
    finally:
        for handle in handles:
            handle.remove()
        model.get_env().close()
