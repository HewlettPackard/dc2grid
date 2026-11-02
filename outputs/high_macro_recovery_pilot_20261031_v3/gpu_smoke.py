"""Small genuine-GPU learning/checkpoint validation, not an efficacy experiment."""

from __future__ import annotations

import copy
import os

# isort: off
from study_common import (
    EXTRA_FEATURES,
    ROOT,
    TASK_SCHEMA,
    load_json,
    tensor_digest,
    write_json,
)
from macro_task import MacroQuantityTask
from prepare import environment_config
from learn import Journal, QuantitySAC, calibrate_critic, initialize_from_teacher
from evaluate import load_macro
import numpy as np
import torch
# isort: on


def main():
    os.sched_setaffinity(0, {60})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU learning smoke requires an actual CUDA device")
    output = ROOT / "smoke/gpu"
    output.mkdir(parents=True, exist_ok=False)
    protocol = copy.deepcopy(load_json(ROOT / "protocol_draft.json"))
    protocol["config_path"] = str(environment_config(output, ticks=360))
    protocol["registration_id"] = "gpu-smoke-only"
    cell = {
        "start_day": protocol["train_days"][0],
        "regd_load_bias": 0.5,
        "seed": 440002,
    }
    env = MacroQuantityTask(protocol, 0.0003, cell=cell)
    try:
        model = QuantitySAC(
            "MlpPolicy",
            env,
            device="cuda:0",
            seed=711,
            teacher_fraction=protocol["teacher_fraction"],
            teacher_explore_every=protocol["teacher_exploration"]["every_local_steps"],
            learning_starts=16,
            train_freq=4,
            gradient_steps=-1,
            gamma=1.0,
            ent_coef=protocol["algorithm"]["ent_coef"],
            batch_size=16,
            buffer_size=128,
            policy_kwargs=protocol["algorithm"]["policy_kwargs"],
        )
        initialization = initialize_from_teacher(model, protocol)
        journal = Journal(
            output, {"kind": "GPU_smoke", "training_is_not_efficacy_evidence": True}
        )
        model.learn(16, reset_num_timesteps=False, callback=journal)
        assert model.num_timesteps == 16 and model._n_updates == 0
        actor_before = tensor_digest(model.actor.state_dict())
        critic_before = tensor_digest(model.critic.state_dict())
        calibration = calibrate_critic(model, 4)
        journal.calibration_updates = 4
        assert tensor_digest(model.actor.state_dict()) == actor_before
        assert tensor_digest(model.critic.state_dict()) != critic_before
        model.learn(16, reset_num_timesteps=False, callback=journal)
        assert model.num_timesteps == 32 and model._n_updates == 20
        assert tensor_digest(model.actor.state_dict()) != actor_before
        assert model.replay_buffer.timeouts.sum() == 0
        assert next(iter(model.actor.optimizer.state.values()))["step"].item() == 16
        model.c2g_context = {
            "task_schema": TASK_SCHEMA,
            "registration_id": protocol["registration_id"],
            "hardware_checkpoint": protocol["hardware"],
            "extra_observation_features": list(EXTRA_FEATURES),
        }
        model.save(output / "model.zip")
        model.save_replay_buffer(output / "replay.pkl")
        restored = load_macro(output / "model.zip", protocol, device="cuda:0")
        observation, _ = env.reset()
        np.testing.assert_array_equal(
            model.predict(observation, deterministic=True)[0],
            restored.predict(observation, deterministic=True)[0],
        )
        report = {
            "passed": True,
            "device": str(model.device),
            "gpu_name": torch.cuda.get_device_name(0),
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "macro_transitions": 32,
            "teacher_guided_transitions": 16,
            "normal_sac_updates": 16,
            "critic_only_updates": 4,
            "supervised_bias_updates": initialization["updates"],
            "actor_changed_under_SAC": True,
            "actor_unchanged_during_calibration": True,
            "zero_timeout_masks": True,
            "checkpoint_deterministic_parity": True,
            "hardware_device": str(env.hardware.model.device),
            "hardware_tensor_digest": env.hardware.verify(),
            "completed_logged_episodes": journal.completed_episodes,
            "initialization": initialization,
            "calibration": calibration,
            "no_development_or_sealed_policy_cases_used": True,
        }
        write_json(ROOT / "smoke/gpu_smoke.json", report)
        print(report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
