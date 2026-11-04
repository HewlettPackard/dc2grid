import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from baselines.train_corrected_sac import (
    EpisodeConditions,
    TrainingJournal,
    evaluate_suite,
    make_vectors,
    validate_protocol,
)
from c2g_env import C2GFastEnv
from c2g_env.train_envs import RuleMacroWrappedEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol(tmp_path):
    protocol = yaml.safe_load((ROOT / "conf/retraining/corrected_sac.yaml").read_text())
    config = yaml.safe_load((ROOT / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = 4
    config_path = tmp_path / "environment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    protocol["config_path"] = str(config_path)
    protocol["n_envs"] = 2
    protocol["train_days"] = [0, 20]
    protocol["validation"] = [
        {"start_day": 196, "regd_load_bias": 0.0, "seed": 90001},
        {"start_day": 256, "regd_load_bias": 0.5, "seed": 90002},
    ]
    return protocol


def test_registered_splits_and_update_ratio(protocol):
    validate_protocol(protocol)
    for key, value in [("train_days", [207]), ("train_days", [196])]:
        invalid = copy.deepcopy(protocol)
        invalid[key] = value
        with pytest.raises(ValueError, match="disjoint"):
            validate_protocol(invalid)
    invalid = copy.deepcopy(protocol)
    invalid["algorithm"]["gradient_steps"] = 1
    with pytest.raises(ValueError, match="updates per transition"):
        validate_protocol(invalid)


def test_sampler_determinism_and_fixed_validation(protocol):
    def factory(seed, cell=None):
        return EpisodeConditions(
            RuleMacroWrappedEnv(C2GFastEnv(config_path=protocol["config_path"])),
            protocol["train_days"],
            protocol["train_biases"],
            seed=seed,
            validation_cell=cell,
        )

    a, b = factory(10), factory(10)
    conditions = []
    for _ in range(5):
        oa, ia = a.reset()
        ob, ib = b.reset()
        np.testing.assert_array_equal(oa, ob)
        assert ia["condition"] == ib["condition"]
        conditions.append(ia["condition"])
    assert len({c["seed"] for c in conditions}) == 5
    fixed = factory(999, protocol["validation"][0])
    oa, ia = fixed.reset(seed=1)
    ob, ib = fixed.reset(seed=2)
    np.testing.assert_array_equal(oa, ob)
    assert ia["condition"] == ib["condition"] == protocol["validation"][0]
    for env in (a, b, fixed):
        env.close()


def test_spawn_vectors_independent_streams_and_gradient_accounting(protocol, tmp_path):
    train = make_vectors(protocol, "low", 100, [], None)
    validation = make_vectors(protocol, "low", 900, [], None, validation=True)
    try:
        train.seed(100)
        train.reset()
        assert (
            train.reset_infos[0]["condition"]["seed"]
            != train.reset_infos[1]["condition"]["seed"]
        )
        settings = protocol["low"]
        settings.update(eval_freq=100, checkpoint_freq=4, status_freq=2)
        model = SAC(
            "MlpPolicy",
            train,
            learning_starts=2,
            batch_size=2,
            buffer_size=16,
            train_freq=1,
            gradient_steps=-1,
            policy_kwargs={"net_arch": [8, 8]},
            device="cpu",
            seed=100,
        )
        journal = TrainingJournal(
            protocol,
            "low",
            tmp_path,
            validation,
            {"protocol_id": "test"},
        )
        model.learn(6, callback=journal)
        assert model.num_timesteps == 6
        assert model._n_updates == 4
        first = evaluate_suite(model, validation, "low", protocol["selection"])
        second = evaluate_suite(model, validation, "low", protocol["selection"])
        assert first == second
        assert len(first) == 2
        assert all(cell["fast_steps"] == 4 for cell in first)
        manifest = json.loads(
            (tmp_path / "checkpoints/step_4/checkpoint.json").read_text()
        )
        assert manifest["timesteps"] == 4
        assert (tmp_path / "checkpoints/step_4/replay_buffer.pkl").exists()
    finally:
        train.close()
        validation.close()


def test_failed_validation_cells_are_retained(protocol):
    class HotEnv(C2GFastEnv):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self._thermal.temp_A = 40
            return obs, info

    vec = DummyVecEnv(
        [
            lambda: EpisodeConditions(
                HotEnv(config_path=protocol["config_path"]),
                [0],
                [0.0],
                seed=1,
                validation_cell=protocol["validation"][0],
            ),
        ]
    )
    model = SAC("MlpPolicy", vec, policy_kwargs={"net_arch": [8, 8]}, device="cpu")
    try:
        results = evaluate_suite(model, vec, "low", protocol["selection"])
        assert len(results) == 1
        assert not results[0]["qualified"] and results[0]["terminated"]
    finally:
        vec.close()


def test_macro_vectors_freeze_selected_hardware_and_account_work(protocol, tmp_path):
    hardware_env = RuleMacroWrappedEnv(C2GFastEnv(config_path=protocol["config_path"]))
    hardware = SAC(
        "MlpPolicy",
        hardware_env,
        policy_kwargs={"net_arch": [8, 8]},
        device="cpu",
        seed=100,
    )
    checkpoint = tmp_path / "hardware.zip"
    hardware.save(checkpoint)
    macro = make_vectors(
        protocol,
        "high",
        100,
        [],
        str(checkpoint),
        validation=True,
    )
    try:
        model = SAC(
            "MlpPolicy",
            macro,
            policy_kwargs={"net_arch": [8, 8]},
            device="cpu",
            seed=101,
        )
        result = evaluate_suite(model, macro, "high", protocol["selection"])
        assert len(result) == 2
        assert all(cell["steps"] == 1 and cell["fast_steps"] == 4 for cell in result)
        assert all(abs(cell["work_conservation_error_kwh"]) < 1e-6 for cell in result)
        assert all(cell["electricity_cost"] > 0 for cell in result)
    finally:
        macro.close()
        hardware_env.close()
