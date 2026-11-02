"""Bounded task, recovery, anti-withdrawal, initialization and optimizer contracts."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

# isort: off
from study_common import (
    EXTRA_FEATURES,
    ROOT,
    TASK_SCHEMA,
    cell_key,
    load_json,
    tensor_digest,
)
from macro_task import MacroQuantityTask, map_quantity, potential, score_transition
from prepare import environment_config
from evaluate import OneShotEvaluation, load_macro
from learn import Journal, QuantitySAC, calibrate_critic, initialize_from_teacher
import gymnasium as gym
import numpy as np
import pytest
# isort: on


@pytest.fixture
def protocol(tmp_path):
    value = copy.deepcopy(load_json(ROOT / "protocol_draft.json"))
    value["config_path"] = str(environment_config(tmp_path, ticks=360))
    value["registration_id"] = "unit-test"
    value["development_cells"] = [
        {"start_day": 16, "regd_load_bias": 0.5, "seed": 440001}
    ]
    return value


@pytest.mark.parametrize("residual", [5.00001, 12.000001, 17.999999, 29.999999, 30.0])
def test_float32_quantities_never_oversize(residual):
    core = SimpleNamespace(
        _pending_offer={"residual_mw": residual},
        _committed_max_mw=30.0,
        _dr_baseline_mw=5.0,
    )
    for fraction in [0.0, 1e-8, 0.5, 0.98, 1.0]:
        native, maximum = map_quantity(core, fraction)
        assert native.dtype == np.float32 and native.shape == (2,)
        assert 5 <= float(native[0]) * 30 <= maximum <= residual
        assert native[1] == 0


def test_no_feasible_quantity_is_an_error():
    core = SimpleNamespace(
        _pending_offer={"residual_mw": 4.9},
        _committed_max_mw=30.0,
        _dr_baseline_mw=5.0,
    )
    with pytest.raises(ValueError, match="quantity range"):
        map_quantity(core, 0.5)
    with pytest.raises(ValueError, match="fraction"):
        map_quantity(core, float("nan"))
    with pytest.raises(ValueError, match="integer"):
        cell_key({"start_day": 4.5, "seed": 1, "regd_load_bias": 0.0})


@pytest.mark.parametrize(
    "cash_increments", [[10, 20, 30], [-80, 10], [1], [40, -30, 20, -60]]
)
def test_fault_timing_cannot_improve_base_return(cash_increments):
    cash = total = 0.0
    for index, increment in enumerate(cash_increments):
        terminal = index == len(cash_increments) - 1
        reward, _, cash = score_transition(
            cash, increment, 1000, 0, 0, terminal=terminal, valid=not terminal
        )
        total += reward
    assert total == pytest.approx(-2, abs=1e-12)
    assert cash == sum(cash_increments)


def test_potential_preserves_valid_and_invalid_episode_ranking():
    for valid in (True, False):
        cash = base_total = shaped_total = 0.0
        phis = [0.0, -0.004, -0.01, 0.0]
        for index, increment in enumerate([1, 2, 3]):
            reward, shaping, cash = score_transition(
                cash,
                increment,
                1000,
                phis[index],
                phis[index + 1],
                terminal=index == 2,
                valid=valid,
            )
            base_total += reward
            shaped_total += reward + shaping
        assert shaped_total == pytest.approx(base_total, abs=1e-12)
        assert base_total == pytest.approx(0.006 if valid else -2)
    assert potential(0.5, 0.01, target=0.3, margin=0.1) == 0
    assert potential(0.2, 0.01, target=0.3, margin=0.1) == pytest.approx(-0.01)
    with pytest.raises(ValueError, match="bound"):
        score_transition(900, 200, 1000, 0, 0, terminal=True, valid=True)


def test_shaping_does_not_change_physics_and_end_is_not_timeout(protocol):
    records = []
    for kappa in (0.0, 0.01):
        env = MacroQuantityTask(protocol, kappa, cell=protocol["development_cells"][0])
        try:
            observation, _ = env.reset(seed=7)
            seen = [observation.copy()]
            while True:
                observation, _, terminal, truncated, info = env.step(
                    np.array([0.98], dtype=np.float32)
                )
                seen.append(observation.copy())
                if terminal:
                    break
            result = info["c2g_episode"]
            assert not truncated
            assert result["qualified"] and result["recovery"]["closed"]
            assert result["decisions"] == 2
            assert result["hardware_predictions"] == 360
            assert result["service"]["steps"] == 360
            assert result["recovery"]["ledger"]["steps"] == 1440
            assert result["recovery"]["regulation_revenue_usd"] == 0
            assert result["base_return"] == pytest.approx(
                result["shaped_return"], abs=1e-12
            )
            records.append((seen, result["cashflow_26h_usd"]))
        finally:
            env.close()
    np.testing.assert_array_equal(records[0][0], records[1][0])
    assert records[0][1] == records[1][1]


def test_withdrawal_is_not_a_success_even_if_recovery_closes(protocol):
    env = MacroQuantityTask(protocol, 0.01, cell=protocol["development_cells"][0])
    try:
        env.reset()
        while True:
            _, _, terminal, _, info = env.step(np.array([0.0], dtype=np.float32))
            if terminal:
                break
        result = info["c2g_episode"]
        assert result["service_complete"] and result["recovery"]["closed"]
        assert result["delivered_service_ratio"] < 0.9
        assert not result["qualified"]
        assert result["base_return"] == pytest.approx(-2)
        assert result["shaped_return"] == pytest.approx(-2)
        assert result["cashflow_closed_recovery_usd"] is not None
    finally:
        env.close()


def test_physical_fault_cancels_previous_cash_without_recovery(protocol, monkeypatch):
    from c2g_env.physics.weather import WeatherLoader

    env = MacroQuantityTask(protocol, 0.01, cell=protocol["development_cells"][0])
    try:
        env.reset()
        env.step(np.array([0.98], dtype=np.float32))
        fast = env.core._fast_env
        fast._thermal.temp_B = 35.0
        workload = fast._workload
        index = (workload._tick // workload._steps_per_trace_tick) % workload._n
        workload._p_base_b_arr[index] = 100000.0
        workload._p_base_arr[index] = workload._p_base_a_arr[index] + 100000.0
        monkeypatch.setattr(WeatherLoader, "temp_c", lambda self, tick: 60.0)
        _, _, terminal, truncated, info = env.step(np.array([0.98], dtype=np.float32))
        result = info["c2g_episode"]
        assert terminal and not truncated
        assert result["service_physical_terminated"]
        assert not result["recovery"]["performed"]
        assert result["cashflow_26h_usd"] is None
        assert result["base_return"] == pytest.approx(-2, abs=1e-12)
        assert result["shaped_return"] == pytest.approx(-2, abs=1e-12)
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(np.array([0.98], dtype=np.float32))
    finally:
        env.close()


def test_padding_does_not_reset_or_step_the_plant(protocol):
    class FakeTask(gym.Env):
        action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)

        def __init__(self):
            self.protocol = protocol
            self.done = True
            self.resets = self.steps = 0

        def reset(self, **kwargs):
            self.resets += 1
            self.done = False
            return np.array([0], dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            self.done = True
            return np.array([1], dtype=np.float32), 0.0, True, False, {}

    task = FakeTask()
    wrapped = OneShotEvaluation(task)
    wrapped.reset()
    wrapped.step(np.array([1]))
    for _ in range(4):
        _, info = wrapped.reset()
        assert info["evaluation_padding_reset"]
        _, _, _, _, info = wrapped.step(np.array([1]))
        assert info["evaluation_padding"]
    assert task.steps == task.resets == 1
    with pytest.raises(ValueError, match="sealed"):
        wrapped.assign_cell({"start_day": 72, "seed": 10, "regd_load_bias": 0.0})


def test_gpu_visibility_survives_recovery_import():
    environment = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="3")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import recovery_ops; "
            "assert os.environ['CUDA_VISIBLE_DEVICES']=='3'; print('preserved')",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "preserved"


def test_staggered_warmup_exploration_covers_all_workers():
    model = object.__new__(QuantitySAC)
    model.teacher_fraction = 0.98
    model.teacher_explore_every = 20
    model.teacher_rng = np.random.default_rng(771)
    model.warm_exploration_count = 0
    model.policy = SimpleNamespace(scale_action=lambda value: 2 * value - 1)
    counts = [0] * 4
    for row in range(256):
        model.num_timesteps = row * 4
        actions, scaled = model._sample_action(1024, n_envs=4)
        np.testing.assert_allclose(scaled, 2 * actions - 1)
        for rank in range(4):
            if (row + rank * 5) % 20 == 0:
                counts[rank] += 1
                assert 0 <= actions[rank, 0] <= 1
            else:
                assert actions[rank, 0] == np.float32(0.98)
    assert counts == [13, 13, 13, 13]
    assert model.warm_exploration_count == 52


def test_entropy_is_scaled_to_the_registered_cash_units(protocol):
    from register import entropy_calibration

    report = entropy_calibration(protocol)
    assert report["maximum_registered_gradient_ratio"] < 0.05
    assert report["rows"][2]["old_alpha_gradient_ratio"] > 20


def test_artificial_collection_budget_is_not_a_task_terminal(protocol, tmp_path):
    protocol = copy.deepcopy(protocol)
    protocol["config_path"] = str(environment_config(tmp_path / "longer", ticks=540))
    env = MacroQuantityTask(protocol, 0.0003, cell=protocol["development_cells"][0])
    try:
        model = QuantitySAC(
            "MlpPolicy",
            env,
            seed=94,
            device="cpu",
            learning_starts=4,
            train_freq=4,
            gradient_steps=-1,
            gamma=1.0,
            batch_size=4,
            buffer_size=64,
            ent_coef=1e-9,
            policy_kwargs={"net_arch": [16, 16]},
        )
        initialize_from_teacher(model, protocol)
        model.learn(4, reset_num_timesteps=False)
        assert model.replay_buffer.dones[3, 0] == 0
        assert model.replay_buffer.timeouts[3, 0] == 0
        assert not env.done and env.service.steps == 180
    finally:
        env.close()


def test_full_sac_cannot_bypass_a_failed_feasibility_gate(
    protocol, tmp_path, monkeypatch
):
    import register
    from study_common import write_json

    monkeypatch.setattr(register, "ROOT", tmp_path)
    monkeypatch.setattr(register, "verify_registration", lambda value: None)
    write_json(
        tmp_path / "training_gate.json",
        {
            "registration_id": protocol["registration_id"],
            "allowed": False,
        },
    )
    with pytest.raises(RuntimeError, match="feasibility"):
        register.verify_training_gate(protocol)


def test_optimizer_budgets_terminal_replay_and_model_loading(protocol, tmp_path):
    env = MacroQuantityTask(protocol, 0.0, cell=protocol["development_cells"][0])
    try:
        model = QuantitySAC(
            "MlpPolicy",
            env,
            seed=92,
            device="cpu",
            learning_starts=8,
            train_freq=4,
            gradient_steps=-1,
            gamma=1.0,
            batch_size=8,
            buffer_size=128,
            ent_coef=protocol["algorithm"]["ent_coef"],
            policy_kwargs={"net_arch": [32, 32]},
        )
        initialization = initialize_from_teacher(model, protocol)
        assert initialization["final_loss"] < initialization["initial_loss"]
        journal = Journal(tmp_path, {"kind": "unit_test"})
        model.learn(8, reset_num_timesteps=False, callback=journal)
        assert model.num_timesteps == 8 and model._n_updates == 0
        assert model.warm_exploration_count == 1
        warm_actions = model.policy.unscale_action(
            model.replay_buffer.actions[:8].reshape(-1, 1)
        )
        np.testing.assert_allclose(warm_actions[1:], 0.98, atol=1e-7)
        recorded = [
            json.loads(line)
            for line in (tmp_path / "training_episodes.jsonl").read_text().splitlines()
        ]
        assert len(recorded) == 4
        recovered = [row for row in recorded if row["recovery"]["performed"]]
        assert recovered
        assert all(
            isinstance(row["recovery"]["thermal_A_closed"], bool) for row in recovered
        )
        actor = tensor_digest(model.actor.state_dict())
        critic = tensor_digest(model.critic.state_dict())
        calibrate_critic(model, 4)
        journal.calibration_updates = 4
        assert tensor_digest(model.actor.state_dict()) == actor
        assert tensor_digest(model.critic.state_dict()) != critic
        model.learn(8, reset_num_timesteps=False, callback=journal)
        assert model.num_timesteps == 16 and model._n_updates == 12
        assert next(iter(model.actor.optimizer.state.values()))["step"].item() == 8
        assert model.replay_buffer.timeouts.sum() == 0
        assert (
            model.replay_buffer.dones[: model.replay_buffer.pos].sum()
            == journal.completed_episodes
        )
        assert journal.action_count == 16
        model.c2g_context = {
            "task_schema": TASK_SCHEMA,
            "registration_id": protocol["registration_id"],
            "extra_observation_features": list(EXTRA_FEATURES),
            "hardware_checkpoint": protocol["hardware"],
        }
        model.save(tmp_path / "model.zip")
        restored = load_macro(tmp_path / "model.zip", protocol)
        observation, _ = env.reset()
        np.testing.assert_array_equal(
            model.predict(observation, deterministic=True)[0],
            restored.predict(observation, deterministic=True)[0],
        )
        altered = copy.deepcopy(protocol)
        altered["hardware"]["sha256"] = "wrong"
        with pytest.raises(ValueError, match="mismatch"):
            load_macro(tmp_path / "model.zip", altered)
    finally:
        env.close()
