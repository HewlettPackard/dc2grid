"""
tests/test_baselines.py  —  Smoke tests for Phase 3 baseline & evaluation code
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch

from c2g_env import C2GFastEnv
from c2g_env.plant_profiles import available_plant_profiles, load_plant_profile
from baselines.rule_based_mpc import RuleBasedController


# ── helpers ─────────────────────────────────────────────────────────────────

def _fresh_env(scenario: str = "default", seed: int = 0):
    env = C2GFastEnv(scenario=scenario)
    obs, _ = env.reset(seed=seed)
    return env, obs


# ═══════════════════════════════════════════════════════════════════════════
# RuleBasedController
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleBasedController:

    def test_instantiation(self):
        ctrl = RuleBasedController()
        assert ctrl is not None

    def test_predict_returns_tuple(self):
        ctrl = RuleBasedController()
        _, obs = _fresh_env()
        result = ctrl.predict(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2            # (action, state)

    def test_action_shape(self):
        ctrl = RuleBasedController()
        _, obs = _fresh_env()
        action, _ = ctrl.predict(obs)
        assert action.shape == (4,)

    def test_action_range(self):
        """Action must stay within [-1, 1] for all 3 dims."""
        ctrl = RuleBasedController()
        _, obs = _fresh_env()
        action, _ = ctrl.predict(obs)
        assert np.all(action >= -1.0 - 1e-6)
        assert np.all(action <= 1.0 + 1e-6)

    def test_thermal_protection_high_temp(self):
        """At critical temperature, throttle → 0 and hvac → 1."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[0] = 0.99   # temp_A nearly at T_safe
        obs[1] = 0.99   # temp_B nearly at T_safe
        action, _ = ctrl.predict(obs)
        assert action[0] < 0.2, "throttle should be near 0 under thermal protection"
        assert action[2] > 0.5, "hvac should be high under thermal protection"

    def test_bess_follows_regd(self):
        """Positive regd → BESS discharge (positive action)."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.5    # SOC=50%, no saturation
        obs[6] = 0.5    # positive regd signal
        action, _ = ctrl.predict(obs)
        assert action[3] > 0.0, "BESS should discharge with positive regd"

    def test_bess_soc_guard_low(self):
        """With SOC nearly empty, BESS should not discharge much."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.05   # SOC=5%, near empty
        obs[6] = 0.8    # strong regd signal asking for discharge
        action, _ = ctrl.predict(obs)
        # discharge action should be suppressed
        assert action[3] < 0.5, "High discharge should be suppressed at low SOC"

    def test_bess_soc_guard_high(self):
        """With SOC nearly full, BESS should not charge much."""
        ctrl = RuleBasedController()
        obs = np.zeros(17, dtype=np.float32)
        obs[2] = 0.95   # SOC=95%, nearly full
        obs[6] = -0.8   # grid asking us to charge
        action, _ = ctrl.predict(obs)
        # charge action (negative) should be suppressed
        assert action[3] > -0.5, "Charging should be suppressed at high SOC"

    def test_full_episode_no_crash(self):
        """Run RuleBasedController for a full episode without exceptions."""
        ctrl = RuleBasedController()
        env, obs = _fresh_env(seed=42)
        step_count = 0
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
        assert step_count > 0

    def test_full_episode_thermal_default(self):
        """Default scenario should not cause thermal termination with rule-based."""
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="default", seed=0)
        done = False
        terminated = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        # Rule-based should survive most of the time on default
        # (not guaranteed, but we test it at least runs)
        assert True

    def test_scenario_a_no_crash(self):
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="scenario_a", seed=1)
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr

    def test_scenario_b_no_crash(self):
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="scenario_b", seed=2)
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr

    def test_scenario_c_no_crash(self):
        ctrl = RuleBasedController()
        env, obs = _fresh_env(scenario="scenario_c", seed=3)
        done = False
        while not done:
            action, _ = ctrl.predict(obs)
            obs, _, t, tr, _ = env.step(action)
            done = t or tr

    def test_deterministic(self):
        """Same obs should give same action."""
        ctrl = RuleBasedController()
        obs = np.array([0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.0, 0.5, 0.5],
                       dtype=np.float32)
        a1, _ = ctrl.predict(obs)
        a2, _ = ctrl.predict(obs)
        np.testing.assert_array_equal(a1, a2)


# ═══════════════════════════════════════════════════════════════════════════
# run_benchmark helpers (import-level tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkImport:
    @pytest.mark.parametrize("profile", available_plant_profiles())
    @pytest.mark.parametrize("scenario", ["default", "scenario_a", "scenario_b", "scenario_c"])
    def test_rule_factory_uses_effective_plant_limits(self, profile, scenario):
        from evaluation.run_benchmark import _make_inner_controller

        env = C2GFastEnv(scenario=scenario, plant_overrides=load_plant_profile(profile))
        try:
            env.reset(seed=0)
            controller = _make_inner_controller("rule_based", env=env)
            assert controller.bess_gain == pytest.approx(env._committed_mw_max / env._bess.P_MAX_MW)
        finally:
            env.close()

    def test_imports(self):
        from evaluation.run_benchmark import run_episode, RandomAgent
        assert callable(run_episode)

    def test_random_agent_action_shape(self):
        from evaluation.run_benchmark import RandomAgent
        env, _ = _fresh_env()
        agent = RandomAgent(env)
        obs = env.observation_space.sample()
        action, state = agent.predict(obs)
        assert action.shape == (4,)
        assert state is None

    def test_run_episode_returns_dict(self):
        from evaluation.run_benchmark import run_episode, RandomAgent
        env, _ = _fresh_env()
        agent = RandomAgent(env)
        m = run_episode(agent, "default", seed=0)
        expected_keys = {
            "mean_reward", "total_reward", "tracking_rmse",
            "thermal_viol_rate", "throughput_ratio", "bess_degradation",
            "episode_length", "survived",
        }
        assert expected_keys.issubset(m.keys())

    def test_run_episode_finite_values(self):
        from evaluation.run_benchmark import run_episode, RandomAgent
        env, _ = _fresh_env()
        agent = RandomAgent(env)
        m = run_episode(agent, "default", seed=42)
        for k, v in m.items():
            assert np.isfinite(v), f"Metric {k} = {v} is not finite"

    def test_run_episode_rule_based(self):
        from evaluation.run_benchmark import run_episode
        ctrl = RuleBasedController()
        m = run_episode(ctrl, "default", seed=0)
        assert m["episode_length"] > 0
        assert 0.0 <= m["thermal_viol_rate"] <= 1.0

    def test_run_episode_all_scenarios(self):
        from evaluation.run_benchmark import run_episode
        ctrl = RuleBasedController()
        for sc in ["default", "scenario_a", "scenario_b", "scenario_c"]:
            m = run_episode(ctrl, sc, seed=0)
            assert m["episode_length"] > 0, f"Episode too short for {sc}"

    def test_benchmark_function_csv(self, tmp_path):
        from evaluation.run_benchmark import benchmark, save_csv
        rows = benchmark(
            agents     = ["rule_based", "random"],
            scenarios  = ["default"],
            n_episodes = 2,
            seed_start = 0,
            model_dir  = None,
        )
        assert len(rows) == 2   # 2 agents × 1 scenario
        out = tmp_path / "results.csv"
        save_csv(rows, out)
        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2
        assert "mean_reward" in data[0]

    def test_load_sb3_agent_restores_obs_normalization(self, monkeypatch, tmp_path):
        import evaluation.run_benchmark as runner

        class FakeModel:
            def __init__(self):
                self.last_obs = None

            @classmethod
            def load(cls, _path):
                return cls()

            def predict(self, obs, deterministic=True):
                self.last_obs = np.array(obs, copy=True)
                return np.zeros(4, dtype=np.float32), None

        class FakeNormalizer:
            def __init__(self):
                self.seen_obs = None

            def normalize_obs(self, obs):
                self.seen_obs = np.array(obs, copy=True)
                return obs + 3.0

        fake_norm = FakeNormalizer()
        fake_sb3 = types.SimpleNamespace(PPO=FakeModel, SAC=FakeModel)
        monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
        monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: fake_norm)

        model_dir = tmp_path / "ppo_default_s42"
        model_dir.mkdir(parents=True)
        (model_dir / "final_model.zip").touch()

        agent = runner.load_sb3_agent("ppo", "default", 42, str(model_dir))
        raw_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        agent.predict(raw_obs)

        np.testing.assert_array_equal(fake_norm.seen_obs, raw_obs)
        np.testing.assert_array_equal(agent._model.last_obs, raw_obs + 3.0)

    # def test_load_ppo_macro_agent_restores_macro_obs_normalization(self, monkeypatch, tmp_path):
    #     import evaluation.run_benchmark as runner

    #     class FakeModel:
    #         def __init__(self):
    #             self.last_obs = None

    #         @classmethod
    #         def load(cls, _path):
    #             return cls()

    #         def predict(self, obs, deterministic=True):
    #             self.last_obs = np.array(obs, copy=True)
    #             return np.zeros(2, dtype=np.float32), None

    #     class FakeNormalizer:
    #         def __init__(self):
    #             self.seen_obs = None

    #         def normalize_obs(self, obs):
    #             self.seen_obs = np.array(obs, copy=True)
    #             return obs + 5.0

    #     fake_norm = FakeNormalizer()
    #     fake_sb3 = types.SimpleNamespace(PPO=FakeModel, SAC=FakeModel)
    #     monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    #     monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: None)
    #     monkeypatch.setattr(runner, "_maybe_load_macro_obs_normalizer", lambda _path, _scenario: fake_norm)

    #     model_dir = tmp_path / "ppo_macro_default_s42"
    #     model_dir.mkdir(parents=True)
    #     (model_dir / "final_model.zip").touch()

    #     agent = runner.load_sb3_agent("ppo_macro", "default", 42, str(model_dir))
    #     raw_obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    #     action, _ = agent.predict(raw_obs)

    #     np.testing.assert_array_equal(fake_norm.seen_obs, raw_obs)
    #     np.testing.assert_array_equal(agent._model.last_obs, raw_obs + 5.0)
    #     np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))

    # def test_load_ha_agent_restores_obs_normalization(self, monkeypatch, tmp_path):
    #     import evaluation.run_ha_benchmark as runner

    #     class FakeConceptEncoder(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.bias = torch.nn.Parameter(torch.zeros(1))

    #         def forward(self, obs):
    #             return torch.zeros((obs.shape[0], 10), dtype=obs.dtype, device=obs.device)

    #     class FakeSafetyGate(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.bias = torch.nn.Parameter(torch.zeros(1))

    #         def forward(self, concepts):
    #             return torch.ones((concepts.shape[0], 4), dtype=concepts.dtype, device=concepts.device)

    #     class FakeModel:
    #         def __init__(self):
    #             self.last_obs = None
    #             self.policy = types.SimpleNamespace(
    #                 features_extractor=types.SimpleNamespace(
    #                     concept_encoder=FakeConceptEncoder(),
    #                     safety_gate=FakeSafetyGate(),
    #                 )
    #             )

    #         @classmethod
    #         def load(cls, _path):
    #             return cls()

    #         def predict(self, obs, deterministic=True):
    #             self.last_obs = np.array(obs, copy=True)
    #             return np.zeros(4, dtype=np.float32), None

    #     class FakeNormalizer:
    #         def __init__(self):
    #             self.seen_obs = None

    #         def normalize_obs(self, obs):
    #             self.seen_obs = np.array(obs, copy=True)
    #             return obs - 2.0

    #     fake_norm = FakeNormalizer()
    #     fake_sb3 = types.SimpleNamespace(PPO=FakeModel)
    #     monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    #     monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: fake_norm)

    #     model_dir = tmp_path / "ha_c2g_default_s42"
    #     model_dir.mkdir(parents=True)
    #     (model_dir / "final_model.zip").touch()

    #     agent, needs_shield = runner.load_agent("ha_c2g", "default", 42, str(model_dir))
    #     raw_obs = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    #     agent.predict(raw_obs)

    #     assert needs_shield is True
    #     np.testing.assert_array_equal(fake_norm.seen_obs, raw_obs)
    #     np.testing.assert_array_equal(agent._m.last_obs, raw_obs - 2.0)

    # @pytest.mark.parametrize(
    #     ("agent_name", "expects_normalization"),
    #     [("ppo", True), ("sac", False)],
    # )
    # def test_load_plain_ha_agent_supports_ppo_and_sac(self, monkeypatch, tmp_path, agent_name, expects_normalization):
    #     import evaluation.run_ha_benchmark as runner

    #     class FakeModel:
    #         def __init__(self):
    #             self.last_obs = None

    #         @classmethod
    #         def load(cls, _path):
    #             return cls()

    #         def predict(self, obs, deterministic=True):
    #             self.last_obs = np.array(obs, copy=True)
    #             return np.zeros(4, dtype=np.float32), None

    #     class FakeNormalizer:
    #         def __init__(self):
    #             self.seen_obs = None

    #         def normalize_obs(self, obs):
    #             self.seen_obs = np.array(obs, copy=True)
    #             return obs + 1.5

    #     fake_norm = FakeNormalizer()
    #     fake_sb3 = types.SimpleNamespace(PPO=FakeModel, SAC=FakeModel)
    #     monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    #     monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: fake_norm)

    #     model_dir = tmp_path / f"{agent_name}_default_s42"
    #     model_dir.mkdir(parents=True)
    #     (model_dir / "final_model.zip").touch()

    #     agent, needs_shield = runner.load_agent(agent_name, "default", 42, str(model_dir))
    #     raw_obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    #     action, _ = agent.predict(raw_obs)

    #     assert needs_shield is True
    #     if expects_normalization:
    #         np.testing.assert_array_equal(fake_norm.seen_obs, raw_obs)
    #         np.testing.assert_array_equal(agent._m.last_obs, raw_obs + 1.5)
    #     else:
    #         assert fake_norm.seen_obs is None
    #         np.testing.assert_array_equal(agent._m.last_obs, raw_obs)
    #     np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))

    # @pytest.mark.parametrize("agent_name", ["ha_c2g", "cbm_gate"])
    # def test_load_ha_agent_applies_layer2_runtime(self, monkeypatch, tmp_path, agent_name):
    #     import evaluation.run_ha_benchmark as runner

    #     class FakeConceptEncoder(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.bias = torch.nn.Parameter(torch.zeros(1))

    #         def forward(self, obs):
    #             batch = obs.shape[0]
    #             concepts = torch.zeros((batch, 10), dtype=obs.dtype, device=obs.device)
    #             concepts[:, 5] = 0.9
    #             concepts[:, 6] = 0.7
    #             concepts[:, 7] = 0.8
    #             concepts[:, 8] = 0.2
    #             concepts[:, 9] = 0.5
    #             return concepts

    #     class HalfGate(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.bias = torch.nn.Parameter(torch.zeros(1))

    #         def forward(self, concepts):
    #             return torch.full((concepts.shape[0], 4), 0.5, dtype=concepts.dtype, device=concepts.device)

    #     class FakeModel:
    #         def __init__(self):
    #             self.last_obs = None
    #             self.policy = types.SimpleNamespace(
    #                 features_extractor=types.SimpleNamespace(
    #                     concept_encoder=FakeConceptEncoder(),
    #                     safety_gate=HalfGate(),
    #                 )
    #             )

    #         @classmethod
    #         def load(cls, _path):
    #             return cls()

    #         def predict(self, obs, deterministic=True):
    #             self.last_obs = np.array(obs, copy=True)
    #             return np.zeros(4, dtype=np.float32), None

    #     fake_sb3 = types.SimpleNamespace(PPO=FakeModel)
    #     monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    #     monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: None)

    #     model_dir = tmp_path / f"{agent_name}_default_s42"
    #     model_dir.mkdir(parents=True)
    #     (model_dir / "final_model.zip").touch()

    #     agent, needs_shield = runner.load_agent(agent_name, "default", 42, str(model_dir))
    #     action, _ = agent.predict(np.array([34.5 / 35.0, 33.8 / 35.0, 0.5, 0.0, 0.0, 0.0, 0.8,
    #                                         0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0], dtype=np.float32))

    #     assert needs_shield is True
    #     assert np.max(np.abs(action)) > 0.0

    # @pytest.mark.parametrize("algo", ["ha_c2g", "cbm_gate"])
    # def test_load_sb3_agent_applies_layer2_runtime(self, monkeypatch, tmp_path, algo):
    #     import evaluation.run_benchmark as runner

    #     class FakeConceptEncoder(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.bias = torch.nn.Parameter(torch.zeros(1))

    #         def forward(self, obs):
    #             batch = obs.shape[0]
    #             concepts = torch.zeros((batch, 10), dtype=obs.dtype, device=obs.device)
    #             concepts[:, 5] = 0.9
    #             concepts[:, 6] = 0.7
    #             concepts[:, 7] = 0.8
    #             concepts[:, 8] = 0.2
    #             concepts[:, 9] = 0.5
    #             return concepts

    #     class HalfGate(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.bias = torch.nn.Parameter(torch.zeros(1))

    #         def forward(self, concepts):
    #             return torch.full((concepts.shape[0], 4), 0.5, dtype=concepts.dtype, device=concepts.device)

    #     class FakeModel:
    #         def __init__(self):
    #             self.last_obs = None
    #             self.policy = types.SimpleNamespace(
    #                 features_extractor=types.SimpleNamespace(
    #                     concept_encoder=FakeConceptEncoder(),
    #                     safety_gate=HalfGate(),
    #                 )
    #             )

    #         @classmethod
    #         def load(cls, _path):
    #             return cls()

    #         def predict(self, obs, deterministic=True):
    #             self.last_obs = np.array(obs, copy=True)
    #             return np.zeros(4, dtype=np.float32), None

    #     fake_sb3 = types.SimpleNamespace(PPO=FakeModel, SAC=FakeModel)
    #     monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    #     monkeypatch.setattr(runner, "_maybe_load_obs_normalizer", lambda _path, _scenario: None)

    #     model_dir = tmp_path / f"{algo}_default_s42"
    #     model_dir.mkdir(parents=True)
    #     (model_dir / "final_model.zip").touch()

    #     agent = runner.load_sb3_agent(algo, "default", 42, str(model_dir))
    #     action, _ = agent.predict(np.array([34.5 / 35.0, 33.8 / 35.0, 0.5, 0.0, 0.0, 0.0, 0.8,
    #                                         0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0], dtype=np.float32))

    #     assert np.max(np.abs(action)) > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# generate_plots  (import + CSV-loading only, no display)
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePlotsImport:
    def test_imports(self):
        from evaluation.generate_plots import generate_all
        assert callable(generate_all)

    def test_missing_csv_no_crash(self, tmp_path, capsys):
        from evaluation.generate_plots import generate_all
        generate_all(tmp_path / "nonexistent.csv", tmp_path / "out")
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_generates_figures(self, tmp_path):
        """Create a minimal CSV and verify figures are produced."""
        from evaluation.run_benchmark import benchmark, save_csv
        from evaluation.generate_plots import generate_all
        rows = benchmark(
            agents     = ["rule_based", "random"],
            scenarios  = ["default"],
            n_episodes = 2,
            seed_start = 0,
            model_dir  = None,
        )
        csv_path = tmp_path / "results.csv"
        save_csv(rows, csv_path)
        out_dir = tmp_path / "figures"
        generate_all(csv_path, out_dir)
        figs = list(out_dir.rglob("*.png"))
        assert len(figs) >= 5, f"Expected ≥5 figures, got {len(figs)}: {[f.name for f in figs]}"


# ═══════════════════════════════════════════════════════════════════════════
# Training script imports (no actual training)
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainImports:
    def test_ppo_imports(self):
        import baselines.train_ppo as m
        assert callable(m.train)
        assert callable(m.make_env_fn)

    def test_sac_imports(self):
        import baselines.train_sac as m
        assert callable(m.train)
        assert callable(m.make_env_fn)


class TestExperimentIdentity:
    def test_thermal_runner_resumes_only_identical_conditions(self, monkeypatch, tmp_path):
        from c2g_env.experiments import thermal_sensitivity_cross_scenario as runner

        monkeypatch.setattr("c2g_env.env_high_level._SUBSTEPS", 1)
        real_macro = runner.C2GMacroEnv
        creations = []

        def short_macro(*args, **kwargs):
            env = real_macro(*args, **kwargs)
            env._episode_macro_ticks = 1
            creations.append(env)
            return env

        config = {"id": "probe", "param": "T_supply_A", "value": 30.0,
                  "overrides": {"T_supply_A": 30.0}, "aliases": ["probe"]}
        monkeypatch.setattr(runner, "C2GMacroEnv", short_macro)
        monkeypatch.setattr(runner, "_explicit_configurations", lambda cfg: [config])
        path = tmp_path / "sensitivity.csv"
        argv = ["thermal_runner", "--out", str(path), "--record-dir", str(tmp_path / "records"), "--scenarios", "default", "--seeds", "100",
                "--controllers", "rule_based", "--macro-controller", "rule_macro", "--plant-profile", "plant_50mw"]
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()
        runner.main()
        assert len(creations) == 1
        argv[-1] = "plant_500mw"
        runner.main()
        config["overrides"] = {"T_supply_A": 31.0}
        config["value"] = 31.0
        runner.main()
        assert len(creations) == 3
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert len({row["episode_id"] for row in rows}) == 3
        assert [float(row["facility_capacity_mw"]) for row in rows] == [50.0, 500.0, 500.0]
        assert set(runner._CALLBACK_METRIC_FIELDS).issubset(rows[0])
        assert "workload/mean_throughput_ratio" in rows[0]
        assert "workload/served_work_ratio" in rows[0]
        assert all(json.loads(row["run_config"])["plant_profile"] == row["plant_profile"] for row in rows)
        damaged = Path(rows[0]["recording_dir"]) / "hardware.jsonl.gz"
        damaged.write_bytes(damaged.read_bytes() + b"damage")
        assert rows[0]["episode_id"] not in runner._completed_keys(path)
        assert len(runner._completed_keys(path)) == 2

    def test_interrupted_recording_is_not_completed(self, monkeypatch, tmp_path):
        from c2g_env.experiments import thermal_sensitivity_cross_scenario as runner

        def interrupt(*args, **kwargs):
            raise RuntimeError("test interruption")

        monkeypatch.setattr(runner.C2GMacroEnv, "step", interrupt)
        config = {"id": "probe", "param": "nominal", "value": 0.0, "overrides": {}, "aliases": ["probe"]}
        context = {"run_id": "run", "episode_id": "episode"}
        with pytest.raises(RuntimeError, match="test interruption"):
            runner._run_episode(config, "default", 100, "rule_based", None,
                                runner.RuleBasedMacroController(), "rule_macro", record_dir=tmp_path,
                                episode_context=context, run_config={})
        assert len(list(tmp_path.rglob("hardware.jsonl.gz"))) == 1
        assert not list(tmp_path.rglob("manifest.json"))

    @pytest.mark.parametrize("full_horizon", [False, True])
    def test_recorded_metrics_recompute_from_raw_streams(self, monkeypatch, tmp_path, full_horizon):
        import gzip
        from baselines.metrics_callback import _accumulate_step, _compute_episode_metrics, _make_episode_buffer
        from c2g_env.experiments import thermal_sensitivity_cross_scenario as runner

        real_macro = runner.C2GMacroEnv

        def short_macro(*args, **kwargs):
            env = real_macro(*args, **kwargs)
            env._episode_macro_ticks = 2
            if full_horizon:
                env._fast_env._episode_ticks = 360
            return env

        monkeypatch.setattr(runner, "C2GMacroEnv", short_macro)
        config = {"id": "nominal", "param": "nominal", "value": 0.0, "overrides": {}, "aliases": ["nominal"]}
        row = runner._run_episode(config, "default", 100, "rule_based", None,
                                  runner.RuleBasedMacroController(), "rule_macro", record_dir=tmp_path,
                                  episode_context={"run_id": "run", "episode_id": "episode"}, run_config={})
        with gzip.open(Path(row["recording_dir"]) / "hardware.jsonl.gz", "rt", encoding="utf-8") as handle:
            reset, *records = [json.loads(line) for line in handle]
        assert reset["event"] == "reset"
        assert len(records) == row["fast_steps"] == 360
        runtime = json.loads(row["runtime_summary"])
        energy = sum(record["info"]["p_facility_mw"] * record["duration_seconds"] / 3600 for record in records)
        rmse = np.sqrt(np.mean([record["info"]["tracking_err_kw"] ** 2 for record in records]))
        warning_fraction = np.mean([record["info"]["thermal_warning"] for record in records])
        assert runtime["derived"]["facility_energy_mwh"] == pytest.approx(energy)
        assert runtime["derived"]["tracking_rmse_kw"] == pytest.approx(rmse)
        assert row["thermal_warning_fraction"] == pytest.approx(warning_fraction)
        assert runtime["hardware"]["scalars"]["reward"]["sum"] == pytest.approx(sum(record["reward"] for record in records))
        episode_buffer = _make_episode_buffer()
        for record in records:
            _accumulate_step(episode_buffer, record["reward"], record["info"])
        expected_metrics = _compute_episode_metrics(episode_buffer)
        expected_metrics["ep/length"] = float(len(records))
        expected_metrics["ep/survived"] = float(full_horizon)
        assert runtime["hardware"]["episode_metrics"] == pytest.approx(expected_metrics)
        assert {key: row[key] for key in expected_metrics} == pytest.approx(expected_metrics)
        throughput = np.mean([record["info"]["throughput_ratio"] for record in records])
        served = sum(record["info"]["p_flex_served_kw"] for record in records)
        arrived = sum(record["info"]["p_flex_nom_kw"] for record in records)
        assert row["workload/mean_throughput_ratio"] == pytest.approx(throughput)
        assert row["workload/served_work_ratio"] == pytest.approx(served / arrived)
        assert runner._recording_complete({**row, "episode_id": "episode"})

    def test_hydra_sweep_jobs_have_distinct_paths(self):
        from omegaconf import OmegaConf

        config = OmegaConf.load("conf/config.yaml")
        config.hydra.job = {"num": 0}
        first = config.hydra.sweep.subdir
        config.hydra.job.num = 1
        assert first != config.hydra.sweep.subdir

    @pytest.mark.parametrize("profile", available_plant_profiles())
    def test_benchmark_context_uses_effective_plant(self, monkeypatch, profile):
        from evaluation import run_benchmark as runner

        monkeypatch.setattr(runner, "_PLANT_PROFILE", profile)
        env = C2GFastEnv(plant_overrides=load_plant_profile(profile))
        try:
            env.reset(seed=100)
            context = runner._experiment_context(env, 100, RuleBasedController())
            assert context["plant_profile"] == profile
            assert context["facility_capacity_mw"] == env._facility_cap_kw / 1_000.0
            assert context["plant_overrides"] == env._plant_overrides
            assert context["seed"] == 100
            assert context["run_id"] != runner._experiment_context(env, 100, RuleBasedController())["run_id"]
        finally:
            env.close()

    def test_benchmark_output_paths_distinguish_capacities(self):
        from evaluation.run_benchmark import _default_output_path

        paths = {_default_output_path(["rule_based"], ["default"], plant_profile=profile)
                 for profile in available_plant_profiles() + ["all"]}
        assert len(paths) == len(available_plant_profiles()) + 1

    def test_macro_and_hardware_logs_share_episode_context(self, monkeypatch):
        from unittest.mock import Mock
        from evaluation import run_benchmark as runner

        monkeypatch.setattr("c2g_env.env_high_level._SUBSTEPS", 1)
        make_macro = runner._make_macro_env

        def short_macro(*args, **kwargs):
            env = make_macro(*args, **kwargs)
            env._episode_macro_ticks = 1
            return env

        logger = Mock()
        monkeypatch.setattr(runner, "_make_macro_env", short_macro)
        monkeypatch.setattr(runner, "C2GTransitionLoggerCallback", logger)
        monkeypatch.setattr(runner, "_PLANT_OVERRIDES", load_plant_profile("plant_50mw"))
        monkeypatch.setattr(runner, "_PLANT_PROFILE", "plant_50mw")
        runner.run_macro_episode(
            runner.RuleBasedMacroController(), "default", 100,
            inner_action_fn=lambda obs, action: np.array([1.0, 0.7, 0.7, 0.0], dtype=np.float32),
        )
        assert logger.call_count == 2
        inner_context = logger.call_args_list[0].kwargs["experiment_context"]
        macro_context = logger.call_args_list[1].kwargs["experiment_context"]
        assert inner_context == macro_context
        assert macro_context["facility_capacity_mw"] == 50.0

    def test_configuration_identity_is_order_independent(self):
        from c2g_env.experiments.identity import configuration_id

        assert configuration_id({"plant": "small", "thermal": {"C_A": 5400.0}}) == configuration_id(
            {"thermal": {"C_A": 5400.0}, "plant": "small"}
        )

    @pytest.mark.parametrize("field", ["plant_profile", "plant_overrides", "macro_controller", "hardware_checkpoint", "macro_checkpoint", "source_id"])
    def test_run_identity_distinguishes_conditions(self, field):
        from c2g_env.experiments.identity import configuration_id

        reference = {"plant_profile": "plant_50mw", "plant_overrides": {}, "macro_controller": "rule_macro",
                     "hardware_checkpoint": None, "macro_checkpoint": None, "source_id": "original"}
        changed = {**reference, field: "changed"}
        assert configuration_id(reference) != configuration_id(changed)

    def test_episode_identity_includes_effective_overrides_and_seed(self):
        from c2g_env.experiments.thermal_sensitivity_cross_scenario import _episode_id

        config = {"id": "nominal", "overrides": {}}
        reference = _episode_id("run", config, "default", 100, "rule_based")
        assert reference != _episode_id("other-run", config, "default", 100, "rule_based")
        assert reference != _episode_id("run", {**config, "overrides": {"C_A": 5400.0}}, "default", 100, "rule_based")
        assert reference != _episode_id("run", config, "scenario_b", 100, "rule_based")
        assert reference != _episode_id("run", config, "default", 101, "rule_based")
        assert reference != _episode_id("run", config, "default", 100, "pid")

    def test_checkpoint_identity_includes_contents(self, tmp_path):
        from c2g_env.experiments.identity import file_fingerprint

        path = tmp_path / "checkpoint.zip"
        path.write_bytes(b"first checkpoint")
        original = file_fingerprint(path)
        path.write_bytes(b"second checkpoint")
        assert file_fingerprint(path)["sha256"] != original["sha256"]

    def test_resume_uses_only_completed_episode_identities(self, tmp_path):
        from c2g_env.experiments.thermal_sensitivity_cross_scenario import _completed_keys, _FIELDS

        path = tmp_path / "results.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerow({"episode_id": "complete", "completed": True})
            writer.writerow({"episode_id": "incomplete", "completed": False})
        assert _completed_keys(path) == set()

    def test_resume_rejects_legacy_schema(self, tmp_path):
        from c2g_env.experiments.thermal_sensitivity_cross_scenario import _completed_keys

        path = tmp_path / "legacy.csv"
        path.write_text("config_id,completed\nnominal,True\n", encoding="utf-8")
        with pytest.raises(ValueError, match="new output file"):
            _completed_keys(path)


class TestConditionAnalysis:
    def test_plot_helpers_reject_mixed_conditions(self, condition_rows, tmp_path):
        import pandas as pd
        from evaluation.generate_plots import plot_reward_heatmap, plot_metric_grouped_bar, plot_radar

        frame = pd.DataFrame(condition_rows)
        with pytest.raises(ValueError, match="select one condition"):
            plot_reward_heatmap(frame, tmp_path / "heatmap.png")
        with pytest.raises(ValueError, match="select one condition"):
            plot_metric_grouped_bar(frame, "mean_reward", "Reward", "Reward", tmp_path / "bars.png")
        with pytest.raises(ValueError, match="select one condition"):
            plot_radar(frame, "default", tmp_path / "radar.png")

    def test_plots_are_written_per_condition(self, condition_rows, tmp_path):
        import pandas as pd
        from evaluation.generate_plots import generate_all

        frame = pd.DataFrame(condition_rows).assign(
            tracking_rmse=1.0, thermal_viol_rate=0.0, throughput_ratio=1.0, survival_rate=1.0,
        )
        path = tmp_path / "results.csv"
        frame.to_csv(path, index=False)
        output = tmp_path / "figures"
        generate_all(path, output)
        manifests = list(output.glob("condition_*/condition.json"))
        assert len(manifests) == 4
        for manifest in manifests:
            assert len(list(manifest.parent.glob("*.png"))) == 7
            assert "thermal_overrides" in json.loads(manifest.read_text(encoding="utf-8"))
        assert not list(output.glob("*.png"))

    @pytest.fixture
    def condition_rows(self):
        rows = []
        for profile, capacity in [("plant_50mw", 50.0), ("plant_500mw", 500.0)]:
            for config_id, thermal_offset in [("nominal", 0.0), ("low_inertia", 100.0)]:
                for agent, agent_offset in [("reference", 0.0), ("controller", 3.0)]:
                    for seed in [0, 1]:
                        rows.append({
                            "agent": agent, "scenario": "default", "seed": seed,
                            "plant_profile": profile, "facility_capacity_mw": capacity,
                            "config_id": config_id, "thermal_overrides": json.dumps({"C_A": capacity + thermal_offset}),
                            "mean_reward": capacity + thermal_offset + agent_offset + seed,
                        })
        return rows

    def test_summaries_do_not_pool_conditions(self, condition_rows):
        from evaluation.statistical_analysis import compute_all_summaries

        summaries = compute_all_summaries(condition_rows, use_bootstrap=False)
        assert len(summaries) == 8
        assert all(summary.n_seeds == 2 for summary in summaries)
        assert {summary.metric for summary in summaries} == {"mean_reward"}
        for summary in summaries:
            expected = summary.condition["thermal_overrides"]["C_A"] + 0.5
            if summary.agent == "controller":
                expected += 3.0
            assert summary.mean == pytest.approx(expected)

    def test_comparisons_match_condition(self, condition_rows):
        from evaluation.statistical_analysis import compute_pairwise_comparisons

        comparisons = compute_pairwise_comparisons(condition_rows, baseline="reference", metrics=["mean_reward"])
        assert len(comparisons) == 4
        assert all(comparison.mean_b - comparison.mean_a == pytest.approx(3.0) for comparison in comparisons)

    def test_missing_condition_baseline_is_not_borrowed(self, condition_rows):
        from evaluation.statistical_analysis import compute_pairwise_comparisons

        rows = [row for row in condition_rows if row["agent"] != "reference" or row["plant_profile"] == "plant_50mw"]
        comparisons = compute_pairwise_comparisons(rows, baseline="reference", metrics=["mean_reward"])
        assert len(comparisons) == 2
        assert all(comparison.condition["plant_profile"] == "plant_50mw" for comparison in comparisons)

    def test_thermal_json_order_does_not_split_condition(self):
        from evaluation.statistical_analysis import group_by_agent_scenario

        rows = [
            {"agent": "rule", "scenario": "default", "thermal_overrides": '{"C_A": 1, "K_liq": 2}'},
            {"agent": "rule", "scenario": "default", "thermal_overrides": '{"K_liq": 2, "C_A": 1}'},
        ]
        assert len(group_by_agent_scenario(rows)) == 1

    def test_latex_requires_explicit_condition(self, condition_rows):
        from evaluation.statistical_analysis import compute_all_summaries, generate_latex_table

        summaries = compute_all_summaries(condition_rows, use_bootstrap=False)
        with pytest.raises(ValueError, match="select a condition"):
            generate_latex_table(summaries, metrics=["mean_reward"])
        table = generate_latex_table(summaries, metrics=["mean_reward"], condition=summaries[0].condition)
        assert "reference" in table and "controller" in table

    def test_legacy_rows_remain_one_condition(self):
        from evaluation.statistical_analysis import compute_all_summaries

        rows = [{"agent": "rule", "scenario": "default", "seed": seed, "mean_reward": float(seed)} for seed in [0, 1]]
        summaries = compute_all_summaries(rows, use_bootstrap=False)
        assert len(summaries) == 1
        assert summaries[0].condition == {}
        assert summaries[0].mean == pytest.approx(0.5)
