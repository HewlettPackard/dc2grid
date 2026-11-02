#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# scripts/run_sweep.sh  —  Full C2G-Bench Training & Evaluation Sweep
# ══════════════════════════════════════════════════════════════════════════════
# Runs: 4 scenarios × {PPO, SAC, PPO-Lag, BangBang, PID, MPC, RuleBased, Random} × 3 seeds
#        4 scenarios × {PPO-Macro, HRL, MPC-Macro, MILP, RuleBased-Macro} × 3 seeds
#   - PPO: 300k steps    (~8 min/run  on H100)
#   - SAC: 200k steps    (~6 min/run)
#   - Rule-based: no training, evaluation only
#   - Random: no training, evaluation only
#
# Parallelism: up to $MAX_PARALLEL background jobs.
# Results: results/sweep_results.csv  (one row per run)
#
# Usage:
#   bash scripts/run_sweep.sh              # full sweep
#   bash scripts/run_sweep.sh --dry-run    # print commands only
#   MAX_PARALLEL=16 bash scripts/run_sweep.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="${PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python3" ]]; then
    PYTHON="${ROOT}/.venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "ERROR: No usable python3 found. Set PYTHON=/abs/path/to/python3 and retry."
    exit 1
fi

RESULTS_DIR="${ROOT}/results"
RESULTS_CSV="${RESULTS_DIR}/sweep_results.csv"
MAX_PARALLEL=${MAX_PARALLEL:-4}
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCENARIOS=(default scenario_a scenario_b scenario_c)
SEEDS=(1 2 3)

mkdir -p "$RESULTS_DIR"

# ── Write CSV header ─────────────────────────────────────────────────────────
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "algo,scenario,seed,total_reward,mean_reward,mean_pue,max_temp_A,max_temp_B,viol_rate,mean_soc,tracking_rmse_kw,mean_lmp,survived,ep_length,wall_seconds" \
        > "$RESULTS_CSV"
fi

# ── Evaluate a trained/rule-based/random agent ────────────────────────────────
# Writes one CSV row per (algo, scenario, seed)
evaluate() {
    local algo=$1 scenario=$2 seed=$3 model_path=${4:-}
    echo "[eval] ${algo} / ${scenario} / seed=${seed}"

    $PYTHON - "$algo" "$scenario" "$seed" "$model_path" << 'PYEOF'
import sys, time, csv, numpy as np
from pathlib import Path
import torch

algo       = sys.argv[1]
scenario   = sys.argv[2]
seed       = int(sys.argv[3])
model_path = sys.argv[4] if sys.argv[4] else None

from c2g_env import C2GFastEnv
from baselines.safety.safe_projection import compute_layer2_action

env = C2GFastEnv(scenario=scenario)

# Load agent
if algo == "random":
    class RandomAgent:
        def predict(self, obs, deterministic=False):
            return env.action_space.sample(), None
    agent = RandomAgent()
elif algo == "rule_based":
    from baselines.rule_based_mpc import RuleBasedController
    agent = RuleBasedController()
elif algo == "bang_bang":
    from baselines.bang_bang import BangBangController
    agent = BangBangController()
elif algo == "pid":
    from baselines.pid_controller import PIDController
    agent = PIDController()
elif algo == "mpc_fast":
    from baselines.mpc_fast import MPCFastController
    agent = MPCFastController()
else:
    # RL agent — load from zip and restore observation normalization when used
    model_root = Path(model_path) if model_path else None
    if model_root is None:
        print(f"  SKIP: no model path provided for {algo}")
        exit(0)
    if model_root.is_dir():
        model_root = model_root / "final_model"
    elif model_root.suffix == ".zip":
        model_root = model_root.with_suffix("")

    if not model_root.with_suffix(".zip").exists():
        print(f"  SKIP: missing model at {model_root}.zip")
        exit(0)

    ppo_like_algos = {
        "ppo", "ppo_lag", "cbf_ppo", "hj_ppo", "mpcsf_ppo",
        "cpo", "reward_shaping", "ha_c2g", "cbm_only", "cbm_gate", "cbm_shield",
    }
    obs_normalizer = None

    if algo in ppo_like_algos:
        from stable_baselines3 import PPO as AlgoCls
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        stats_path = model_root.parent / "vec_normalize.pkl"
        if stats_path.exists():
            stats_env = DummyVecEnv([lambda: C2GFastEnv(scenario=scenario)])
            obs_normalizer = VecNormalize.load(str(stats_path), stats_env)
            obs_normalizer.training = False
            obs_normalizer.norm_reward = False
    else:
        from stable_baselines3 import SAC as AlgoCls

    model = AlgoCls.load(str(model_root))

    class SB3Agent:
        def __init__(self, model, obs_normalizer=None):
            self.model = model
            self.obs_normalizer = obs_normalizer

        def predict(self, obs, deterministic=True):
            if self.obs_normalizer is not None:
                obs = self.obs_normalizer.normalize_obs(np.asarray(obs, dtype=np.float32))
            return self.model.predict(obs, deterministic=deterministic)

    class HAC2GAgent(SB3Agent):
        def __init__(self, model, concept_encoder, safety_gate, obs_normalizer=None):
            super().__init__(model, obs_normalizer=obs_normalizer)
            self.concept_encoder = concept_encoder
            self.safety_gate = safety_gate

        def predict(self, obs, deterministic=True):
            action, state = super().predict(obs, deterministic=deterministic)
            raw_obs = np.asarray(obs, dtype=np.float32)
            with torch.no_grad():
                device = next(self.concept_encoder.parameters()).device
                obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                concepts = self.concept_encoder(obs_t)
                gated_action, _, _ = compute_layer2_action(
                    action_t,
                    concepts,
                    obs=obs_t,
                    safety_gate=self.safety_gate,
                )
            return gated_action.squeeze(0).detach().cpu().numpy(), state

    if algo in {"ha_c2g", "cbm_gate"}:
        fe = model.policy.features_extractor
        agent = HAC2GAgent(
            model,
            concept_encoder=fe.concept_encoder,
            safety_gate=fe.safety_gate,
            obs_normalizer=obs_normalizer,
        )
    else:
        agent = SB3Agent(model, obs_normalizer=obs_normalizer)

# Run 3 eval episodes and average
N_EVAL = 3
all_metrics = []
for ep in range(N_EVAL):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    ep_reward = 0.0
    temps_A, temps_B, pues, socs = [], [], [], []
    tracking_errs, lmps = [], []
    steps = 0
    survived = True
    t0 = time.time()
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        temps_A.append(info.get("temp_A", 0.0))
        temps_B.append(info.get("temp_B", 0.0))
        pues.append(info.get("pue", 1.0))
        socs.append(info.get("bess_soc", 0.5))
        tracking_errs.append(info.get("tracking_err_kw", 0.0) ** 2)
        lmps.append(info.get("lmp", 0.0))
        steps += 1
        if terminated or truncated:
            survived = not terminated
            break
    wall = time.time() - t0

    all_metrics.append({
        "total_reward":    ep_reward,
        "mean_reward":     ep_reward / max(steps, 1),
        "mean_pue":        np.mean(pues) if pues else 1.0,
        "max_temp_A":      np.max(temps_A) if temps_A else 0.0,
        "max_temp_B":      np.max(temps_B) if temps_B else 0.0,
        "viol_rate":       np.mean([t > 33.0 for t in temps_A]) if temps_A else 0.0,
        "mean_soc":        np.mean(socs) if socs else 0.5,
        "tracking_rmse_kw":np.sqrt(np.mean(tracking_errs)) if tracking_errs else 0.0,
        "mean_lmp":        np.mean(lmps) if lmps else 0.0,
        "survived":        int(survived),
        "ep_length":       steps,
        "wall_seconds":    wall,
    })

# Average across eval episodes
avg = {}
for k in all_metrics[0]:
    vals = [m[k] for m in all_metrics]
    avg[k] = np.mean(vals)

csv_path = Path("results/sweep_results.csv")
# Upsert: remove any prior row for this (algo, scenario, seed) before appending
import pandas as _pd, io as _io
_header = "algo,scenario,seed,total_reward,mean_reward,mean_pue,max_temp_A,max_temp_B,viol_rate,mean_soc,tracking_rmse_kw,mean_lmp,survived,ep_length,wall_seconds"
if csv_path.exists():
    _df = _pd.read_csv(csv_path)
    _df = _df[~((_df.algo == algo) & (_df.scenario == scenario) & (_df.seed == int(seed)))]
else:
    _df = _pd.read_csv(_io.StringIO(_header + "\n"))
with open(csv_path, "w", newline="") as _f:
    _df.to_csv(_f, index=False)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        algo, scenario, seed,
        f"{avg['total_reward']:.1f}",
        f"{avg['mean_reward']:.4f}",
        f"{avg['mean_pue']:.4f}",
        f"{avg['max_temp_A']:.2f}",
        f"{avg['max_temp_B']:.2f}",
        f"{avg['viol_rate']:.4f}",
        f"{avg['mean_soc']:.4f}",
        f"{avg['tracking_rmse_kw']:.1f}",
        f"{avg['mean_lmp']:.2f}",
        f"{avg['survived']:.0f}",
        f"{avg['ep_length']:.0f}",
        f"{avg['wall_seconds']:.1f}",
    ])
print(f"  reward={avg['total_reward']:.1f}  pue={avg['mean_pue']:.3f}  "
      f"maxT={avg['max_temp_A']:.1f}/{avg['max_temp_B']:.1f}  "
      f"survived={avg['survived']:.0f}")
PYEOF
}

# ── Job counter for parallelism ──────────────────────────────────────────────
NJOBS=0
# ── Evaluate a macro-level agent ──────────────────────────────────────────────
evaluate_macro() {
    local algo=$1 scenario=$2 seed=$3 model_path=${4:-}
    echo "[eval-macro] ${algo} / ${scenario} / seed=${seed}"

    $PYTHON - "$algo" "$scenario" "$seed" "$model_path" << 'PYMEOF'
import sys, time, csv, numpy as np
from pathlib import Path

algo       = sys.argv[1]
scenario   = sys.argv[2]
seed       = int(sys.argv[3])
model_path = sys.argv[4] if sys.argv[4] else None

from c2g_env import C2GMacroEnv

env = C2GMacroEnv(scenario=scenario)

# Load agent
if algo == "rule_based_macro":
    from baselines.rule_based_macro import RuleBasedMacroController
    agent = RuleBasedMacroController()
elif algo == "random_macro":
    class RandomAgent:
        def predict(self, obs, deterministic=False):
            return env.action_space.sample(), None
    agent = RandomAgent()
elif algo == "mpc_macro":
    from baselines.mpc_macro import MPCMacroController
    agent = MPCMacroController()
elif algo == "milp":
    from baselines.milp_dispatch import MILPDispatchController
    agent = MILPDispatchController()
else:
    # RL agent (PPO-Macro or HRL)
    from stable_baselines3 import PPO as AlgoCls
    agent = AlgoCls.load(model_path, env=env)

# Run 3 eval episodes and average
N_EVAL = 3
all_metrics = []
for ep in range(N_EVAL):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    ep_reward = 0.0
    steps = 0
    survived = True
    t0 = time.time()
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        if terminated or truncated:
            survived = not terminated
            break
    wall = time.time() - t0

    all_metrics.append({
        "total_reward":    ep_reward,
        "mean_reward":     ep_reward / max(steps, 1),
        "mean_lmp":        info.get("mean_lmp", 0.0),
        "committed_mw":    info.get("committed_mw", 0.0),
        "mean_tracking_err": info.get("mean_tracking_err", 0.0),
        "survived":        int(survived),
        "ep_length":       steps,
        "wall_seconds":    wall,
    })

avg = {}
for k in all_metrics[0]:
    avg[k] = np.mean([m[k] for m in all_metrics])

csv_path = Path("results/sweep_results_macro.csv")
_header = "algo,scenario,seed,total_reward,mean_reward,mean_lmp,committed_mw,mean_tracking_err,survived,ep_length,wall_seconds"
import pandas as _pd, io as _io
if csv_path.exists():
    _df = _pd.read_csv(csv_path)
    _df = _df[~((_df.algo == algo) & (_df.scenario == scenario) & (_df.seed == int(seed)))]
else:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _df = _pd.read_csv(_io.StringIO(_header + "\n"))
with open(csv_path, "w", newline="") as _f:
    _df.to_csv(_f, index=False)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        algo, scenario, seed,
        f"{avg['total_reward']:.1f}",
        f"{avg['mean_reward']:.4f}",
        f"{avg['mean_lmp']:.2f}",
        f"{avg['committed_mw']:.2f}",
        f"{avg['mean_tracking_err']:.1f}",
        f"{avg['survived']:.0f}",
        f"{avg['ep_length']:.0f}",
        f"{avg['wall_seconds']:.1f}",
    ])
print(f"  reward={avg['total_reward']:.1f}  survived={avg['survived']:.0f}")
PYMEOF
}

wait_for_slots() {
    while (( NJOBS >= MAX_PARALLEL )); do
        wait -n 2>/dev/null || true
        NJOBS=$((NJOBS - 1))
    done
}

echo "═══════════════════════════════════════════════════════════════"
echo "  C2G-Bench Full Training & Evaluation Sweep"
echo "  Scenarios: ${SCENARIOS[*]}"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Algos:     PPO, SAC, RuleBased, Random + HA: CBF, HJ, MPCSF, CPO, Shield-RS, HA-C2G"
echo "  Python:    ${PYTHON}"
echo "  Parallel:  ${MAX_PARALLEL} jobs"
echo "  Output:    ${RESULTS_CSV}"
echo "═══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Rule-based & Random (no training needed, fast)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 1: Evaluating Rule-Based & Random baselines …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for algo in rule_based random; do
            if $DRY_RUN; then
                echo "  [dry-run] evaluate $algo $scenario $seed"
            else
                wait_for_slots
                evaluate "$algo" "$scenario" "$seed" "" &
                NJOBS=$((NJOBS + 1))
            fi
        done
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: PPO training + evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 2: PPO training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train PPO $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] PPO / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/ppo_${scenario}/seed_${seed}"
            # Run Hydra training
            $PYTHON baselines/train_ppo.py \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            # Find the latest run's final model
            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                MODEL="${LATEST%.zip}"
                evaluate "ppo" "$scenario" "$seed" "$MODEL"
            else
                echo "  [WARN] PPO ${scenario}/seed_${seed}: no final_model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: SAC training + evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 3: SAC training (200k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train SAC $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] SAC / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/sac_${scenario}/seed_${seed}"
            $PYTHON baselines/train_sac.py \
                algo=sac \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                MODEL="${LATEST%.zip}"
                evaluate "sac" "$scenario" "$seed" "$MODEL"
            else
                echo "  [WARN] SAC ${scenario}/seed_${seed}: no final_model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Macro-level Rule-Based & Random baselines
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 4: Evaluating Macro Rule-Based baselines …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] evaluate rule_based_macro $scenario $seed"
        else
            wait_for_slots
            evaluate_macro "rule_based_macro" "$scenario" "$seed" "" &
            NJOBS=$((NJOBS + 1))
        fi
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: PPO-Macro training + evaluation (macro env, fixed inner defaults)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 5: PPO-Macro training (100k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train PPO-Macro $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] PPO-Macro / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/ppo_macro_${scenario}/seed_${seed}"
            $PYTHON baselines/train_ppo_macro.py \
                algo=ppo_macro \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                MODEL="${LATEST%.zip}"
                evaluate_macro "ppo_macro" "$scenario" "$seed" "$MODEL"
            else
                echo "  [WARN] PPO-Macro ${scenario}/seed_${seed}: no final_model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: Hierarchical RL training (low-level PPO → frozen → macro PPO)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 6: HRL sequential training (300k + 100k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train HRL $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] HRL / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/hrl_${scenario}/seed_${seed}"
            $PYTHON baselines/train_hierarchical.py \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                MODEL="${LATEST%.zip}"
                evaluate_macro "hrl" "$scenario" "$seed" "$MODEL"
            else
                echo "  [WARN] HRL ${scenario}/seed_${seed}: no final_model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# Phase 7: Bang-Bang, PID, MPC evaluation (no training, fast env)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 7: Evaluating Bang-Bang, PID, MPC-Fast baselines …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for algo in bang_bang pid mpc_fast; do
            if $DRY_RUN; then
                echo "  [dry-run] evaluate $algo $scenario $seed"
            else
                wait_for_slots
                evaluate "$algo" "$scenario" "$seed" "" &
                NJOBS=$((NJOBS + 1))
            fi
        done
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 8: MPC-Macro & MILP evaluation (no training, macro env)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 8: Evaluating MPC-Macro & MILP baselines …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for algo in mpc_macro milp; do
            if $DRY_RUN; then
                echo "  [dry-run] evaluate_macro $algo $scenario $seed"
            else
                wait_for_slots
                evaluate_macro "$algo" "$scenario" "$seed" "" &
                NJOBS=$((NJOBS + 1))
            fi
        done
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 9: PPO-Lagrangian training + evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 9: PPO-Lagrangian training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train PPO-Lagrangian $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            $PYTHON baselines/safety/train_ppo_lagrangian.py algo=ppo_lagrangian scenario="$scenario" experiment.seed="$seed"
            MODEL_DIR="outputs/ppo_lagrangian_${scenario}/seed_${seed}"
            LATEST=$(ls -td "${MODEL_DIR}/"*/ 2>/dev/null | head -1)
            if [[ -n "$LATEST" && -f "${LATEST}final_model.zip" ]]; then
                evaluate "ppo_lag" "$scenario" "$seed" "$LATEST"
            else
                echo "  WARN: No PPO-Lagrangian model for $scenario/$seed"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 12: CBF-PPO training + evaluation (High-Assurance Tier 1)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 12: CBF-PPO training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train CBF-PPO $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] CBF-PPO / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/cbf_ppo_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_cbf_ppo.py \
                algo=cbf_ppo \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "cbf_ppo" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] CBF-PPO ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 13: HJ-PPO training + evaluation (High-Assurance Tier 1)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 13: HJ-PPO training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train HJ-PPO $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] HJ-PPO / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/hj_ppo_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_hj_ppo.py \
                algo=hj_ppo \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "hj_ppo" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] HJ-PPO ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 14: MPC-SF-PPO training + evaluation (High-Assurance Tier 1)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 14: MPC-SF-PPO training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train MPCSF-PPO $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] MPCSF-PPO / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/mpcsf_ppo_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_mpcsf_ppo.py \
                algo=mpcsf_ppo \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "mpcsf_ppo" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] MPCSF-PPO ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 15: CPO training + evaluation (High-Assurance Tier 2)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 15: CPO training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train CPO $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] CPO / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/cpo_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_cpo.py \
                algo=cpo \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "cpo" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] CPO ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 16: Shield Reward Shaping training + evaluation (High-Assurance Tier 2)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 16: Shield-Reward-Shaping training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train Shield-RS $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] Shield-RS / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/reward_shaping_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_shield_reward_shaping.py \
                algo=shield_reward_shaping \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "reward_shaping" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] Shield-RS ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 17: HA-C2G training + evaluation (High-Assurance Tier 3)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 17: HA-C2G neuro-symbolic training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train HA-C2G $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] HA-C2G / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/ha_c2g_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_ha_c2g.py \
                algo=ha_c2g \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "ha_c2g" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] HA-C2G ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 18: CBM-Only ablation training (Tier 3 Ablation)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 18: CBM-Only ablation training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train CBM-Only $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] CBM-Only / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/cbm_only_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_cbm_only.py \
                algo=cbm_only \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "cbm_only" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] CBM-Only ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 19: CBM+Gate ablation training (Tier 3 Ablation)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 19: CBM+Gate ablation training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train CBM+Gate $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] CBM+Gate / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/cbm_gate_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_cbm_gate.py \
                algo=cbm_gate \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "cbm_gate" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] CBM+Gate ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 20: CBM+Shield ablation training (Tier 3 Ablation)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 20: CBM+Shield ablation training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train CBM+Shield $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] CBM+Shield / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/cbm_shield_${scenario}/seed_${seed}"
            $PYTHON baselines/safety/train_cbm_shield.py \
                algo=cbm_shield \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                evaluate "cbm_shield" "$scenario" "$seed" "${LATEST%.zip}"
            else
                echo "  [WARN] CBM+Shield ${scenario}/seed_${seed}: no model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 21: HA Benchmark Evaluation (all shields × all scenarios)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 21: HA Benchmark evaluation (11 metrics) …"
if $DRY_RUN; then
    echo "  [dry-run] python evaluation/run_ha_benchmark.py --agents all --n_episodes 5"
else
    $PYTHON evaluation/run_ha_benchmark.py \
        --agents simplex cbf hj mpcsf \
        --n_episodes 5 \
        --output results/ha_benchmark_results.csv
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 22: Generate summary table
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 22: Generating results summary …"
if $DRY_RUN; then
    echo "  [dry-run] summarize results/sweep_results.csv"
else
    "$PYTHON" - << 'PYEOF'
import pandas as pd
from pathlib import Path

csv_path = Path("results/sweep_results.csv")
if not csv_path.exists():
    print("No results CSV found."); exit(1)

df = pd.read_csv(csv_path)
print(f"\nTotal runs: {len(df)}")
print(f"Algos:      {sorted(df.algo.unique())}")
print(f"Scenarios:  {sorted(df.scenario.unique())}")
print()

# Aggregate: mean ± std across seeds
cols = ["total_reward", "mean_pue", "max_temp_A", "tracking_rmse_kw", "survived"]
agg = df.groupby(["scenario", "algo"])[cols].agg(["mean", "std"]).round(2)
print(agg.to_string())
print()

# Macro-level results (if any)
csv_macro = Path("results/sweep_results_macro.csv")
if csv_macro.exists():
    df2 = pd.read_csv(csv_macro)
    print(f"\n\nMacro-level runs: {len(df2)}")
    print(f"Algos: {sorted(df2.algo.unique())}")
    mcols = ["total_reward", "mean_lmp", "committed_mw", "survived"]
    avail = [c for c in mcols if c in df2.columns]
    if avail:
        magg = df2.groupby(["scenario", "algo"])[avail].agg(["mean", "std"]).round(2)
        print(magg.to_string())
    print()

# LaTeX-ready table
print("═══ LaTeX table rows (paste into paper) ═══")
for scenario in ["default", "scenario_a", "scenario_b", "scenario_c"]:
    for algo in ["random", "bang_bang", "pid", "rule_based", "mpc_fast", "ppo", "sac", "ppo_lag", "cbf_ppo", "hj_ppo", "mpcsf_ppo", "cpo", "reward_shaping", "cbm_only", "cbm_gate", "cbm_shield", "ha_c2g"]:
        sub = df[(df.scenario == scenario) & (df.algo == algo)]
        if sub.empty:
            continue
        r   = sub.total_reward
        pue = sub.mean_pue
        trk = sub.tracking_rmse_kw
        vr  = sub.viol_rate
        sr  = sub.survived
        print(f"  {scenario:12} & {algo:12} & "
              f"${r.mean():.0f} \\pm {r.std():.0f}$ & "
              f"${pue.mean():.3f} \\pm {pue.std():.3f}$ & "
              f"${trk.mean():.0f} \\pm {trk.std():.0f}$ & "
              f"${vr.mean():.2f}$ & "
              f"${sr.mean():.2f}$ \\\\")

# Save summary too
summary = df.groupby(["scenario", "algo"])[["total_reward","mean_pue","max_temp_A",
    "tracking_rmse_kw","viol_rate","survived"]].agg(["mean","std"]).round(3)
summary.to_csv("results/sweep_summary.csv")
print(f"\nSummary saved to results/sweep_summary.csv")
PYEOF
fi

echo -e "\n═══════════════════════════════════════════════════════════════"
echo "  Sweep complete.  Results: ${RESULTS_CSV}"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════
# Phase 23 — Multi-seed HA benchmark (for CIs + significance)
# ═══════════════════════════════════════════════════════════════════
echo -e "\n══════════════════════════════════════════════"
echo "  Phase 23: Multi-seed HA benchmark (10 seeds)"
echo "══════════════════════════════════════════════"

if $DRY_RUN; then
    echo "  [dry-run] ${PYTHON} evaluation/run_ha_benchmark.py --n_seeds 10 --n_episodes 5 --seed 100 --output evaluation/ha_results_multiseed.csv"
else
    "${PYTHON}" evaluation/run_ha_benchmark.py \
        --n_seeds 10 \
        --n_episodes 5 \
        --seed 100 \
        --output evaluation/ha_results_multiseed.csv
fi

# ═══════════════════════════════════════════════════════════════════
# Phase 24 — Statistical analysis (CIs + significance tests)
# ═══════════════════════════════════════════════════════════════════
echo -e "\n══════════════════════════════════════════════"
echo "  Phase 24: Statistical analysis"
echo "══════════════════════════════════════════════"

if $DRY_RUN; then
    echo "  [dry-run] ${PYTHON} evaluation/statistical_analysis.py evaluation/ha_results_multiseed.csv --baseline ha_c2g --alpha 0.05 --confidence 0.95 --latex paper/tables/ha_benchmark_table.tex"
else
    "${PYTHON}" evaluation/statistical_analysis.py \
        evaluation/ha_results_multiseed.csv \
        --baseline ha_c2g \
        --alpha 0.05 \
        --confidence 0.95 \
        --latex paper/tables/ha_benchmark_table.tex
fi

# ═══════════════════════════════════════════════════════════════════
# Phase 25 — Failure-case analysis
# ═══════════════════════════════════════════════════════════════════
echo -e "\n══════════════════════════════════════════════"
echo "  Phase 25: Failure-case analysis"
echo "══════════════════════════════════════════════"

if $DRY_RUN; then
    echo "  [dry-run] ${PYTHON} evaluation/failure_analysis.py --agents ha_c2g cbm_only cbm_gate cbm_shield simplex_ppo cbf_ppo random --n_seeds 10 --seed_start 100 --output evaluation/failure_analysis.json"
else
    "${PYTHON}" evaluation/failure_analysis.py \
        --agents ha_c2g cbm_only cbm_gate cbm_shield simplex_ppo cbf_ppo random \
        --n_seeds 10 \
        --seed_start 100 \
        --output evaluation/failure_analysis.json
fi

echo -e "\n═══════════════════════════════════════════════════════════════"
echo "  Full sweep + analysis complete."
echo "  Statistical results: evaluation/ha_results_multiseed.csv"
echo "  LaTeX table:         paper/tables/ha_benchmark_table.tex"
echo "  Failure analysis:    evaluation/failure_analysis.json"
echo "═══════════════════════════════════════════════════════════════"
