# Fractional-window Track B follow-up

**Adaptive, Dev-informed follow-up—not fresh validation.** V1 was completed and
snapshotted at `5e2bf2855a3da4ac8cfea1763df8e9d8b28e9f82` /
`results/ppo-sac-two-track-dev-20261101`. The new parameterization responds to
the already-observed near-greedy behavior; the potential coefficient **10 USD/MWh**
comes from the already-registered margin10 heuristic. This implementation makes
no learned-improvement or promotion claim and does not open the fresh reservation.

Only this sibling directory contains new implementation/tests/documentation.
`FractionalDispatchTask` subclasses the frozen `track_b.economy_macro` task; it
does not copy the environment, edit frozen code, or monkeypatch global factories.
The task classes do not create registrations or execute training/evaluation
panels. The parent integration supplies the separate runners described below.

## Native factorial integration

The registered comparison is **fractional action versus fractional action plus
credit potential**, each with PPO and SAC and four matched seeds: 16 full fits.
Every fit receives 24,576 macro transitions with two environments, 256
policy-collected warmup transitions, and 256 critic-only updates. Only the
mandatory final checkpoints can enter the primary development comparison.
Actor bodies, seeds, initial Gaussian sigma 0.05, and all other settings match
within the factorial. Exploration in physical MW differs from the earlier
absolute-MW coordinate and is not claimed to match it.

Before the full campaign, four native 512-transition smoke fits must pass.
For each algorithm, the two reward variants must have bit-identical warmup
observations, next observations, actions, and terminal flags; their reward
differences must match the observed-state potential formula.

The 16 learners/simulator groups occupy 48 distinct physical cores with one
numerical thread per process. Evaluation uses the remaining 16-core group.
All old protocols, failed outcomes, and comparator identities are preserved.
An explicit adaptive-attempt ledger and one-further-campaign cap are bound
before any future opening of the shared calendar reservation.

```bash
cd outputs/rl_grid_cash_ppo_20261101/track_b_fraction
../../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -p no:logging
../../../.venv/bin/python study_fraction.py prepare
# Commit registered source and protocol before launching.
../../../.venv/bin/python study_fraction.py smoke
../../../.venv/bin/python study_fraction.py full
../../../.venv/bin/python evaluate_fraction.py run
../../../.venv/bin/python evaluate_fraction.py report
```

The combined task and native integration suite passed 38 tests before
registration. No new development or final outcome is produced by those tests.

## Parent-facing interface

```python
from track_b_fraction.fraction_macro import (
    FractionalDispatchTask,
    CreditPotentialEnv,
)

task = FractionalDispatchTask(
    protocol,
    seed=4200,
    rank=0,
    fixed_cell=None,
    record_trace=False,
    evaluation_mode=False,
    evaluation_cells=None,
)
env = CreditPotentialEnv(task, coefficient_usd_per_mwh=10.0)  # optional
observation, reset_info = env.reset()
observation, reward, terminated, truncated, info = env.step([0.5])
env.close()
```

The task constructor, training-only schedule, explicit fixed-cell evaluation
allowlist, `last_episode` / `episode_result`, `progress()`, and `terminal_traces()`
interfaces are inherited. The exact unchanged 23D observation schema and frozen
contracts are documented in [the V1 task README](../track_b/README.md).
Native action remains **one float32 coordinate in `[-1,1]`**. There are no new
learned observations, work/SOC actions, or calendar labels.

`CreditPotentialEnv` is **training-only** and rejects evaluation-mode tasks.
The coefficient argument is accepted for explicit parent configuration but must
equal the fixed **10.0**; no new coefficient sweep is authorized by this task.
An optional `gamma=1.0` keyword is checked. Both task and wrapper expose
`required_gamma == 1.0`; the trainer itself must also use **gamma 1**.
Compatibility aliases `FractionalWindowTask` and `PotentialEnv` name these same
implementations.

For PPO, the parent's existing `MacroLatentEnv` belongs **outside** this optional
reward wrapper:

```
parent PPO action-coordinate wrapper
    -> optional CreditPotentialEnv
        -> FractionalDispatchTask
            -> unchanged full-residual macro and physical controller
```

The SAC path can use the native 1D box directly. At deployment use the
**unshaped fractional task**, retaining whichever parent action-coordinate
wrapper is required by the saved policy. The parent must validate registration,
source/candidate locks, and any final-opening ledger **before** constructing an
evaluation-mode task with its explicitly registered `evaluation_cells` list.
Neither class performs or bypasses that authorization.

## A fraction of a window allowance, not a changing tick cap

For a validated native action `a`, set `fraction = max(a[0], 0)`.

**After the actual market handshake, at the first hardware tick only**, freeze:

```
C = actual committed MW after acceptance/rejection
credit_start = current realized capacity-error credit [MWh]
H = full remaining duration of THIS service window [hours]

allowance = min(6, max(0, 0.08*C + (credit_start - 0.02)/H))
fixed_request_MW = fraction * allowance
```

Healthy windows have `H = 0.25`. The implementation calls the frozen budget's
`cap(C, H)`, retaining its identical floating arithmetic (`1 - 0.92` for 0.08).
It cannot freeze against the previous window's capacity or assume acceptance:
the allowance is created inside the first post-handshake hardware callback.

For **every** tick of that window, the unchanged original guard still applies:

```
current_cap = original_guard(actual_C, current_credit, current_remaining_hours)
applied_extra_request = min(fixed_request_MW, current_cap)
```

The **fixed request never increases within the window**, even if current credit
improves and the current cap rises. The allowance is refreshed exactly once in
the next window using its new actual contract and current credit. A new episode
clears all window snapshots.

In particular, this code does **not** use
`fraction * continually_recomputed_cap` at every tick. That alternative can
accelerate spending near the end of a window and undo the intended fractional
budget choice. Fraction 1 also means the full **initial** allowance, not V1's
continuously greedy 6 MW request. Nonpositive actions remain the exact analytic
zero-overlay control.

The original metered absolute-error credit update is unchanged. Actuator
limitations can still prevent delivery; actual errors debit credit, not desired
errors. Guard/service buffers are unchanged, with no absolute 90% guarantee.
Qualification still audits actual performance-weighted service against the
true native reference capacity, plus all original physical/recovery gates.

Full-residual bids, zero ask, posted clearing payment, rejected-window 5 MW
fallback, true contractual demand, RegD, full throttle 1, nominal pump 0.7,
SOC target 0.5, baseline thermal/BESS compensation, physical settlement and fixed
two-hour recovery remain untouched. Credit banking is not a new SOC-inventory
action. Electricity savings remain physically metered; this does not relabel
substitution or constant efficiency offsets as free service arbitrage or load
shifting.

## Optional credit potential: unchanged complete objective

For the unchanged observation:

```
observed_credit_MWh = obs[21] * 30
Phi(obs) = 10 USD/MWh * observed_credit_MWh / 100 USD

shaped_reward = raw_reward + next_Phi - previous_Phi
next_Phi = 0 on EVERY true terminal, including invalid/fault terminals
```

The function reads only the already-observed credit feature, not private
simulator state or a forecast. Reset requires initial Phi exactly zero. Native
observations are not changed: a terminal observation may still show nonzero
credit, while its **terminal potential is zero**.

With gamma 1, the shaping terms telescope:

```
sum(shaped_rewards)
  = sum(raw_rewards) + terminal_Phi - initial_Phi
  = sum(raw_rewards)
```

Thus qualified complete episodes retain the original
`(actual_26h_cash - exogenous_reference_cash)/100` objective. Invalid scored
episodes retain the original **−65,400** return and incomplete actual 26-hour
cash remains `None`. The unchanged raw excess-cash bound is 3,270,000 USD.
The wrapper asserts telescoping and raw-return preservation to an absolute
`1e-8` reward-unit floating tolerance; it does not add an undisclosed terminal
correction.

An administrative close or propagated exception is **not** a returned true
terminal transition. Partial raw/shaped sums are retained with
`telescoping_complete=False`; no missing refund or penalty is invented. Ordinary
native timeouts remain forbidden because the underlying task is a true-terminal
finite 26-hour task.

## Terminal journal and raw traces

The parent's existing `MacroJournal` can continue recording
**`info["episode_result"]`**. No journal changes or private-state access are needed.

The fractional task retains all inherited result fields and adds:

* `parent_task_schema`, `action_semantics`;
* `fraction_window_count`, `fraction_windows`;
* `adaptive_dev_informed_followup=True`, `fresh_validation_claimed=False`.

Its `task_schema` is `track-b-fractional-window-economic-dispatch-v1`.
Each window snapshot contains:

```
dispatch_window_index
window_start_tick
window_committed_mw
window_credit_start_mwh
full_current_window_hours
initial_allowance_mw
window_fraction
fixed_window_request_mw
```

These fields also appear in captured service/macro rows and returned window info.
The inherited `desired_extra_response_mw` summary is corrected to the actual
fixed request computed after the handshake. Existing `guard_cap_mw`,
`guarded_extra_response_mw`, `guard_applied`, and before/after credit fields
continue to expose the current per-tick guard and realized account.

The potential wrapper copies the raw terminal result without mutating it and
adds:

* `raw_return`, `shaped_return`;
* `potential_shaping`, including coefficient/source, initial/final Phi, sum of
  potential deltas, gamma, and returned-terminal/telescoping status.

**`reward_sum`, cash fields, original ledgers, and `reward_components` retain
their raw meanings and values.** In particular, `reward_sum` is not replaced
with a shaped training score. Step info additionally contains the raw/shaped
reward, previous/next Phi and delta under `potential_shaping`.

With `record_trace=True`, call public `terminal_traces()` before reset/autoreset.
It returns independent row snapshots keyed by `hardware.csv.gz`, `macro.csv.gz`
and `recovery.csv.gz`; the potential wrapper additionally returns
`potential.csv.gz`. The original macro `training_reward` column remains the raw
reward, and the extra potential trace reports shaped rewards separately.
The parent can explicitly persist these lists with its authorized CSV/JSON
writer. Nothing writes experiment outputs automatically. With capture disabled,
grid metrics are still available after complete 24-hour service, as in V1.

## Focused tests

Only training day **16**, seed **97000001**, bias **0.5** is replayed. The
evaluation-mode test copies that already-training fixture into a test allowlist;
no actual development/final condition is run.

From this directory:

```bash
mkdir -p .cache
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 BLIS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  TMPDIR="$PWD/.cache" XDG_CACHE_HOME="$PWD/.cache" \
  MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ../../../.venv/bin/python -B -m pytest -q --capture=sys \
  -p no:cacheprovider -p no:logging test_fraction_macro.py

../../../.venv/bin/ruff check --no-cache fraction_macro.py test_fraction_macro.py
../../../.venv/bin/ruff format --check --no-cache \
  fraction_macro.py test_fraction_macro.py
```

Coverage includes distinct realized credit budgets for constant fractions,
post-handshake allowance freezing, window/episode reset, the unchanged guard
under a measured delivery error, input validation, no contract/work/reference
mutation, exact zero full-26-hour V1 parity, nonzero healthy raw/shaped cash
invariance, service/recovery fault terminal refunds, evaluation exclusions,
terminal journal diagnostics and public trace snapshots. Fault tests inject
physical state errors without altering thresholds.

Validated: **34 tests passed**; Ruff lint and formatting passed. The test run
also passed a filesystem audit rejecting mutations outside `track_b_fraction`.

Tests verify the inherited task against the V1 snapshot and check protected
Track A/B/root sources and registrations for unchanged hashes. They are
implementation checks, not fresh policy-selection evidence. Training and
factorial/source registrations belong to the parent after this interface is
reviewed.
