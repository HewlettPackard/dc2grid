# Track B: fixed-bid economic dispatch task

## Native learner integration

The parent integration adds `study_macro.py`, `train_macro.py`, and
`evaluate_macro.py`. PPO and SAC have identical zero-overlay deterministic
initial policies, shared seeded actor bodies, and initial Gaussian sigma 0.05.
They are not BC-initialized low-level actors. Their critic architectures,
entropy regularization, and later exploration differ and are recorded.

Each full fit uses 24,576 **macro transitions**, two parallel environments,
and 256 critic-only updates after 256 policy-collected warmup transitions.
Four seeds are used per algorithm. Only the final checkpoint is eligible.
When all training episodes finish normally this corresponds to 256 complete
24-hour-plus-recovery episodes per fit; actual episode and physical-step
counts must be read from the saved diagnostics rather than assumed.

The eight fits run on capped CPU learner/simulator groups, leaving all eight
GPUs to the concurrent low-level study. Full macro training waits for the
low-level baseline evaluator to release the overlapping CPU range. Every
numerical process uses one thread.

Scripted comparisons include zero overlay, a 1 MW request, greedy use of the
causal error budget, and a current-price-margin threshold of USD 10/MWh.
Learned policies must improve both facility/PCC relief and cash against the
strongest qualified scripted and frozen reference families, not only the
unmodified analytic controller. All evaluation remains development-only until
the shared final reservation is explicitly locked and opened.

```bash
cd outputs/rl_grid_cash_ppo_20261101/track_b
../../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -p no:logging
../../../.venv/bin/python study_macro.py prepare
# Commit the source and protocol before launching.
../../../.venv/bin/python study_macro.py smoke
../../../.venv/bin/python study_macro.py full
../../../.venv/bin/python evaluate_macro.py baselines
../../../.venv/bin/python evaluate_macro.py full
../../../.venv/bin/python evaluate_macro.py report
```

The environment and integration suite passed 54 focused tests before
registration, including the exact cash identity and raw-auditor integration.

This is a **new economic-dispatch task**, not the original tracking benchmark,
learned bidding, an optimal quantity policy, or a learned-policy result. No
registration, development/final panel, or model is created by the task class.
All implementation and tests are local to this directory. Frozen dependencies
are imported through the parent `economy_context` bootstrap; none are edited.

## Interface

```python
from track_b.economy_macro import EconomicDispatchTask

env = EconomicDispatchTask(
    protocol,
    seed=4200,
    rank=0,
    fixed_cell=None,
    record_trace=False,
    evaluation_mode=False,
    evaluation_cells=None,
)
observation, reset_info = env.reset()
observation, reward, terminated, truncated, info = env.step([0.0])
env.close()
```

Required protocol fields: `config_path`, `train_days`, `train_biases`, `n_envs`,
`normalization`, `service_gate`, `recovery`, `service_floor`. The gate, two-hour
recovery settings and normalization come from the existing frozen contract;
the constructor rejects relaxed gates/recovery or a changed published `0.90`
service floor. No new protocol is written. Inherited unrelated trainer settings
are not used: a subsequent trainer **must set gamma to 1**, as exposed by
`required_gamma = 1.0`. Rewards are dollars divided by **100**, with no shaping
or original simulator reward mixed into them.

In the default training mode, `fixed_cell` is optional and training-only.
Otherwise `condition()` uses exactly
the `StandaloneTask` permutation/seed convention: slot
`episode_ordinal*n_envs + rank`, day permutation seed `[seed,713,cycle]`,
environment seed `[seed,157,cycle,position]`, and cyclic training bias.
Known evaluation seeds and seeds in supplied development/final/sealed/test/
evaluation/validation condition lists are excluded, including for fixed cells.
The constructor seed controls this schedule; `reset(seed=...)` seeds Gym's RNG,
not the protocol schedule. Calendar-day labels and seeds are diagnostics only,
not learned features. Nonempty reset options are rejected.

### Explicit evaluation-only construction

After the parent launcher has authorized evaluation:

```python
env = EconomicDispatchTask(
    protocol,
    seed=4200,
    fixed_cell=registered_eval_cells[cell_index],
    evaluation_mode=True,
    evaluation_cells=registered_eval_cells,
    record_trace=True,
)
```

`evaluation_mode=True` **requires** a fixed cell and a nonempty, explicitly
supplied `evaluation_cells` list/tuple. Membership uses the complete
`(start_day, seed, regd_load_bias)` triple, with finite numeric fields and
nonnegative integer day/seed; extra descriptive metadata is not part of this
physical-condition key. Duplicates and unallowlisted cells are rejected.
The allowlist keys and fixed cell are snapshotted against caller mutation.
Supplying `evaluation_cells` without explicit evaluation mode is an error.
The mode is constructor-fixed and exposed as a read-only property.

This separate path permits an allowlisted evaluation day/seed/bias without
changing `protocol["train_days"]`, `train_biases`, or the original training
exclusions. It does **not** sample a training condition, even if the fixed cell
is later removed: that is an error rather than a fallback to random scheduling.
The same fixed condition is used on subsequent resets. Constructor seed/rank
do not select evaluation conditions; an evaluation rank need only be
nonnegative, not less than the training `n_envs`.

`condition_role` is `"evaluation_only"` in reset/step info, `progress()` and
`last_episode`; its default remains `"training_only"`. Training mode continues
to enforce the unchanged training schedule and evaluation-seed exclusions.
Both modes enforce identical physics, bidding, reward, service, recovery and
qualification contracts.

**Authorization separation:** the parent evaluation launcher must validate the
registration, source hashes, candidate/model lock, and any final-opening ledger
**before constructing** this mode, then supply that registration's approved
condition list. This class validates membership, not registration authenticity
or permission to open final data. A caller-supplied list alone is not evidence
of authorization. It does not read/open outcomes, mark a final-opening ledger,
create a registration, or bypass/mutate any old verification guards. No such
launcher is implemented here.

One action advances **180 five-second service ticks**. The 96th action also
executes all **1,440 recovery ticks**, unless the unchanged physical simulator
faults. The 26-hour task is a true terminal: `terminated=True, truncated=False`.
Early physical faults and malformed early native time limits also terminate,
but are explicitly distinguished in diagnostics. There is no timeout bootstrap.
Active episodes cannot be reset away. `close()` is idempotent and records an
incomplete administrative abort without inventing returned rewards or cash.

`last_episode` and the `episode_result` property retain the last result; terminal
step `info["episode_result"]` references it. They include actual service/recovery
ledgers, qualification, grid metrics, condition, tick/window counts, physical
fault flags, cash/reference/reward components, and credit statistics.
`progress()` reports the active prefix; incomplete 26-hour cash is always `None`.
Support/implementation exceptions propagate with an invalid diagnostic result;
they are not silently converted to a successful or averaged outcome.

`record_trace=True` retains `service_rows`, `macro_rows`, and `recovery_rows`
in memory. Service rows include true contractual target, observation-rounded
controller target, desired extra response, cap/applied guard, realized error,
credit before/after, actuator compensation, and metered physics. These are
diagnostics, not policy inputs. There is no implicit experiment-output writing.
Optional dependency caches are directed into this directory.

### Terminal diagnostics and public trace persistence

**Grid metrics are available with `record_trace=False`.** Their compact
accumulator is independent of raw trace capture. `last_episode["grid"]` is the
frozen 24-hour service grid-metrics dictionary whenever service completes,
including if recovery subsequently faults. It is `None` for incomplete
service. Recovery electricity remains separately metered; the grid metrics are
not relabeled as 26-hour metrics.

For explicit raw persistence, construct with `record_trace=True` and call the
public **`terminal_traces()` before the next reset/autoreset**. It returns an
independent deep-copied dictionary:

```
"hardware.csv.gz" -> service_rows
"macro.csv.gz"    -> macro_rows
"recovery.csv.gz" -> recovery_rows
```

It requires a terminated/closed episode and refuses capture-disabled or active
episodes. Faults/administrative aborts retain partial rows; empty phases return
empty lists. Capture-disabled episodes cannot reconstruct raw traces from the
grid accumulator. For example, the authorized parent launcher can persist
results without accessing any private environment state:

```python
from pathlib import Path
import pandas as pd
from economy_context import write_json

destination = Path(approved_episode_directory)
destination.mkdir(parents=True, exist_ok=False)
write_json(destination / "episode.json", env.episode_result)
for filename, rows in env.terminal_traces().items():
    if rows:
        pd.DataFrame(rows).to_csv(
            destination / filename, index=False, compression="gzip", mode="x"
        )
```

The caller chooses and authorizes its output directory; the class itself writes
nothing. Persist **invalid results too**, using their recorded phase counts and
qualification instead of inferring success from a trace file's existence. With
an autoresetting vector wrapper, intercept termination and call this public
method before that wrapper resets the task. `last_episode` survives reset, but
the raw row buffers start a new episode.

### Exact `last_episode` top-level keys

Before a first result, `last_episode` / `episode_result` is `None`. Every result
dictionary has these **46 common keys**:

```
task_schema
condition
cell
condition_role
decisions
service_fast_steps
recovery_fast_steps
complete
service_complete
service_physical_terminated
physical_terminated
raw_service_truncated
termination_reason
true_terminal
truncated
timeout_bootstrap
service_checks
physical_qualified
qualified
delivered_service_floor_required
delivered_service_floor
delivered_service_gate
delivered_service_ratio
committed_mw_hours
performance_weighted_mw_hours
maximum_reference_mw_hours
initial
service_terminal
service
recovery
grid
regulation_revenue_usd
ideal_regulation_reference_usd
contractual_demand_electricity_usd
observed_cashflow_usd
cashflow_26h_usd
cashflow_closed_recovery_usd
cash_vs_nominal_26h_usd
exogenous_reference_cash_usd
credit
bound_derivation
original_service_reward_sum
original_macro_reward_sum
original_recovery_reward_sum
inventory_accumulation_is_not_an_action
fixed_offsets_are_efficiency_dispatch_not_time_shifting
```

A terminal `step()` result, whether qualified, physically faulted or an invalid
early time limit, adds these **8 keys (54 total)**:

```
service_last_faults
reward_sum
excess_cash_usd
centered_cash_usd
reward_components
invalid_score_is_not_money
required_gamma
reward_scale_usd
```

A propagated implementation/support exception instead adds
`error`, `reward_sum`, `exception_propagated`, and
`invalid_score_is_not_money` (**50 total**); it is not a returned training
transition. An administrative close before completion instead adds `reward_sum`,
`administrative_abort`, and `invalid_penalty_returned` (**49 total**) and sets
the existing `true_terminal` key to `False`. No penalty is pretended to have been
returned by `close()`.

`reward_components` has exactly:
`actual_regulation_revenue_usd`, `ideal_regulation_reference_usd`,
`service_centered_electricity_usd`, `contractual_demand_electricity_usd`,
`recovery_centered_electricity_usd`, `excess_cash_usd`,
`invalid_adjustment_usd`. `service` and `recovery["ledger"]` use the unchanged
`Ledger.report()` schema; `grid` uses the unchanged `grid_metrics()` schema.
The recovery dictionary is phase-dependent (at minimum `performed`, `complete`,
`closed`), so consumers must check whether recovery was performed before reading
its ledger. A complete-but-unqualified episode has actual cash; an incomplete
26-hour episode has `cashflow_26h_usd=None`.

## Action and unchanged market/physics

Action: `Box(-1, 1, shape=(1,), dtype=float32)`.

```
desired extra PCC response [MW] = 6 * max(action[0], 0)
```

Negative and zero actions are the analytic no-overlay control. Actions are
validated, not silently clipped. Every window uses the unchanged
`FullResidualMacro` native bid and zero ask. Accepted bids receive the posted
clearing price; rejection retains **5 MW at the frozen fallback rate**, not a
zero-service window. An overlay can operate on that fallback obligation subject
to the same guard; rejection does not authorize an unmetered bypass.

`EconomicDispatchController` derives from `CoolingFirstController`. Its scoped
action-algebra copy permits an explicit extra request up to 6 MW while retaining
the original observation-only nominal observer and physical inverse helpers.
Full work/throttle **1**, nominal pump **0.7**, and SOC target **0.5** remain fixed
policy choices: there is no work, pump-residual, or inventory-target action.
The original temperature-dependent **pump boost**, HVAC thermal floor/rails, and
bounded BESS compensation are deliberately retained for exact zero-overlay
parity. Thus *nominal* pump 0.7 does not mean suppressing the baseline's thermal
compensation. Those commands and actual energy flows are logged and metered.

The overlay never changes contractual demand, RegD, offers, or settlement.
Failed delivery of a guarded target still incurs the **actual** tracking error.
Cooling/BESS substitution at identical PCC tracking is not free service
arbitrage. Constant offsets are efficiency/dispatch offsets, not load shifting.

## Causal service guard and audit

Credit starts at zero and carries across windows. After each actual service tick:

```
credit += ((1 - 0.92)*actual_committed_MW - abs(tracking_error_kW)/1000)*dt_hours
```

Before each hardware action:

```
cap = min(6, max(0, (1 - 0.92)*actual_committed_MW
                   + (credit - 0.02)/remaining_CURRENT_window_hours))
guarded_extra = min(desired_extra, cap)
```

The 0.92 internal target and 0.02 MWh reserve are operational buffers, **not**
changes to the published 0.90 qualification floor. Credit is never clipped or
reset to hide a deficit. No future load masks, future prices, forecasts, or
calendar labels are used by the policy/guard.

Final service ratio is audited independently:

```
sum(actual_committed_MW * actual_performance * actual_window_hours)
-----------------------------------------------------------------
sum(reference_capacity_MW * actual_window_hours)
```

Reference capacity is `maximum_native_quantity(true_posted_residual)` on
accepted windows and exactly **5 MW** on rejected windows. Native bid rounding,
baseline errors and actuator limits mean that the guard is **not an absolute
service guarantee**. Qualification uses the realized audit, unchanged physical
service gates, and actual recovery closure; invalid cases are retained.
Top-load grid metrics are computed only after service for diagnostics, never
fed back as action features.

## Exact observation schema: 23 float32 values

Indices 0–18 are the unchanged causal macro observation (at reset, its native
single-current-sample initialization). Existing normalization/clipping remains.

| Index | Feature |
|---:|---|
| 0 | Mean actual zone-A temperature / 35 |
| 1 | Mean actual zone-B temperature / 35 |
| 2 | End SOC |
| 3 | Mean base power / 250 MW |
| 4 | Mean facility power / 250 MW |
| 5 | Mean absolute RegD |
| 6 | Mean LMP / 200 |
| 7 | Mean normalized grid load |
| 8 | Mean absolute tracking error / committed capacity, with frozen floor |
| 9 | Any workload spike |
| 10 | Zone-A thermal headroom / 35 |
| 11 | Zone-B thermal headroom / 35 |
| 12 | Previous bid MW / 30 |
| 13 | Previous native bid-price action (the frozen `Macro.BESS_PREV` name) |
| 14 | Mean normalized frequency deviation |
| 15 | Mean PCC voltage, pu |
| 16 | Mean normalized backlog |
| 17 | Currently posted RMCP / 100 |
| 18 | Currently posted residual MW / 30 |
| 19 | Current fast signed RegD, `Fast.REGD == 6` |
| 20 | Current fast LMP / 200, already normalized at `Fast.LMP == 7` |
| 21 | Realized capacity-error credit / 30 MWh |
| 22 | Remaining 24-hour service fraction |

`Macro.REGD == 5`, `Macro.LMP == 6`; these are not the fast indices. The
constructor verifies the frozen index map. No independent-reference internal
temperature is a learned feature. Final observations contain the last recovery
window/current fast sample, zero posted-service fields, and zero remaining
service fraction.

## Reward, physical recovery and invalid-episode bound

Each service-window reward numerator is:

```
actual_regulation_revenue - rate*actual_committed_MW*hours
+ (nominal_electricity_cost - actual_PCC_electricity_cost)
- sum(actual_LMP * true_contractual_demand_MW * dt_hours)
```

All demand electricity is signed and metered at each five-second tick. Under
the fixed bids and exogenous market inputs, the subtracted reference is
action-independent. The final reward additionally includes the **actual**
two-hour centered recovery electricity. Recovery uses the frozen continuation,
fixed throttle 1 / pump 0.7 / HVAC 0.7, bounded SOC restoration to initial SOC,
zero committed regulation and zero new regulation revenue. It is not shortened
when inventories close early. There is no assumed extra HVAC cooling power
beyond the implemented ledger.

For qualified episodes:

```
sum(rewards) = (actual_26h_cash - exogenous_reference_cash)/100

reference_cash = ideal_service_revenue + contractual_demand_electricity
                 - nominal_26h_electricity
```

Actual revenue, nominal/actual electricity and original simulator rewards remain
separate diagnostic ledgers. Cash for complete but unqualified runs remains
actual cash; their invalid training penalty is explicitly **not money**.

Using the supplied frozen price support `L = 118.42008580000001 USD/MWh`:

* Original centered-cash prefix bound: **2,824,000 USD**.
* Its checked support: `24*500*30 + 26*L*(400+400) = 2,823,137.78464`.
* Subtracted ideal regulation bound: `24*500*30 = 360,000 USD`.
* Signed contractual-demand electricity bound: `24*30*L = 85,262.461776 USD`.
* Excess-cash prefix bound: round up
  `2,824,000 + 360,000 + 85,262.461776` to **3,270,000 USD**.

The frozen ledger verifies observed power/price/ambient support. Missing bound
fields are errors, never defaults. `score_transition` applies a terminal
clawback so every invalid episode's total return is **−65,400**, strictly below
the worst enclosed qualified return **−32,700**, irrespective of prefix earnings
or early fault time. Partial observed cash is retained, but incomplete actual
26-hour/reference cash is `None`.

## Focused validation

Run from this directory using the project's existing environment:

```bash
mkdir -p .cache
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 BLIS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  TMPDIR="$PWD/.cache" XDG_CACHE_HOME="$PWD/.cache" \
  MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ../../../.venv/bin/python -B -m pytest -q --capture=sys \
  -p no:cacheprovider -p no:logging test_economy_macro.py

../../../.venv/bin/ruff check --no-cache economy_macro.py test_economy_macro.py
../../../.venv/bin/ruff format --check --no-cache \
  economy_macro.py test_economy_macro.py
```

Validated: **49 tests passed**, Ruff lint passed, and formatting passed. The
final test run also passed an audit rejecting filesystem mutations outside
`track_b`.

Python stream capture avoids pytest's filesystem-backed capture; the unused file
logging plugin is disabled. Any dependency scratch/cache files are confined to
this directory.

All physical smoke inputs are the **training** cell day 16, seed 97000001,
bias 0.5. Tests compare full 26-hour zero-overlay physics, ledgers and recovery
against the existing analytic evaluator; compare a nonbinding +1 MW window
against the existing constant residual probe; replay the budget at every tick
of a guarded nonzero 26-hour run; and verify reward identities, unchanged
exogenous contracts, natural rejected windows, true terminals, fault penalties,
incomplete recovery, clock mismatch, schema, schedule, close and frozen hashes.
Fault/clock tests explicitly inject invalid state/clock conditions without
changing physical thresholds. These are focused contract tests, **not**
development/final results or evidence of learned improvement. Test-only
reference CSVs are explicitly saved below `.validation/` and cleaned afterward.

The evaluation-mode integration tests duplicate only that already-training
fixture into a test evaluation allowlist. They verify fixed-only scheduling,
caller-snapshot isolation, unchanged training exclusions, explicit allowlist
rejections, and full 26-hour parity with `record_trace=False`. They do not run
actual development/final conditions or validate/open a final-opening ledger.
Additional tests verify the exact terminal result keys and public trace export.
