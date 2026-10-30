# Three-level Ireland power-market experiment

This optional experiment combines C2G's DC physics with 3L's Irish market
simulation and outdoor-temperature data. Existing C2G and 3L source files are
not modified. Configuration, adapters, tests, documentation, and generated
results belong to this module.

## Model boundary

The plant wrapper composes C2G's workload, thermal, battery, and facility
electrical models. It preserves the four hardware actions: batch service
capacity, liquid-loop pump, air-side HVAC effort, and battery dispatch.
Batch service is a fraction of available hardware capacity, not a fraction
of the current queue.

The battery uses C2G's existing simple battery implementation explicitly.
Installing PySAM does not change the selected model. Capacity and thermal
overrides are supported; the master C2G configuration remains their source
of defaults. Ambient-temperature overrides are rejected.

All outdoor temperature and grid/market context come from 3L. The initial
thermal state uses the temperature at the scenario start. The current
quarter-hour temperature is held across the physical substeps. There is no
C2G weather loader, RegD signal, regulation handshake, frequency surrogate,
or regulation reward in this execution path.

C2G's electrical model still computes its original feeder-voltage diagnostic
internally. The wrapper excludes it from the Ireland state and objectives.
The active 3L backend is copperplate energy-market clearing, so frequency and
network voltage are reported as **not modeled**, not as zero violations.
The alternative SustainDC plant is not constructed.

## Three decision levels

| Level | Decision | Execution |
|---|---|---|
| L1 | DA operating schedule and hourly energy bids | On EOM openings |
| L2 | Operating-plan revisions and signed ID corrections | On IDM openings |
| L3 | Residual BM bids and physical actions | BM openings and every physical tick |

L1 calls 3L's existing `RuleBasedL1Policy.plan`. Its inputs are the supplied
DA price forecast, a workload forecast, projected delivery-start backlog,
and C2G service capacity. The workload forecast for each delivery hour uses
the latest completed observations for its four time-of-day quarter-hours.
If any of those observations are unavailable, that hour uses current
observed values. The source is recorded with the policy call. Since the first EOM
opening occurs at the start of the first warm-up day, the day-ahead plan
for the second day is made from persistence. The default 48-hour warm-up
allows history-informed plans for scored days. Completed workload inputs
are published before a simultaneous midnight market opening. Financial
settlement still follows market-clock advancement. The planning horizon is the common
allocation boundary; it does not create per-job deadlines. The native
`would_drop_mwh` output is recorded as unallocated planning work. It does not
discard work from C2G's queue.

L2 calls 3L's existing `RuleBasedL2Policy.bid_gate`, retaining its upstream
price thresholds, margins, and quantity rules. Its gate lead and product
count come from the actual ASSUME opening and offered products, rather
than the standalone policy's illustrative one-hour cadence. The current
Ireland setup offers one quarter-hour product with a two-hour-fifteen-minute
opening lead. The reference is the available DA clearing price, or an
explicitly identified supplied DA forecast before that clearing exists.
The buy-side inventory cap uses work projected to remain at delivery,
including forecast arrivals and intervening accepted service. Both sides
are bounded by the executable service that can be added or removed from
the current plan. These marginal bounds are expressed around the accepted
energy position for the native policy. They do not overwrite that position.
Correcting its difference from forecast consumption is the BM layer's role.

The operating plan is defined by accepted fills, not submitted bids. An
accepted DA order sets the hourly service fraction; a DA shortfall lowers
it in proportion to the unfilled incremental service energy. Work below
the electrical idle floor has no incremental energy to release by
deferral, so an energy shortfall does not reduce that service. An accepted
IDM order revises the affected quarter-hour from its quoted executable
baseline. The full-fill rate is found by inverting the same physical
forecast; a partial fill interpolates the service change between the
quoted endpoints. Forecasts are rebuilt from current state and accepted
plans at every opening. Any
part of a revision outside [0, 1] is recorded as clipped service and
remains visible as imbalance. Rejected bids do not change the plan.

L3 uses the existing C2G hardware controller selected by `controllers.inner`.
The default is C2G's `RuleBasedController`, not the experiment's former local
hardware heuristic. The plan caps the batch-throttle action: the applied
throttle is the minimum of the inner controller's output and the scheduled
service fraction, so thermal shedding can still lower it but no controller
can serve more than the schedule. Unscheduled intervals, such as the
warm-up period before the first DA delivery, are uncapped. Residual BM bids
compare executable forecast consumption with the full accepted DA+ID+BM
position. An empty queue cannot consume scheduled nameplate service.
Excess procurement therefore produces a sell correction, even if an earlier
ID purchase contributed to it. Only a cleared correction changes the
position. Rejected or partially filled corrections remain visible as imbalance.

The upper policies and C2G controller implementations are imported, not
copied. Native policy inputs, outputs, invocation counts, controller
identities, and any learned-policy assets are recorded.

All three levels share one live plant and one causal forecasting path.
A bid or clearing callback cannot advance that plant. Forecasts roll the
aggregate queue through the opening-to-delivery lead and the delivery
window at the physical timestep. Each step adds forecast arrivals and
serves the lesser of available work and planned capacity. The remaining
queue carries into the next step, including when work runs out partway
through a product. DA bids are generated chronologically under the native
candidate schedule; ID and BM forecasts use accepted plans.

Consumption is integrated using copies of the existing thermal and
electrical models. The explicit assumptions are nominal pump/HVAC effort,
an idle battery, persistent current 3L ambient temperature, and hardware
service at the accepted cap. This is feasible consumption under those
assumptions, not a forecast of the chosen hardware policy's future actions.
It can still differ from realized consumption because of workload, cooling,
battery dispatch, or hardware shedding. No future workload trace is read,
and live thermal, electrical, workload, battery, and random state remain
unchanged. Forecast delivery energy and work balances are retained with
the decisions.
The local clock adapter stops at the requested physical timestamp rather
than allowing the market engine to jump to its next future event.

## Accounting and episode boundaries

Accepted purchases are positive and sales are negative in the delivery
ledger. The ASSUME demand sign is converted once at the adapter boundary.
Hourly DA positions are integrated over the corresponding settlement
intervals; ID and BM trades are incremental positions.

The meter integrates gross facility power minus **actual** battery discharge.
It includes cooling, auxiliary consumption, and electrical losses. Remaining
energy imbalance settles at that delivery interval's live BM clearing price.
This is a single-price residual-settlement convention, not a claim to
implement every I-SEM settlement rule.

Identical trade callbacks are idempotent. Conflicting callbacks, missing
scored BM prices, malformed inputs, and duplicate settlement raise errors.
Zero fills do not prevent physical execution.

The default experiment uses two warm-up days followed by two scored days.
The first warm-up day has no day-ahead position and builds the observed
workload profile; the second is planned from persistence and provides the
first history-informed day-ahead opening.
Early warm-up intervals may precede the first BM product. Their electricity
bill is explicitly unpriced, not silently assigned a zero or forecast price.
Such missing prices are not permitted in scored intervals.

The plant is not reset at midnight. New C2G bids are restricted to deliveries
within the episode. Final backlog, SOC, open positions, and any incomplete
metering interval are reported. Terminal backlog and SOC-shortfall penalties
are separate from the electricity bill. A thermal trip produces an
incomplete episode, not a completed benchmark result.

The financial-completeness flag covers the scored delivery window. It is
separate from successful physical completion and the count of unpriced
warm-up intervals.

C2G is additional demand on top of the existing 3L background demand.
The experiment does not subtract a presumed data-centre share from the
historical load.

## Running

Use a dedicated market environment so the root project's environment remains
separate. For first-time setup, run the following from the repository root,
using its existing Python 3.11 environment:

```powershell
.\.venv\Scripts\Activate.ps1
$env:UV_SYSTEM_CERTS = "true"
uv venv --python .\.venv\Scripts\python.exe --seed .\.venv-3L
.\.venv-3L\Scripts\Activate.ps1
$env:UV_PROJECT_ENVIRONMENT = (Resolve-Path .\.venv-3L).Path
uv sync --locked --inexact --extra dev
uv pip install --python .\.venv-3L\Scripts\python.exe -r c2g_env\experiments\3l-power-market\requirements.txt
```

This installs the root project, its development tools, and the market
requirements without changing the root lockfile. On Windows, the system
certificate store permits trusted enterprise certificates without disabling
TLS verification. Do not rerun the environment-creation command over an
existing environment.

For subsequent runs:

```powershell
.\.venv-3L\Scripts\Activate.ps1
python -B -m c2g_env.experiments.3l-power-market.run --check
python -B -m c2g_env.experiments.3l-power-market.run
python -B -m c2g_env.experiments.3l-power-market.run --inner-controller rule_based --seeds 0 1 2
```

These requirements pin ASSUME and mango to 3L's locked versions. They do not
reproduce the complete 3L environment; additional dependencies may be needed
when loading the full backend.

The hyphenated directory name is intentional. The module entry point above
uses Python's module loader; ordinary source-level import syntax cannot use
that name. Programmatic callers can use `importlib.import_module`.

The entry point reads the module-local configuration. An alternative
configuration can be selected with `--config`. Path values in the
configuration use forward slashes and are resolved from the repository root
on every platform; backslash paths are rejected. `--seeds` evaluates each
requested seed in a fresh process and writes combined CSV and JSON results.
The seed drives ASSUME's clearing tie-breaks and the `random` controller;
the C2G workload trace always starts at its first tick, as in C2G itself,
so plant inputs are identical across seeds.
The default horizon is two warm-up days and two scored days per seed.
`--warmup-hours` and `--scoring-hours` permit explicit horizon overrides;
short checks are not substitutes for full DA-delivery evaluations.

Available inner controllers are `rule_based`, `pid`, `bang_bang`, `mpc_fast`,
`random`, `ppo`, `sac`, and `fixed`. The existing benchmark factory supplies
the other controller families. `fixed` reuses C2G's no-inner-controller
defaults: full batch throttle, pump and HVAC at 0.7, and idle BESS. It does
not construct a C2G macro environment; like every inner controller, its
throttle is capped by the operating plan.
PPO/SAC may select an existing training directory using `--model-dir`.
Missing checkpoints raise errors. PPO also requires its accompanying
normalization state; a normalizer present with SAC is restored as well.
Loading normalization wraps the Ireland environment's observation interface
and does not construct a legacy C2G environment. Learned inference runs on
CPU in this runner. The existing MPC predictor retains its approximate
fixed physical parameters, while its ambient input is updated from 3L.
Checkpoint manifests include the saved training seed and timestep count,
which need not agree with a directory label. Provenance also records the
compute-thread environment settings used by each process.

Existing C2G and 3L source
configuration and datasets are referenced rather than copied. The effective
auction calendar is read from 3L's selected scenario, not inferred from the
outer JSON alone.

The 3L checkout lives in the repository-local dependency directory selected
by `three_l.root`, not inside the Python virtual environment. Its configuration,
data, imports, and provenance resolve from that shared source root, independent
of the working directory. The backend adds this root to Python's search path
when loading 3L's existing `assume_env` package; no 3L source rewrite is needed.

Relocating or staging the checkout does not install ASSUME, mango, or the
other dependencies imported by 3L. The primary runtime versions must match
3L's lockfile. Missing dependencies produce an actionable error; this module
does not automatically install packages or change root dependency manifests.

The preflight checks input coverage and primary dependencies. Complete 3L
imports are checked when the backend starts. The adapter supports one local
ASSUME world; use a fresh process for each experiment.

Each run creates a new contained result directory, whose location is printed.
Outputs include source/input fingerprints, configuration and revisions,
per-tick transitions, encoded controller observations, requested and applied
actions, native upper-policy calls, accepted trades, plan revisions from
fills, clearing-event counts, interval settlements, and an episode summary.
Summaries include scored procurement and imbalance costs, energy by market,
metered and cooling energy, tracking MAE/RMSE, the mean gross-facility
request, the number of schedule-capped ticks, thermal peaks and warning
duration, SOC range, backlog, and controller inference time. Frequency and
voltage violations are not manufactured from compatibility inputs. The
existing 3L study-case runner and its scenario-copying routines are not
invoked.

The step reward adds the scored physical term (throughput minus thermal
warning penalty per tick) to the negative settled cost divided by
`cost_scale_eur`. At the default scale the settlement term of a quarter-hour
is roughly two orders of magnitude larger than the physical term summed over
the same interval, so the reward is effectively an electricity-bill signal.
No controller in this module is trained on it.

## Observation and controller interfaces

`ThreeLevelEnv` exposes a Gymnasium dictionary observation with separate
plant and market arrays. Field names are included in reset information.
The market array includes signed positions, metered energy, service target,
remaining interval time, supplied EOM/IDM forecasts, and observed 3L system
context. A separate validity indicator accompanies the BM-price field;
its numeric placeholder before clearing is not a settlement price.

`hardware_action()` invokes the selected C2G policy through an 18-field
observation adaptor. It reuses C2G's existing state encoder and benchmark
factory with a data-only context. The C2G environment, weather loader, and
grid-signal generator are not constructed.

Physical channels retain C2G scaling. The workload-burst flag comes from
the actual workload snapshot. Ambient temperature is supplied by 3L;
only the legacy temperature-normalization bounds are retained. Irish load
is normalized by the sum of 3L's demand-unit ratings, not a future realized
demand maximum. The price channel uses an available live clearing price;
an explicitly marked supplied forecast is permitted only during warm-up.
Values remain EUR/MWh despite the encoder's legacy USD-named input key.

The old regulation and commitment channels encode a signed power-reduction
request and its scale. The request is gross facility power minus the
accepted DA+ID+BM delivery target; it is a setpoint that stays non-zero while
the battery covers a gap, which keeps the response stable. The reported
tracking MAE/RMSE use net import minus the target, so the two quantities
differ whenever the battery is active; the mean absolute request is reported
separately. The target uses cleared quantities, not
submitted bids. Requests beyond the legacy normalization range are clipped
only for the policy input, with the omitted MW recorded; settlement and
tracking metrics still use the full target.

The frequency-deviation and voltage channels contain declared compatibility
placeholders of zero and one pu. They are not simulated grid measurements.
Existing PPO/SAC checkpoints are therefore transfer evaluations, not
unchanged in-distribution evaluations.

`step(action)` retains the four original actuator meanings. The batch
throttle applied to the plant is the minimum of the inner policy's output
and the scheduled service fraction for the current interval; the inner
output and the cap are both recorded per tick. The selected inner policy
remains responsible for cooling, battery dispatch, and thermal shedding;
deviations remain visible in tracking, settlement, and backlog. The
observation does not convey the complete future workload schedule to an
existing C2G policy.

## Native rule-controller evaluation

The two historical evaluations below were produced before the operating plan capped
the hardware throttle, before IDM fills revised the plan, and before the
observed same-hour workload profile replaced single-sample persistence.
They remain a record of that earlier configuration. The current-setup
rerun is reported separately after those evaluations.

We evaluated the native 3L L1/L2 rule policies with C2G's rule-based
hardware controller for seeds 0, 1, and 2. Each run used January 1, 2025
for warm-up and January 2 through January 3 for scoring on the scenario
clock. All runs completed 51,840 hardware steps and 192 scored settlement
intervals, with no partial settlement or outstanding position.

| Seed | Scored electricity bill (EUR) | Residual imbalance portion (EUR) |
| --- | ---: | ---: |
| 0 | 736,551.47 | 13,140.00 |
| 1 | 736,326.08 | 12,990.63 |
| 2 | 738,402.66 | 13,286.54 |

Bills exclude warm-up costs and the additional terminal SOC penalty of
EUR 2,273.79 per run. SOC declined from 78.14% at the scored start to
34.84% at the end, so this evaluation is not charge-sustaining.

The physical and action traces were identical across seeds. Each run
imported 4,209.01 MWh, served 587.052 work-MWh, and ended with no backlog;
the peak scored backlog was 1.638 work-MWh. Peak zone temperatures were
30.92 and 21.86 degrees C, with no thermal-warning time. Tracking MAE and
RMSE were 1.796 and 5.144 MW. The largest instantaneous tracking error was
78.00 MW. The legacy controller-input range clipped requests for 11.67
minutes, while tracking and settlement retained the full target.

Across each complete run, L1 executed twice and placed 48 accepted hourly
DA orders. L2 executed 279 times but submitted no intraday orders: all 27
cheap-price gates had empty queues, and none of the 240 expensive-price
gates had releasable capacity above the native policy's minimum. All 286
submitted balancing orders were accepted. These counts include warm-up.
This exercises native policy invocation, but not useful intraday trading.

The results use one fixed weather, demand, and workload window, not three
independent scenario samples. This initial rule-only batch did not include
a matched controller comparison. The comparison below uses fresh runs.
Two additional same-seed smoke runs
had identical input fingerprints and physical/quantity summaries but
differed by EUR 2.08 in BM-related costs. Native clearing uses random
tie-breaking; the cause of that same-seed variation was not isolated, and
bit-for-bit financial replay is not established.

All 155,520 recorded actions were replayed against the original C2G rule
controller without a mismatch. Independent reconciliation of physical
integrals, accepted trades, residual settlement, tracking, and combined
outputs passed, and the recorded source/input fingerprints matched the
files used for validation.

## Six-controller hardware comparison

We paired the same native 3L L1/L2 policies with PID, SAC, rule-based,
fixed-action, random, and bang-bang hardware control. All six modes were
rerun for seeds 0, 1, and 2, using the same 24-hour warm-up and 48-hour
scoring window described above. The plant retained its 250 MW IT nameplate
and 5 MW / 15 MWh battery. At most two evaluations ran concurrently, with
numeric-library thread counts set to one and recorded in provenance.

All 18 runs completed. Each served 587.052 work-MWh during scoring and
ended with no backlog. There were no thermal trips or scored thermal
warnings. The highest scored temperatures across the matrix were
31.55 degrees C in zone A and 22.12 degrees C in zone B.

The table reports means of per-run metrics. Parentheses give the sample
standard deviation of the electricity bill. Bills exclude warm-up and
terminal objective penalties. Tracking compares net import with the full
accepted DA+ID+BM position at five-second resolution, not with a separate
regulation-only commitment. The tail column averages the per-run 95th
percentiles of absolute tracking error.

| Hardware controller | Bill, kEUR (SD) | MAE, MW | RMSE, MW | Absolute-error p95, MW |
| --- | ---: | ---: | ---: | ---: |
| PID | 735.148 (0.642) | 1.637 | 5.092 | 8.801 |
| SAC | 755.467 (0.561) | 5.704 | 8.538 | 14.265 |
| Rule-based | 736.897 (0.252) | 1.796 | 5.144 | 8.850 |
| Fixed-action | 741.282 (0.761) | 3.830 | 8.238 | 14.822 |
| Random | 631.519 (0.689) | 15.493 | 19.303 | 34.799 |
| Bang-bang | 734.197 (0.453) | 2.595 | 5.195 | 10.361 |

PID had the lowest mean tracking error in this window. Bang-bang had a
slightly lower bill, with higher MAE and tail error. Random control used
less cooling energy, averaging 1,247.639 MWh versus 1,724.422 MWh for
rule-based control, but had substantially poorer tracking. Its mean
absolute quarter-hour imbalance was 11.528% of metered energy, compared
with 1.825% for PID and 2.006% for rule-based control. Lower electricity
cost alone therefore does not identify a better load-following policy.
Large transient errors remain: PID's maximum instantaneous error was
73.02 MW, despite its lower MAE.

Warm-up produced different scored-start battery states. The following
values are means across seeds. The terminal penalty uses the configured
simulation-start SOC of 50%, not the scored-start SOC; it is an objective
term rather than an electricity charge.

| Hardware controller | Scored-start SOC, % | Terminal SOC, % | Terminal penalty, kEUR |
| --- | ---: | ---: | ---: |
| PID | 83.36 | 32.50 | 2.624 |
| SAC | 10.61 | 10.97 | 5.854 |
| Rule-based | 78.14 | 34.84 | 2.274 |
| Fixed-action | 50.00 | 50.00 | 0.000 |
| Random | 39.66 | 19.79 | 4.532 |
| Bang-bang | 87.26 | 63.82 | 0.000 |

Scored-start SOC is not matched across controllers, so these bills should
not be read as a charge-balanced comparison. SAC used one
existing hardware checkpoint with saved seed 200 and 215,000 timesteps,
not three independently trained policies. No 3L-specific retraining was
performed. This transferred policy had higher bills and mean tracking
errors than PID, rule-based, and bang-bang control in the tested window.

Every run invoked L1 twice and L2 279 times. Random control had 2, 5, and
4 accepted intraday orders across the three seeds; 10 of those 11 orders
delivered during scoring, totaling approximately 0.086 MWh. The other
controllers submitted no intraday orders. All submitted C2G orders were
accepted. Intraday participation was therefore limited even though the
native L2 policy was active.

The audit also identified a pre-existing low-load boundary in C2G:
electrical IT power retains rack idle-power floors, while thermal heat
input uses workload IT power. This integration preserves that behavior.
Random control entered this regime for a mean 4.366 scored hours, with an
electrical-versus-thermal IT energy gap of 18.457 MWh. Its temperature
results should be interpreted with this modeling limitation.

Independent reconciliation replayed all 933,120 hardware actions without
a mismatch, checked every settlement against physical metering and
accepted trades, and verified the native clearing exports. Native CSV
exports use five significant digits; financial reconciliation used the
full-precision JSON records instead. The largest numerical discrepancy
was below `2.35e-10`, with relative tolerance disabled. All 88 source and
20 input fingerprints matched. The comparison remains limited to one
Irish input window, unmodeled frequency and voltage, and the unresolved
native financial repeatability limitation noted above.

## Schedule-capped rerun before executable forecasting

We reran all six hardware controllers for seeds 0, 1, and 2 with the then-current
accepted-fill schedule cap and observed-profile forecasting. Each run used
January 1 and 2, 2025 for warm-up and was configured to score January 3 and
4. The plant and SAC checkpoint were unchanged. This is not a matched
repeat of the earlier uncapped comparison, which also scored different
dates.

The table gives means and sample standard deviations across seeds.
Thermal warning is the fraction of scored hardware steps exceeding either
zone's configured warning temperature. BESS aging is the change in the
simple battery model's aging fraction during the scored window, expressed
in units of 0.0001. It is not a measured physical capacity loss.

| Hardware | Completed | Scored hours per run | RMSE, kW (SD) | Thermal-warning fraction | BESS aging (SD) | Native 3L reward per step (SD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PID | 3/3 | 48 | 15,792.56 (0.00) | 0.000000 | 18.9895 (0.0000) | -0.023234 (0.000015) |
| SAC, partial | 0/3 | 19.7125 | 1,466.93 (0.49) | 0.057493 | 10.3492 (0.0276) | -0.024609 (0.000008) |
| Rule-based | 3/3 | 48 | 15,724.68 (0.00) | 0.000000 | 17.7466 (0.0000) | -0.023260 (0.000013) |
| Fixed-action | 3/3 | 48 | 21,043.67 (0.00) | 0.000000 | 0.0000 (0.0000) | -0.022583 (0.000017) |
| Random | 3/3 | 48 | 31,138.53 (128.38) | 0.000000 | 78.3378 (0.8131) | -0.019739 (0.000023) |
| Bang-bang | 3/3 | 48 | 17,026.04 (0.00) | 0.000000 | 90.0311 (0.0000) | -0.023093 (0.000008) |

All three SAC runs terminated at January 3, 19:42:45 when zone A reached
35.0009 degrees C, above the 35-degree safety limit. They each retained
20.2144 work-MWh of backlog. Their lower RMSE and cumulative battery aging
cover only the partial window and should not be ranked against completed
48-hour runs. The other 15 runs completed without scored thermal warnings.
Rule-based control had the lowest RMSE among the completed controllers in
this window.

Regulation revenue, the C2G macro reward, and the original C2G low-level
reward are unavailable in this execution path. These fields are recorded
as unavailable, not zero. The separately labeled native 3L reward is the
mean of the recorded scored step rewards, including throughput, thermal,
electricity-settlement, and terminal-penalty terms. It is not the paper's
regulation-tracking reward. Similarly, RMSE uses the full accepted energy
position rather than a regulation-only commitment.

The fixed-action controller still proposes the canonical fixed vector,
but its applied batch throttle is schedule-capped like the other
controllers. The recorded L1 calls used observed profiles for 47 of the
48 planned scored delivery hours and current persistence for one hour.
The hourly exception reflects the earlier midnight ordering, which published
the last completed quarter-hour only after the market opening. The executable
forecasting change described above corrects that ordering.

Preflight passed 62 tests, with the opt-in live test skipped; all 18
evaluations used live ASSUME clearing. Independent verification replayed
all 1,183,059 recorded hardware actions, including the schedule cap, and
all battery state updates. Scored battery aging excludes warm-up and
includes the first scored step. Metering, native fills, settlement,
reward reconstruction, and all 88 source and 20 input fingerprints were
checked. The largest numerical reconciliation discrepancy was below
`6.4e-10`, with relative tolerance disabled. No simulation source was
changed for this rerun.

## Queue-aware matched comparison

The executable-forecasting implementation was evaluated against the preceding
schedule-capped run, not the earlier uncapped comparison. Both evaluations used
the same six controller configurations, seeds 0/1/2, input data, initial-state
protocol, and SAC checkpoint. Each episode began on January 1, 2025, with
48 warm-up hours and a planned 48-hour scoring window beginning January 3.
No thermal guard, SAC retraining, checkpoint substitution, or change to C2G
core or vendored 3L sources was introduced.

The table reports means over three seeds, with sample standard deviation shown
for post-change RMSE. Warning percentages count scored hardware steps above
either configured warning threshold. BESS aging is the modeled age-fraction
increment in units of 0.0001, excluding warm-up.

| Controller | Full-horizon runs | Before RMSE (kW) | After RMSE (kW) | After warnings (%) | After BESS aging | After native 3L reward/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PID | 3/3 | 15,793 | 10,664 +/- 1 | 0.00 | 27.7500 | -0.024215 |
| Rule-based | 3/3 | 15,725 | 10,814 +/- 100 | 0.00 | 31.5016 | -0.024383 |
| Fixed | 3/3 | 21,044 | 6,871 +/- 167 | 2.20 | 0.0000 | -0.024608 |
| Random | 3/3 | 31,139 | 26,138 +/- 205 | 0.00 | 78.3378 | -0.026298 |
| Bang-bang | 3/3 | 17,026 | 8,257 +/- 37 | 0.00 | 41.0782 | -0.024307 |
| SAC, partial | 0/3 | 1,467 | 5,645 +/- 0.34 | 4.18 | 0.0762 | -0.048669 |

All five full-horizon controller groups had lower RMSE, with reductions from
16.1% for random control to 67.3% for fixed control. Fixed control had the lowest
full-window RMSE, but introduced thermal warnings on 2.20% of scored steps.
Its mean peak zone-A temperature was 33.233 degrees C, below the unchanged
35-degree termination limit.

SAC thermally terminated in all three seeds at January 3, 19:29:30, after
19.4917 scored hours, compared with 19.7125 hours previously. Its peak zone-A
temperature was about 35.0015 degrees C. Restricting the previous traces to the
same 19.4917-hour prefix gives mean RMSE of 1,468 kW before and 5,645 kW after.
Thus, the changed schedule did not improve this transferred checkpoint.
Its partial-window cost, reward, warning rate, and cumulative battery aging
should not be ranked against completed 48-hour episodes.

Regulation revenue, the original C2G macro reward, and the original C2G
low-level reward remain unavailable, not zero. The native 3L reward above is
a separate metric that includes settlement and terminal penalties.

### Inventory, exposure, and remaining error

The warm-up protocol was matched, but the scheduler changes altered the state
at the start of scoring. These are endogenous outcomes, not additional
initial-state overrides. Mean backlog and served work are reported in work-MWh;
SOC is a percentage.

| Controller | Scored-start backlog before / after | Scored work served before / after | Ending backlog after | Scored-start SOC before / after |
| --- | ---: | ---: | ---: | ---: |
| PID | 66.03 / 268.86 | 641.10 / 843.93 | 0.00 | 93.00 / 12.00 |
| Rule-based | 66.03 / 278.46 | 641.10 / 853.53 | 0.00 | 90.45 / 10.75 |
| Fixed | 66.03 / 278.46 | 641.10 / 853.53 | 0.00 | 50.00 / 50.00 |
| Random | 78.27 / 278.24 | 653.34 / 667.21 | 186.09 | 25.30 / 25.30 |
| Bang-bang | 66.03 / 269.33 | 641.10 / 844.39 | 0.00 | 93.01 / 12.00 |
| SAC, partial | 66.03 / 266.46 | 282.39 / 126.18 | 373.28 | 14.29 / 10.00 |

The SAC work totals retain their respective failed-episode exposures.
Completing the time horizon does not imply clearing all work: random control
retained a mean 186.09 work-MWh at the end. The other completed controller
groups cleared their queues. All complete windows received the same
575.07 work-MWh of arrivals; additional service came from warm-up inventory.
Bills increased rather than decreased in the five complete controller groups.
For example, the mean rule-based bill rose from EUR 810,993 to EUR 846,766,
while scored service rose from 641.10 to 853.53 work-MWh. These are not
equal-work electricity-cost comparisons.

The previously diagnosed empty-queue error was substantially reduced. In
rule-based seed 0, empty-queue periods decreased from 6.943 to 1.236 hours.
Their RMSE fell from 41.325 to 4.423 MW, and their share of total squared
tracking error fell from 99.9009% to 0.4217%. The queue classification uses
post-step backlog at or below `1e-10` work-MWh in both evaluations.

This does not establish tight tracking overall. The remaining rule-based
error is largely in backlog-positive periods. At January 3, 22:00, for example,
the accepted target was 164.840 MW, while consecutive net-import measurements
were 194.062 and 104.711 MW. The unchanged hardware policy alternated between
full and reduced batch service while the queue remained nonempty and zone A
was near 31.85 degrees C. The forecast assumes service at the cap and nominal
cooling, not those feedback actions. Imperfect workload profiles, hardware
under-service, feedback oscillation, and changed warm-up SOC remain relevant
limitations. No additional hardware tuning or thermal protection was applied.

All 18 episode records passed independent verification. This replayed
1,182,582 hardware actions and complete C2G physical/BESS steps, reproduced
6,489 native upper-policy calls, and checked 63,374 forecast-window records.
Queue conservation, causal profile inputs, delivery inventory, marginal ID
bounds, BM corrections, fill-derived service caps, metering, residual
settlement, rewards, and partial-episode accounting were reconciled.
All 89 source and 20 input fingerprints remained unchanged during execution.
Full-precision reconciliation discrepancies were below `5.5e-10`, with relative
tolerance disabled; native CSV values were checked at their five-significant-
digit export precision. Bit-for-bit financial repeatability is not established.

## Validation

The executable-forecasting change passed 83 tests, including the live
ASSUME day-rollover test. The tests cover delivery-time backlog, depletion
within a product, successive accepted ID revisions, partial/rejected
corrections, idle-floor behavior, causal input snapshots, unchanged live and
random state, and observation publication before midnight callbacks.
Constant-input forecast energy agrees with the existing physical rollout
within `1e-10` absolute tolerance, with relative tolerance disabled.

Ruff lint and VS Code's file diagnostics reported no errors. The configured
mypy invocation could not check this module because its existing hyphenated
package name is not accepted by mypy. Pylance's separate diagnostic requests
timed out; they are not reported as passing type checks.
The live test also emitted upstream NumPy scalar-conversion deprecations
from the unchanged 3L balancing implementation.

Using Ruff 0.16.0 and the same zero-threshold C901 measurement, the module
and test scope grew from 201 to 224 functions. The maximum remained 6.
The count above 5 changed from three to four, with the additional case
being the midnight regression's setup, callback, and cleanup. No function
exceeded 10. Newly added production forecast functions are at most 5.

Selected executable-forecasting call paths have the following measurements:

| Function | Before CC | After CC | Change or retained behavior |
| --- | ---: | ---: | --- |
| `DayAheadController.decide` | 3 | 2 | Shared forecast and delivery-start inventory |
| `IntradayController.decide` | 4 | 3 | Executable marginal service bounds |
| `ThreeControllers._balancing_bids` | 2 | 2 | Executable demand minus accepted position |
| `ThreeControllers.on_fill` | 4 | 4 | Fill-derived service conversion |
| `ThreeControllers.bid` | 5 | 5 | Shared forecast dispatch |
| `ThreeLevelEnv.step` | 5 | 5 | Publish completed inputs before clock advancement |
| `DeliveryForecast.advance` | New | 5 | Queue-conserving lead and delivery rollout |
| `DeliveryForecast._advance_slot` | New | 2 | Existing physical equations at each substep |
| `DeliveryForecast.rate_for_power` | New | 1 | Invert executable delivery power |

The earlier fixed-action measurements below describe that earlier change,
not the current module.

The fixed-action extension passed 29 focused tests, with the opt-in live
test skipped in that invocation. The 18 comparison episodes subsequently
used live ASSUME clearing. Ruff lint and editor diagnostics reported no
errors.

Ruff 0.16.0 C901 measurements covered the module and its tests, including
nested functions, before and after the extension. The measured function
count changed from 182 to 185. The maximum remained 6; three existing
functions remained above 5, and none exceeded 10. The zero-threshold
measurement command is given below.

| Function | Before CC | After CC | Change |
| --- | ---: | ---: | --- |
| `FixedActionController.predict` | New | 1 | Canonical fixed-action vector |
| `C2GHardwareAdapter._load_controller` | 4 | 5 | Module-local fixed selector |
| `C2GHardwareAdapter._load_learned_controller` | 4 | 4 | Saved checkpoint metadata |
| `provenance` | 2 | 2 | Source and thread-setting records |
| `test_fixed_action_uses_existing_macro_defaults` | New | 2 | Fixed-action behavior and environment-construction guard |

The fixed mode reuses existing C2G constants without constructing a
legacy environment. Other controller-selection paths retain their
existing implementations. No complexity-driven refactoring or threshold
relaxation was introduced.

Run the module's focused tests:

```powershell
.\.venv-3L\Scripts\Activate.ps1
python -B -m pytest c2g_env\experiments\3l-power-market\tests --import-mode=importlib -p no:cacheprovider
```

Temporary test files stay inside the module. The tests cover physical parity,
weather ownership, side-effect-free forecasting, scheduling conservation,
partial fills, signed settlement, negative prices, missing prices,
clock boundaries, controller separation, early-stop accounting, cleanup
after export failure, and 180 recorded physical steps per completed
settlement interval. The physical-parity assertions use an absolute
tolerance of `1e-10` with relative tolerance disabled.

The dedicated market environment uses Ruff 0.16.0. These commands reproduce
the lint gate and the per-function McCabe diagnostic report, respectively,
in the activated environment:

```powershell
python -B -m ruff check --no-cache c2g_env\experiments\3l-power-market --select E,F,I,C901 --config "lint.mccabe.max-complexity = 10"
python -B -m ruff check --no-cache c2g_env\experiments\3l-power-market --select C901 --config "lint.mccabe.max-complexity = 0"
```

The second command intentionally reports every function above zero and
returns a diagnostic exit status. It is a measurement command, not the
lint gate.

The delivery-loop tests use an explicitly labeled test double. That test
double is not selectable by the experiment runner and is not evidence of
live market clearing.

An opt-in live test crosses the DA rollover and checks native upper-policy
invocation, C2G inner-policy execution, and complete scored settlement.
In a provisioned 3L environment, set `C2G_3L_LIVE_TEST=1` before running it.
Unit-test checkpoints are explicitly untrained loading fixtures, not
evidence of learned-policy performance.

The supplied forecast files are treated as available forecasts; they do not
provide issuance-vintage metadata. There is no future realized-weather
forecasting, PPA overlay, new deadline queue, network power flow,
frequency-regulation stacking, training harness, or run resumption in this
initial implementation.
