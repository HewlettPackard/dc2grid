# Table 2: corrected grid-relief controller study

This study targets the camera-ready manuscript's **battery-exclusive facility
relief**, not an unqualified claim about maximum system demand or real deployment.
It uses the previously frozen `physical_v1` simulator, unchanged plant capacity,
5-second hardware decisions, 15-minute market decisions and 24-hour service cases.
Complete evaluations append the same fixed two-hour physical recovery.

The study is complete. See the [verified results and manuscript deliverables](reports/REPORT.md),
the [final machine-readable report](reports/final_report.json), and the
[revised camera-ready source](publication/camera_ready.tex). The two final panels
are now consumed; do not reuse them as untouched tests after further tuning.

The original archive, earlier registered studies, attachments, and everything
under `/workspace/AAAI-demo` are unchanged. The other project supplied
methodological cautions, not observations pooled into this study.

## Controller changes

- **Cooling-first analytic baseline:** uses the original 18 observations and its
  own control history. It inverts the cooling/electrical response and reconstructs
  a nominal liquid-loop state from past measured temperatures. It does not inspect
  the simulator's reference state or future inputs. Most tested operation keeps the
  pump nominal, where the liquid-loop reconstruction cancels out: the main result
  is analytic HVAC inversion, not observer novelty.
- **BC-only:** a neural policy fitted to training-only, dithered demonstrations.
  Its results are supervised imitation, never labelled reinforcement learning.
- **Warm SAC:** starts from the corresponding BC actor, with fresh SAC optimizer
  state, critics and replay, then performs genuine online SAC updates.
- **Guided SAC:** the same initialization plus a fixed-demonstration stochastic
  BC term in the actor objective. This term also discourages action variance.
  Its weight balances initial deterministic actor-gradient norms on a fixed
  training batch; the balance is not assumed to remain constant.

The learned policies receive the original 18 observations, their four previous
commands, a causal 900-second request average and remaining time: **24 features,
without the analytic observer**. They directly choose pump, HVAC and BESS commands.
Full work service is fixed at one. There is no tracking projection, residual
inverter, safety replacement action, or analytic teacher query at deployment.
The unused residual-policy prototype is not a registered learned candidate.

The low-level environment's seven reward components and coefficients are unchanged.
Healthy 24-hour completion is a true finite-task terminal, not a bootstrap into a
new day. Training does not append recovery; evaluation does. Physical failures are
recorded and do not become healthy time limits. Ordinary SAC is not a safety
certificate.

## Data and budgets

The immutable [registration](registration.json) is the source of truth.

- Teacher corpus: 90 training-date/bias cases, **1,555,200 actual transitions**.
  All 90 completed and met the service gates. Three earlier demonstration smoke
  cases, 51,840 transitions, are separate and not in the corpus.
- BC: four main seeds, 12,000 supervised updates each. A six-date subset of
  training support is used for internal checkpoint selection, never the external
  development or final panels. Two additional single-seed observation diagnostics
  are not primary RL candidates.
- Online SAC: two variants times four paired seeds, **153,600 new transitions
  per fit** across four environments. This includes 1,024 policy-warmup transitions,
  followed by 152,576 ordinary SAC updates; each fit also has 256 critic-only
  calibration updates.
- Without early physical failures, a fit completes eight service episodes and
  leaves four partial trajectories. It touches 12 training dates and does not
  complete a 30-date online cycle. Actual exposure is recorded, not inferred from
  the intended support. The two variants share condition order by episode ordinal;
  different failure/reset rates can produce different realized exposure.
- One learner per GPU, four CPU environments per learner, one numerical thread
  per process, and 40 distinct physical CPU cores. Validation reuses environment
  cores while training is paused.

Development comprises six dates, two noise seeds and both biases: 24 cases.
The six reserved final dates are previously unexecuted calendar combinations,
not unseen workload phases; the workload repeats every 30 days. The three
previously consumed peak dates are regression evidence only. Calendar/noise/training
seeds are crossed, so reported dispersion is descriptive rather than an iid
confidence interval.

## Selection and claims

The six post-update milestones are 4,096 / 8,192 / 16,384 / 32,768 / 65,536 /
153,600 transitions. Each seed must qualify on every development case, improve
mean tracking over the frozen historical SAC, retain at least its biased facility
relief, and meet the complete PID cohort's mean cashflow. Eligible checkpoints are
ordered by biased mean facility relief, then lower RMSE.

These are **legacy-improvement gates**, not proof of an RL advantage. BC already
passes them before online training. Both SAC variants, every training-seed slot,
the paired BC initializers, the analytic controller, and the classical baselines
must be reported. The old selected SAC is a frozen historical reference, not an
isolated data-matched algorithm ablation. Unavailable or failing slots cannot
vanish into a successful-subset mean.

No learned-superiority claim is licensed by simply reproducing the analytic rule
or retaining a BC initializer. Incremental RL claims require the paired comparison
with the exact BC initializer. No method is promised to win.

## Metrics and physical accounting

- Primary relief: `reference facility MW - evaluated facility MW`, excluding BESS.
- LFR, SPDR, LWSR and LW-CF follow the manuscript definitions; LW-CF uses 30 MW.
  SPDR is a stress-average proxy, not a difference between actual maxima.
- Revenue sensitivity: `100 * (R_beta0.5 / R_beta0 - 1)` for the same controller.
- Supplemental metrics include both facility- and PCC-boundary modeled maxima,
  actual net-import electricity cost, 26-hour cashflow, battery throughput,
  temperature excursions, warnings and reference-relative thermal inventories.
- A complete but unclosed recovery can have an observed 26-hour cash number;
  `cashflow_closed_recovery_usd` is defined only when recovery closes. Qualification
  remains a separate requirement.
- The frozen model makes HVAC electrical draw proportional to its command rather
  than zone-B temperature. A hotter zone can therefore recover at the same fixed
  HVAC draw. Temperature closure is genuine modeled-state closure, but it does
  not impose a separate electrical charge proportional to thermal excursion depth.
  Neither the nominal baseline nor this model is an efficiency-optimal real plant.

Old comparator trajectories were not rerun merely to add metrics.
[The independent derivation](reports/comparator_metrics.json) rechecked 168 preserved
26-hour trajectories with 52,360 scalar-or-array checks, reproduced the existing
metrics and derived the new PCC/thermal fields without new policy execution.

## Reproduction and preservation

Use the project's existing `.venv/bin/python`; do not resynchronize its shared
environment with `uv run`.

- [Demonstrations](demonstrations.py) and [BC fitting](fit_bc.py) have independent
  manifests and preserved producing-source snapshots.
- [Training](train.py) and [SAC updates](sac_core.py) are pinned by the registration.
  Do not edit registered source during or after a campaign.
- [The launcher](launch_training.py) refuses an existing campaign and requires a
  source-identical, natively completed lifecycle smoke.
- [Final evaluation](final_study.py) requires completed training, locks candidates,
  records secondary-macro development without reselection, and registers final
  conditions before execution. Final output directories cannot be overwritten.
- Checkpoints are inference artifacts, not exact environment/RNG resume snapshots.
  There is intentionally no implicit resume path.
- Thirty-six focused tests and Ruff passed before production registration. These
  include exact zero-regularizer parity with SB3, full-day terminal/reset behaviour,
  actual macro-cadence equality, optimizer and checkpoint invariants, strict action
  identities, and nonmutating diagnostics.

The first successful small learner smoke is retained as a superseded source
revision. The production-matching smoke completed 1,152 transitions, 128 actor
updates and 256 critic-only updates, exited natively with success, and reproduced
the earlier smoke's policy parameters exactly.
