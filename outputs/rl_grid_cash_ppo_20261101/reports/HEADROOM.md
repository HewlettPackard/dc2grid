# Verified pre-learning grid-relief and economic headroom

## What this establishes

There is measured room to improve both facility and PCC grid relief and
26-hour cash under the explicitly approved **economic-dispatch** variant.
This is not yet a learned-controller result, an untouched-test result, or a
replacement for the original regulation-tracking Table 2.

Six predetermined nonlearned probes completed all 144 episodes on the already
consumed 24-cell Dev-B panel. Every episode passed the unchanged physical,
work, thermal, SOC, two-hour recovery, and actual 90% capacity-service gates.
The raw audit passed 36,720 scalar-or-array checks, including matched
exogenous/reference trajectories and unchanged full-residual bid schedules.

The runner took 124.3 seconds using eight CPU workers, one numerical thread
each. No model training or GPU computation was needed for this headroom test.

## Main results

All rows use the **existing full-residual, zero-ask macro**. Means below use
the biased development cases, not the original peak dates or Final-B.
Cash differences compare against the strongest fully qualified frozen
reference family on these same cases; the reference envelope is defined
separately for each metric.

| Nonlearned probe | Facility relief, MW | PCC relief, MW | Tracking RMSE, kW | Cash gain, USD/26h | Minimum service ratio, both biases |
|---|---:|---:|---:|---:|---:|
| Existing analytic control | 6.783 | 6.783 | 0.184 | 0.00 | 99.9999% |
| Additional 0.5 MW dispatch | 7.283 | 7.283 | 500.000 | +134.92 | 96.51% |
| Additional 1 MW dispatch | 7.783 | 7.783 | 1000.000 | +269.83 | 93.02% |
| SOC-inventory diagnostic only | 6.783 | 6.783 | 0.184 | +46.94 | 99.9999% |
| Additional 1 MW plus SOC inventory | 7.783 | 7.783 | 1000.000 | +316.77 | 93.02% |
| Additional 1 MW only with positive current RegD | 7.751 | 7.751 | 742.907 | +184.59 | 95.84% |

The additional-1-MW policy exceeds the strongest frozen development relief
families by **0.491 MW at the facility and 0.581 MW at the PCC**, and the
strongest cash family by **USD 269.83 per biased case**. It also improves mean
cash on unbiased cases. The combined inventory/dispatch probe adds about
USD 46.94 without a material extra stress-relief contribution.

All three predetermined joint-headroom conditions pass for the 1 MW,
1 MW plus inventory, and positive-RegD probes. The 0.5 MW probe improves
cash and PCC relief but misses the strongest biased facility-relief envelope
by 0.0088 MW, so it does not pass the joint screen.

## Interpretation and limits

- The 1 MW probes deliberately accept more tracking error. Their roughly
  1 MW RMSE must not be presented as an accuracy improvement over the analytic
  or BC controllers. This is the separate economic-dispatch track the user
  approved, not a free improvement to regulation service.
- Constant extra curtailment is an efficiency/dispatch offset, not temporal
  load shifting or learned anticipation of peaks. The conditional-RegD probe
  uses only the currently observed signal, never a future top-decile mask.
- The inventory diagnostic charges toward SOC 0.6 using additional facility
  curtailment while preserving PCC tracking, then restores SOC physically
  during recovery. Its cash gain arises in recovery; it does not establish
  additional stress-time grid relief. Both power boundaries remain explicit.
- All work remains served without introducing a workload-deferral action.
  The fixed-effort nominal HVAC reference and the simplified thermal plant
  remain limitations of the frozen benchmark.
- Cash is modeled net cash after the full recovery, not operating profit:
  even the combined probe's biased mean is about **minus USD 96,854**.
  Primary cash excludes monetized battery wear, consistent with the existing
  benchmark; inventory cycling must also be examined under the registered
  wear sensitivities before any economic recommendation.
- These are development results on six known calendar dates and correlated
  stochastic conditions. No confidence interval, significance, fresh
  workload-phase, or new p90-tail claim is made.

The combined probe adds 3.0007 MWh of absolute AC throughput over the plain
1 MW probe for USD 46.94 additional biased-case cash. That increment breaks
even at **USD 15.64/MWh of throughput wear**: it is positive by USD 16.93 at
USD 10/MWh, but negative by USD 28.08 at USD 25/MWh. The plain 1 MW probe
uses zero battery throughput on these cases, so it is the cleaner headroom
demonstration. The next economic planner will not intentionally accumulate
terminal SOC.

## Reproducibility

- Baseline snapshot: `12c362245f51a4f143e76d2d458231bef4ede398`.
- Registered source: `5901eda8e2f0ad4b36fd9f2b958c0e5e5429b079`.
- Headroom registration:
  `f40f33958d29a83906e0fb71e0b926da23129fce7967b278d69e44e5503b95f4`.
- [Immutable protocol](../headroom_registration.json).
- [Raw-audited aggregates and exact decisions](../headroom_report.json).
- [Run manifest](../headroom_development/manifest.json).
- Raw traces and per-case receipts remain in
  [the headroom output directory](../headroom_development/).

The low-level PPO adapter separately passed six focused tests: exact actor
transfer from all four BC initializers, ordinary PPO save/load, physical-action
history, finite Gaussian likelihoods with saturated physical commands, and
explicit invalid-action failures. These are software tests, not PPO performance
results. PPO/SAC training requires a separate committed registration.
