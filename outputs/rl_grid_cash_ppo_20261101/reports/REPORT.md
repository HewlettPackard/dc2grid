# Matched PPO/SAC grid-relief and economics results

These results use the 24 previously consumed Dev-B conditions. They are
**development evidence**, not the original three-date Table 2, a fresh
final test, or a demonstrated replacement of the paper's benchmark.
The two tracks below have different objectives and must not be pooled.

## Track A: original low-level tracking task

The original reward, causal observations, full-work actuator space and
rule macro are unchanged. Means below use bias 0.5 and include the same
two-hour physical recovery in cash. Cash is modeled net cash, not profit.

| Controller family | Models | All physical gates | RMSE, kW | Facility relief, MW | PCC relief, MW | Net cash, USD/26h |
|---|---:|---|---:|---:|---:|---:|
| pid | 1 | Yes | 768.381 | 5.182 | 5.362 | -100,691.01 |
| old_sac | 1 | Yes | 469.089 | 5.383 | 5.314 | -100,645.83 |
| cooling_first | 1 | Yes | 0.174 | 5.461 | 5.461 | -100,366.36 |
| bc_only | 4 | Yes | 69.866 | 5.447 | 5.446 | -100,394.02 |
| guided_sac | 4 | Yes | 206.433 | 5.473 | 5.467 | -100,382.53 |
| union_original | 4 | Yes | 74.121 | 5.440 | 5.438 | -100,402.22 |
| union_conditioned | 4 | Yes | 73.975 | 5.442 | 5.441 | -100,401.70 |
| native_ppo | 4 | No | 726.908 | 5.938 | 5.992 | -100,434.39 |
| native_sac | 4 | No | 1,471.967 | 5.539 | 5.380 | -100,906.36 |

Final-step-only decisions (mean tracking compared with each seed's own
BC in both bias strata; individual-case regressions are in the JSON):

- **PPO:** four-seed tracking non-inferiority = **False**; joint relief/cash improvement = **False**.
- **SAC:** four-seed tracking non-inferiority = **False**; joint relief/cash improvement = **False**.

## Track B: disclosed economic-dispatch variant

All rows retain the existing full-residual, zero-ask bidding policy.
The learned policy allocates additional requested relief using a causal
performance-credit guard. The actual 90% capacity-service floor and
physical/recovery gates are enforced. More tracking error is permitted
in this separate task; it is not a free regulation-accuracy gain.
Means below use bias 0.5; qualification includes both biases.

| Controller family | Models | All gates | RMSE, kW | Facility relief, MW | PCC relief, MW | Net cash, USD/26h |
|---|---:|---|---:|---:|---:|---:|
| cooling_first | 1 | Yes | 0.184 | 6.783 | 6.783 | -97,170.44 |
| bc_only | 4 | Yes | 136.352 | 6.924 | 6.922 | -97,184.66 |
| union_conditioned | 4 | Yes | 133.475 | 6.913 | 6.913 | -97,192.27 |
| constant_one | 1 | Yes | 1,000.000 | 7.783 | 7.783 | -96,900.61 |
| greedy_budget | 1 | Yes | 1,209.691 | 8.019 | 8.019 | -96,841.49 |
| margin_ten | 1 | Yes | 1,937.785 | 7.817 | 7.817 | -96,734.30 |
| economic_ppo | 4 | Yes | 1,094.005 | 7.936 | 7.936 | -96,865.57 |
| economic_sac | 4 | Yes | 1,133.407 | 8.013 | 8.013 | -96,860.21 |

Predetermined four-seed joint-improvement decisions:

- **economic_ppo: False**.
- **economic_sac: False**.

The economic comparison includes the newly measured scripted budget
and price-margin controls, not just the weaker zero-overlay controller.
Facility/PCC differences and inventory effects remain explicit.
A constant offset is not temporal load shifting or learned forecasting.

## Exact training budgets

Track A counts five-second physical training transitions; Track B counts
15-minute transitions with physical service and recovery.
Budgets match **within** a track, not between task cadences.
PPO counts below are actual optimizer operations, not SB3 epoch counters.

| Track / algorithm | Completed models | Transitions per model | Actor updates by seed | Critic updates by seed |
|---|---:|---:|---|---|
| track_a / PPO | 4/4 | 552,960 | 3484, 3323, 2498, 2763 | 3740, 3579, 2754, 3019 |
| track_a / SAC | 4/4 | 552,960 | 551936, 551936, 551936, 551936 | 552192, 552192, 552192, 552192 |
| track_b / PPO | 4/4 | 24,576 | 3727, 3800, 3800, 3733 | 3983, 4056, 4056, 3989 |
| track_b / SAC | 4/4 | 24,576 | 24320, 24320, 24320, 24320 | 24576, 24576, 24576, 24576 |

All final checkpoints are the registered mandatory last step. No earlier
'best' checkpoint substitutes for an unsuccessful final model. The actors
were held fixed during 256 critic-only updates in each fit. PPO and SAC
differ in value/Q critics, entropy, exploration after initialization and
sample reuse; equal transitions do not mean equal computation.

## Verification and scope

- Track A raw primary checks: **144,000**.
- Track B raw primary checks: **57,888**.
- Earlier headroom experiment: 144/144 qualified, 36,720 raw checks.
- Failed training slots and incomplete evaluation cases are retained.
- No survivor-only four-seed family average is used.
- Development conditions are correlated; no significance claim is made.
- Five independent adversarial reviews informed the design; tests and
  raw data, not reviewer consensus, determine the claims.
- Frozen physics, prior studies, original manuscript and read-only
  external study materials were not changed.

## Reopening the evidence

- [Track A audited results](../track_a/development_report.json)
- [Track B audited results](../track_b/development_report.json)
- [Track A protocol](../track_a/registration.json)
- [Track B protocol](../track_b/registration.json)
- [Pre-learning headroom](HEADROOM.md)
- [Reserved calendar combinations](../final_reservation.json)

**The six reserved calendar combinations remain unopened.** They
are neither an unseen-workload-phase test nor a p90-tail holdout.
No final-test or paper-replacement claim follows from this report.
