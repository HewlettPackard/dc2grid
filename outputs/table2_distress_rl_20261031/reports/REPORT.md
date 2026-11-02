# Verified Table 2 replacement

## Deliverables

- [Revised camera_ready.tex](../publication/camera_ready.tex)
- [Revised paper archive](../publication/camera_ready_table2_updated.zip)
- [Compiled two-table preview](../publication/build/table2_preview.pdf)
- [Machine-readable final report](final_report.json)
- [All final episode metrics](final_episode_metrics.csv)
- [All-seed learning curves](learning_progress.png)
- [Locked model inventory](../final_candidate_lock.json)
- [Study implementation and reproduction notes](../README.md)

The original manuscript archive, earlier registered studies, user attachments and
all files under `/workspace/AAAI-demo` remain unchanged. Only the requested
camera-ready source is revised inside the new paper archive.

## Main result

There is a defensible improvement over the earlier SAC/PID controller pipelines,
but **not a demonstration that online SAC outperforms behavior cloning or the
analytic controller**. The replacement retains those stronger baselines and does
not hide the failed unregularized SAC arm.

The paper's original three peak dates remain the main Table 2 regression panel.
They were already consumed earlier in this project and are not relabelled as an
untouched test. The six newly evaluated calendar combinations are reported
separately, regardless of which panel gives a more favorable ranking.

### Original peak-date regression: rule macro, beta=0.5

Means below use the 15 day/noise cases. Four training-seed predictions are averaged
within each case for BC and Guided SAC. Battery throughput and cash include the
fixed two-hour recovery; tracking and relief cover the service day.

| Controller | Facility relief MW | RMSE kW | Revenue change vs own beta=0 | BESS MWh | Cash vs PID USD |
|---|---:|---:|---:|---:|---:|
| PID | 5.641 | 821.9 | -1.596% | 29.344 | 0 |
| Previously retrained SAC | 6.007 | 658.4 | -0.342% | 31.682 | +648 |
| Analytic | 5.626 | 50.2 | -0.021% | 0.950 | +466 |
| BC only | 5.583 | 290.5 | -0.261% | 0.109 | +388 |
| Guided SAC | 5.607 | 382.7 | -0.329% | 0.719 | +427 |

Guided SAC has roughly 53.4% lower biased-case RMSE and 97.5% less battery
throughput than PID, with a much smaller revenue penalty and more closed-cycle
cash. Its mean facility relief is slightly lower, not higher. The previously
retrained SAC remains the largest-relief controller on these peak-date cases.
Guided SAC and BC each qualify on 120/120 model/case pairs over both biases;
PID qualifies on 26/30.

### Fresh six-date panel: rule macro, beta=0.5

| Controller | Facility relief MW | RMSE kW | BESS MWh | Cash vs PID USD | Physical pass / planned, both biases |
|---|---:|---:|---:|---:|---:|
| PID | 5.035 | 714.1 | 27.665 | 0 | 35/36 |
| Previously retrained SAC | 4.705 | 577.6 | 37.942 | +238 | 36/36 |
| Analytic | 5.043 | 0.2 | 0.0001 | +418 | 36/36 |
| BC only | 5.021 | 73.9 | 0.090 | +382 | 144/144 |
| Guided SAC | 5.060 | 194.7 | 0.918 | +398 | 144/144 |

Guided SAC has roughly 72.7% lower biased-case RMSE and 96.7% less battery
throughput than PID. The relief difference is only +0.024 MW; it is not described
as a large or statistically significant gain. BC and the analytic controller
track more accurately than Guided SAC.

### Per-training-seed fresh outcomes

Same rule macro and beta=0.5 scope:

| Guided seed | Selected online step | RMSE kW | Relief MW | Cash vs PID USD | RMSE change vs exact BC kW | Relief change vs exact BC MW |
|---|---:|---:|---:|---:|---:|---:|
| 1100 | 4,096 | 95.9 | 5.092 | +421 | +34.0 | +0.062 |
| 1101 | 8,192 | 278.3 | 5.080 | +403 | +207.8 | +0.061 |
| 1102 | 4,096 | 179.3 | 5.117 | +419 | +87.3 | +0.083 |
| 1103 | 4,096 | 225.5 | 4.950 | +350 | +154.1 | -0.053 |

All four seeds are retained. Positive RMSE changes mean worse tracking than BC.
The modest relief/cash tradeoffs do not establish a generic incremental RL benefit.

## High-level control improvement, kept separate

The second macro uses zero asking price and the full admissible posted residual,
bounded by the unchanged 30 MW limit. It is an inspectable price-taking rule,
not a learned macro improvement and not a universal optimality theorem.

With the same four locked Guided-SAC hardware models, on biased cases:

| Panel | Rule-macro relief MW | Full-residual relief MW | Paired 26h cash increase USD | Rule RMSE kW | Full-residual RMSE kW |
|---|---:|---:|---:|---:|---:|
| Peak-date regression | 5.607 | 7.773 | +3,866 | 382.7 | 460.0 |
| Fresh calendar panel | 5.060 | 6.535 | +3,554 | 194.7 | 252.5 |

Every full-residual Guided-SAC case passes the 90% performance-weighted
capacity-hours floor and the physical/recovery gates. Higher service participation
is not confused with a low-level improvement: macro regimes are reported
separately. Under full-residual bidding on fresh biased cases, PID delivers
6.560 MW versus Guided SAC's 6.535 MW, but has 784.7 versus 252.5 kW RMSE.

## What was trained

- Fixed analytic-teacher corpus: 90 training-date/bias cases and 1,555,200 real
  transitions. All completed and passed service gates. An earlier three-case
  demonstration smoke is separate.
- Four primary BC fits, 12,000 supervised updates each, with training-support
  checkpoint selection. Two additional observation diagnostics are not RL rows.
- Eight real SAC fits: Warm SAC and Guided SAC, each with seeds 1100-1103.
  Every fit completed 153,600 new online transitions.
- Exact totals: **1,228,800 online transitions, 1,220,608 actor updates,
  1,222,656 critic updates**, including 2,048 critic-only calibration updates.
- Within-seed initialization and warmup replay hashes match between variants.
- Warm SAC has no eligible checkpoint in any seed. Guided SAC has four eligible
  selected models. Every fit still ran to its full registered budget.

Selection was relief-primary subject to all-case physical/recovery gates, an RMSE
bar and a cash bar. It was performed on development data only. The selected
checkpoints are early, at 2.7-5.3% of their respective online budgets.
The [learning figure](learning_progress.png) retains every milestone, including
late deterioration and one incomplete development evaluation. The final
153,600-step policies are not substituted for the selected models.

The learned controller receives 18 original observations plus its own four prior
commands, a causal request average and a clock. It directly chooses pump, HVAC and
BESS; full work service is fixed. It does not use the analytic observer, hidden
reference state, future inputs or a tracking projection at deployment.

## Physical and scientific checks

- **1,848 final trajectories**, all complete through service and recovery.
- **2,376 planned slots**; the difference of 528 is explicitly non-admitted
  Warm-SAC slots, not dropped test outcomes.
- **Seven service-SOC failures**, all on legacy comparators, retained in the means.
  All learned and analytic final cases pass.
- **591,360 scalar-or-array checks** from raw final traces; no new policy rollouts
  during analysis. An additional 168 comparator traces were checked before
  training with 52,360 checks.
- 36 controller/training tests plus four statistics tests passed. The zero-BC
  update matches SB3 exactly; terminal handling, action identities, checkpoint
  loading, calibration invariants and nonmutating diagnostics are covered.
- Four independent adversarial reviewers audited implementation, learning,
  physical results, data/provenance and publication claims. Unsupported iid tests,
  confidence claims and speculative causal explanations were rejected.
- All 95 frozen physics source files, five data files and locked checkpoints
  remain unchanged. Both final panels are now consumed and must not be reused as
  untouched tests for another tuned method.

The analytic controller's peak-date tracking outliers are real actuator saturation,
not inconsistent units: some requests demand about 24 MW of additional import
while HVAC headroom plus BESS charging supplies about 20.07 MW. The largest
verified error is about 3.93 MW during those saturated ticks.

Relief is predominantly HVAC modulation. The workload cycle is repeated, weather
is calendar-matched climatology, and thermal headroom was not contested in these
panels. The frozen HVAC model does not price excursion depth separately during
fixed-setpoint recovery. State closure is genuine, but this is not a claim about
real-plant thermal flexibility, accredited capacity, emissions, business profit
or superior control of actual maximum grid demand.

## Manuscript validation and remaining build limitation

Both embedded tables are byte-identical to the verified, data-rendered LaTeX
fragments. Best displayed means are bold, including ties at the shown precision;
qualification compares pass rates, and bold does not imply statistical significance.
Their two-page PDF preview compiles with shell escape disabled, has no
overfull warnings, and was visually checked. The exact revised section and
appendix also compile as a five-page scoped preview with resolved citations and
cross-references, independently of unrelated manuscript packages.

The supplied full manuscript has a pre-existing missing dependency,
`algorithm.sty`, in this environment. No system TeX installation was performed.
A full-manuscript PDF is therefore not claimed; the revised source and original-
inventory archive are supplied for the author's normal complete TeX environment.

No commits or pushes were made.
