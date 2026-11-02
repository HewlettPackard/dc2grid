# Verified SAC/PPO improvement research closeout

## Main finding

**A stronger learned SAC controller was obtained for the explicitly separated
economic-dispatch task.** It improves both grid relief and 26-hour cash over
the previous economic SAC and the earlier retrained-SAC comparator.

This is not a free improvement in regulation tracking. The new controller
allocates an allowed tracking-error budget, uses the existing full-residual
zero-ask bidding policy, and executes through the observation-only analytic
low-level controller. It is a hierarchical economic-dispatch variant, not a
replacement for the original rule-macro tracking benchmark without disclosure.

## Exact original peak-date comparison

The following means use the original peak dates, days **207/248/249**, noise
seeds **100--104**, and bias **0.5**. Every row uses the **same existing
full-residual macro**. Cash is the full 26-hour advantage over matched PID,
including the unchanged physical recovery.

**This is distinct from the original Table 2's rule-based macro.** In particular,
the correct prior retrained-SAC comparator here is 8.127 MW / USD 757, not the
6.007 MW / USD 648 numbers from the different rule-macro setup.

| Controller | Facility relief, MW | PCC relief, MW | Cash vs PID, USD | Tracking RMSE, kW | BESS throughput, MWh |
|---|---:|---:|---:|---:|---:|
| Prior retrained SAC | 8.127 | 7.794 | +757 | **686** | 31.149 |
| New economic PPO, V1 | 8.694 | 8.694 | +1,687 | 1,236 | 0.020 |
| New economic SAC, V1 | 8.782 | 8.782 | +1,794 | 1,351 | **0.000** |
| Fractional SAC with credit potential | **8.825** | **8.825** | **+1,868** | 1,458 | 0.020 |

Bold identifies the best displayed mean among these rows, not statistical
significance. The rounded 0.000 MWh value is nonzero (about 0.000093 MWh).
The analytic controller, included in the full result data, remains much more
accurate at approximately 0.153 kW RMSE on this full-residual peak panel.
New neural rows average four training seeds within each date/noise case; the
prior retrained-SAC row is one frozen model. These are descriptive means,
not significance or equal-computation claims.

The final learned SAC therefore improves over the matched prior retrained SAC by:

- **0.698 MW facility relief**, or **8.59%**;
- **1.030 MW PCC relief**, which is the actual grid-import boundary;
- **USD 1,110.99 per 26-hour case** in modeled cash;
- substantially lower battery throughput, but **higher tracking RMSE**.

Its four model seeds passed **120/120** original-peak evaluation cases,
including both biases, all physical/recovery checks, and the actual 90%
capacity-service floor. Its minimum service ratio was **92.0049%**.
That ratio measures performance-weighted capacity-hours against the matched
maximum-admissible reference, not battery energy.
Net modeled cash is still approximately **minus USD 197,762**, not an operating
profit. All 24 economic learned checkpoints and four scripted controls were
included in this 840-episode regression comparison; unsuccessful alternatives
were not hidden.

## Development evidence and the remaining trade-off

On the separate, already-consumed Dev-B panel, the final learned SAC gives:

- **8.870 MW** facility and PCC relief;
- **USD 383.55/case** better biased-case cash than the analytic control;
- **96/96** qualified model/case pairs;
- **0.857 MW and USD 73.31/case** improvement over V1 economic SAC.

All four matched seeds improved both relief and cash over V1 SAC on biased
Dev-B cases. The per-seed
PCC improvements range from 0.071 to 2.416 MW; cash gains range from USD 18.51
to USD 132.34. The mean improvement is therefore real in this measured cohort,
but its magnitude is not uniform across training seeds.

The result improves the observed trade-off, not every competing extreme:

| Dev-B comparison, bias 0.5 | PCC relief difference | Cash difference |
|---|---:|---:|
| Final learned SAC minus greedy-budget rule | +0.851 MW | +USD 54.60 |
| Final learned SAC minus price-margin rule | +1.053 MW | -USD 52.60 |

Consequently **no family passed the preregistered composite per-metric
promotion gate**, which required exceeding both the strongest cash family and
the strongest relief family. That conservative envelope is not a single
achievable incumbent. The positive original-peak regression results were not
used to relax the gate or reselect a checkpoint.

**The six reserved new calendar combinations remain unopened.** These results
are not a fresh-test claim, an unseen-workload-phase result, or new p90-tail
generalization. No further adaptive campaign was run after the registered cap.

## What PPO did and did not improve

- In the original low-level task, four PPO and four native SAC fits each
  completed **552,960 five-second transitions** from the matched BC initializers.
  Neither family retained the required four-seed BC non-inferiority and physical
  qualification. On biased Dev-B, PPO's mean RMSE was 726.9 kW versus BC's
  69.9 kW. These new low-level models are not recommended replacements.
- In the economic task, both algorithms improved over the previous tracking
  controllers under the disclosed trade-off, but **SAC produced the strongest
  learned result** in this campaign.
- Fractional actions alone did not solve the economic gap. The credit-potential
  ablation helped SAC but hurt PPO. It is not evidence that shaping universally
  helps, or that PPO is generally inferior.

PPO used ordinary SB3 Gaussian policies in pre-tanh coordinates, avoiding an
unstable inverse-tanh likelihood reconstruction. Initial BC means/distributions,
physical action history, ordinary checkpoint loading, true terminal handling,
and actor-preserving critic calibration were verified. Actual optimizer
operations were recorded; PPO rollout counts were not called update counts.

## Training and verification

- **32 full fits:** 8 low-level fits at 552,960 physical transitions each;
  24 economic fits at 24,576 macro transitions each.
- Every fit retained its mandatory final checkpoint. No early-best checkpoint
  replaced an unsuccessful final model. Native smoke fits are recorded
  separately, not treated as independent replications.
- **2,376 new evaluation episodes completed 26 hours; 2,272 qualified.**
  The 104 disqualifications belong to the new original-task low-level candidates
  and remain in the evidence.
- **483,792 scalar/array checks** across the distinct evaluation audits.
- **113 focused tests** across the headroom, native PPO/SAC, economic-task,
  fractional-task, reporting, and regression-integration test groups.
- Five independent adversarial reviewers informed the work. Their claims were
  checked against code and raw data; unsupported claims were corrected rather
  than adopted by consensus.

The original paper archive, previous verified union-SAC archive/report,
frozen physics, and read-only external study materials were not modified.
The paper was not silently rewritten with results from a different task.
No Git push was performed.

## Reproducibility and evidence

- Pre-change snapshot: `12c362245f51a4f143e76d2d458231bef4ede398`,
  tag `baseline/pre-economics-ppo-20261101`.
- Full V1 outcome snapshot: `5e2bf2855a3da4ac8cfea1763df8e9d8b28e9f82`.
- Fractional outcome snapshot: `d378e91f543568d62195f9df6fc01e0d1f861c27`,
  tag `results/fraction-potential-sac-dev-20261101`.
- Original-peak plan committed before evaluation:
  `779c2881a8b6f905c29ab2d4f9b5b2208dffdbb2`.
- [Original-peak raw-audited results](original_peak_regression.json).
- [Complete factorial results](FRACTION_REPORT.md).
- [Economic frontier figure](economic_frontier.png).
- [Both original experiment tracks](REPORT.md).
- [Headroom mechanism and wear sensitivity](HEADROOM.md).
- [Native calendar-coverage proof](calendar_coverage_verification.json).
- [V1 bulk inventory](inventory_v1.json) and
  [fractional bulk inventory](inventory_fraction.json).

Selected final models, protocols, code, and compact case receipts are committed.
Bulk traces, replay buffers, and detailed window-level training logs remain on
disk with hashes. **Checksums are not a bulk-data backup.**
