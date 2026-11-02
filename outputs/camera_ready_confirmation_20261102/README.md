# Frozen-policy camera-ready confirmation

This is a new, prospectively registered study, not an extension or relabeling of
the previous failed promotion gate. It performs **no training** and leaves that
study's six-date reservation unopened. The baseline is commit
`b2978771d7e03c0e16ccfbba2712e64b97557bbb`.

## Closed comparison

- Sixteen final 24,576-transition high-level checkpoints: fractional PPO and SAC,
  each with and without the training-only credit potential, seeds 1200--1203.
- Twenty-seven causal scripted controls in the **same native fraction action
  space**: zero; fractions 0.25/0.50/0.75/1.00 at always-on or current posted
  margins 0/5/10/15/20 USD/MWh; a deadline-decreasing margin threshold; and one
  fixed credit-at-least-1-MWh full-allowance rule.
- The credit rule is new in this confirmation design. Its credit is the
  tracking-error allowance, **not battery inventory**.
- Existing full-admissible-residual, zero-ask bidding and the same analytic
  low-level executor are held fixed. Rejected bids retain the standing 5 MW
  contract. The actor does not observe the current handshake outcome before
  selecting its fraction.
- PPO's pre-tanh output is mapped once to the native fraction space. SAC and
  scripts act directly in that space. The first post-handshake tick fixes the
  requested MW for the entire window; the original per-tick guard remains.
- The older eight absolute-coordinate economic models are not reinterpreted as
  fraction-space policies. Their previous results and the failed native
  low-level PPO/SAC experiments remain part of the disclosed research history.

## Panels and provenance

The registered selector excludes all 83 consumed starts, the old six reserved
starts and their closed-26-hour neighbours, and the 12 already-invalid start
windows. The invalid-start list is not expanded a second time. The remaining
eligible pool has 133 starts. Eight dates are selected without policy outcomes:
two p75-but-not-p90 dates, three middle dates, and three low dates.

The primary panel has eight fixed dates, ten new date-unique stochastic streams
per date, and both bias strata: 160 cells. The secondary panel uses only ten new
streams per each of the three known peak dates: 60 cells. It is conditional
stochastic replication, **not unseen-calendar validation**. The two panels open
together, and all 43 policies are run: 9,460 new confirmation episodes.

Before that opening, the 27 scripts are checked on all 54 already-consumed
Dev-B (economic-track)/original-peak cells: 1,458 new episodes. The 864 exact, hash-bound
neural records from those panels are reused. These checks cannot select a new
model or screen out an inconvenient scripted comparator.

Load is NYC/NYISO 2023. The frozen loader pairs it by month/day/time with NYC
Central Park 2024 weather; this is calendar climatology, not contemporaneous
2023 weather. Workload phases repeat and have all appeared in training. Native
load/weather loaders verify every new closed 26-hour window, including its
endpoint and missing-value handling. No fresh p90-high calendar start remains.

## Frozen inference and admissibility

The primary candidate is fixed in advance: the four-checkpoint `potential_sac`
mean. The primary comparator is `script_f100_always`, not the best retrospectively
chosen script and not the historical absolute-MW greedy policy.

The estimand is the equal-date mean over stochastic noise, conditional on the
eight selected dates and four fixed checkpoints. It is not an action-averaging
deployment ensemble and not a population average over future training seeds.
Ten thousand paired bootstrap resamples draw noise indices within each fixed
date. Identical indices are used across policies, metrics, and bias strata.
Dates and checkpoints are not resampled.

At bias 0.5, both stress-window mean PCC relief and net 26-hour cash must have
point estimates **and 2.5th-percentile lower bounds** above +0.001 MW and +5 USD,
respectively. These are small numerical thresholds, not a claim of commercial
materiality. Bias-0 mean cash must not decrease. Every candidate and primary
comparator episode across both confirmation panels must pass the physical,
work, SOC, thermal, recovery, and actual 90% paid-service gates.

No failed case is deleted to repair pairing. An incomplete or disqualified
family stratum receives no admissible aggregate mean. Per-checkpoint and
leave-one-date-out contrasts are sensitivity analyses; a date-robust label
requires all leave-one-out point contrasts to clear both thresholds. Neither
the secondary panel nor an observed full-script frontier rescues a failed
primary rule. All outcomes are reported, including null and negative results.

## Metrics and checks

Service lasts 24 hours; recovery lasts two more hours with continuing physical,
workload, reference, and random-generator state. Full work is served and the
SOC target remains 0.5. The actual service floor is 90%; the common guard uses
a 92% target plus a 0.02 MWh buffer. Qualification is measured, not assumed
to follow automatically from that guard.

The raw audit recomputes settlement, metered cash, both relief boundaries,
allocation/credit identities, window constancy, fallback rates, work
conservation, closure, and reference/exogenous pairing. It distinguishes:

1. mean PCC/facility relief over the daily top-decile exogenous-load ticks;
2. the 24-hour difference of system-boundary maxima;
3. the corresponding 26-hour difference including recovery.

Recovery is allowed to set the maximum. A stress-mean improvement is not
automatically maximum-peak reduction. Cash is metered simulator cash, not
business profit. Extra efficiency dispatch is not automatically temporal
load shifting, accredited capacity, or verified peaker displacement.

## Execution

Use the repository's existing Python 3.11 environment. Thirty-two workers have
distinct CPU affinities, with one numerical thread each; this inference-only
experiment does not require new GPU training.

```bash
.venv/bin/python -m pytest -q outputs/camera_ready_confirmation_20261102/test_confirmation.py
.venv/bin/python outputs/camera_ready_confirmation_20261102/prepare_confirmation.py
# Commit the registration and all source before executing any panel.
.venv/bin/python outputs/camera_ready_confirmation_20261102/run_confirmation.py --stage development
.venv/bin/python outputs/camera_ready_confirmation_20261102/confirmation_analysis.py --stage development
.venv/bin/python outputs/camera_ready_confirmation_20261102/run_confirmation.py --stage final
.venv/bin/python outputs/camera_ready_confirmation_20261102/confirmation_analysis.py --stage final
```

The author attachment is read-only. Publication work starts from a protected
verbatim copy and produces a new manuscript, preserving unrelated author text.
Historical simulator/accounting results must not be silently presented as
reruns under this economic task.

## Technical correction before confirmation

The original registration and its Git commit remain immutable. A separately
hashed, committed technical amendment repairs the raw auditor and operational
resumption without changing policies, physics, cells, metrics, thresholds, or
statistical decision functions. Runtime verification checks that amendment,
the original source objects in Git, the unchanged scientific specification,
and every completed pre-amendment outcome.

Grid frequency is a closed-loop outcome, not an exogenous input. It remains in
the raw traces but is excluded from cross-policy input-equality assertions.
The auditor also mirrors the already-existing ledger's 1e-6 kWh backlog
comparison tolerance and requires the declared historical/new record format.

Restarting a stage reuses every verified completed episode, **including
physical failures**. It never re-rolls their seeds. Interrupted attempts and
old receipt logs are retained; a new complete receipt view identifies reused
and newly executed slots. An existing confirmation opening is validated and
reused byte-for-byte, never deleted or rewritten.

## Finalized outcome

All 9,460 confirmation episodes qualified. On the primary biased panel, the
fixed four-checkpoint SAC-plus-credit-potential mean improved stress-window PCC
relief by **0.866 MW** (paired conditional interval **0.836 to 0.896 MW**) and
26-hour modeled cash by **43.44 USD** (**41.21 to 45.66 USD**) relative to the
prespecified full-fraction rule. All four checkpoint mean contrasts were
positive; relief magnitudes ranged from 0.264 to 2.367 MW. The joint rule and
zero-bias cash guard passed.

This is not universal dominance. The strongest primary-panel cash script earns
81.9 USD more but provides 0.782 MW less relief than the candidate family mean.
The known-peak, new-noise comparison gives a smaller 0.053 MW relief increment
and 71.54 USD cash increment versus full fraction. PPO and plain SAC do not
displace the candidate on the economic comparison, and the earlier low-level
tracking result still favors analytic control or BC.

- [Final results and complete scope](reports/FINAL_RESULTS.md)
- [Camera-ready PDF](publication/full_build/camera_ready.pdf): **10 content pages**,
  with references beginning on page 11; 69 pages including all appendices.
- [One-page Table 2 preview](publication/build/table2_preview.pdf): best displayed
  means, including ties, are bold.
- [Clean manuscript source archive](publication/camera_ready_grid_relief_confirmed.zip):
  only `camera_ready.tex` is a manuscript entry point; internal strategy notes
  and alternative drafts are excluded.
- [Publication validation](publication/validation_receipt.json) and
  [fresh archive-build verification](publication/archive_validation_receipt.json).

The complete effort contains 10,918 newly executed episodes plus 864 exact
reused neural records, with 12,924,854 raw scalar-or-array checks. Twenty
scientific tests and five publication tests passed, and five independent
adversarial reviewers checked the final evidence. The 32,754 new raw trace
files remain local (71,319,302,587 bytes), with a committed hash inventory rather
than a claim that all bulk traces are Git-backed. No branch was pushed.
