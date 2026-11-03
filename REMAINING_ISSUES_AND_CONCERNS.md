# Known remaining issues, limitations, and validation concerns

## Public addendum scope

This documentation-only addendum is published on
`release/grid-relief-reproducibility-20261102`. The frozen result tag
`repro/20261102/pid-comparison` remains at
`246d60e49e2478ededba81e39e60c0141fec41fa`; no previous result tag, scientific
file, checkpoint, dataset, or PDF is changed by this addition.

The local research commits cited below are provenance identifiers, not parents
of this clean public history. Some later audit records and the new Table 1
supporting files are not included in this release. Their links are explicitly
marked **local-only** and lead to the availability inventory in Section 9,
not to a public download. Publishing this disclosure does not publish those
underlying records or constitute a new evaluation.

The original local report is preserved separately. Its SHA256 is
`efc97001e9fea3191dcdb1bbb20e81f0a5684a256da644383d3397e912c10ce8`.
The 41 categorized items below are unchanged apart from public-link availability
annotations. This additional document is not inserted into the frozen release
file inventory; its version is identified by its own Git commit.

**Status date:** 2026-11-03.
**Local evidence snapshot:** `f477cdea7a1a7461d5d67fa27832c36d45d82388`.
**Original code audited:** `7550013331a865fa45684cb7cd973279312d4678`.
**Corrected code endpoint audited:** `5c7e057caf686853934a19f5c56e848dbc96ae68`;
the frozen scientific implementation was subsequently preserved at `34e01546`.

This register consolidates the currently identified open items from the source
audits, experiment reports, and publication checks. It is not a claim that every
possible repository defect has been discovered. No simulator, checkpoint,
registration, result, or previous PDF was changed to prepare this document.

The [detailed source audit (local-only)][audit] supplies original/corrected function and line
anchors, counterexamples, fixed findings, and evidence boundaries. This register
focuses on what remains unresolved or requires careful interpretation.

## 1. Executive summary and status vocabulary

**Frozen means version-locked, not certified bug-free.**

| Item | Origin | Current assessment |
|---|---|---|
| RegD window centering | Inherited from original code | Confirmed unresolved signal-model defect; impact on comparative results after repair is unmeasured. |
| Battery/PCC voltage coupling | Inherited from original code | Confirmed remaining model-coupling limitation; metered power accounting is a separate, corrected path. |
| Macro observation-Box sampling metadata | Introduced during the corrections | Confirmed software defect; no affected observation-sampling path was identified in the checked producing runs. |
| Generic benchmark/callback issues | Mostly inherited, some partially repaired | Several entrypoints still have metric/cohort risks; this does not establish that the isolated audited studies suffered those errors. |
| Reward, workload, reference and economic abstractions | Intentional design choices or changed definitions | Need explicit scope and validation, not blanket classification as SAC implementation bugs. |
| Learning, generalization and result trade-offs | Observed scientific limitations | Successful tracking or one economic contrast is not dominance across tasks or metrics. |
| Historical provenance and external reproduction | Incomplete evidence or distribution/access gaps | Some producing records and externally accessible evidence remain unavailable or unverified. |

Statuses used below:

- **Open defect:** confirmed implementation behavior contradicts its declared
  contract or interface.
- **Model limitation:** a known abstraction or missing coupling; actual impact
  depends on the claim and experiment.
- **Conditional/tooling risk:** a problematic path exists, but its use in a
  particular reported result must be established separately.
- **Evidence/interpretation gap:** a conclusion is not supported by the current
  records; this is not itself proof that a number is wrong.

No priority or potential impact in this document is a measured causal effect
unless a specific experiment is cited.

## 2. Open simulator and interface issues

### SIM-01: RegD centering does not enforce the advertised window neutrality

**Status:** open defect. **Origin:** original code; retained in `physical_v1`.
**Evidence:** M12 in the [macro audit ledger (local-only)][macro-ledger];
[current generator](c2g_env/physics/macro_grid.py#L572).

The generator emits the preceding samples unchanged, subtracts the block mean
only from the final sample, and then clears the buffer. It neither demeans the
whole block nor implements the advertised rolling-window correction.

For a constructed 180-tick test with every raw sample equal to 0.5, the emitted
window mean is approximately 0.497222, not zero. This is a counterexample to
the algorithm's claim, not a measured mean of the published evaluation signals.

**Why it matters:** sustained energy requests can change battery reserve,
tracking, cycling, recovery expenditure, and qualification. A shared generator
does not guarantee that its error cancels between PID, SAC, and other policies.
The effect of repairing it on the published rankings has not been measured.

**Required resolution:** first specify the intended contract: disjoint windows,
sliding windows, or a bounded cumulative-energy target. These are not equivalent.
Test the actual declared property, clipping, reproducibility, and elapsed-time
handling; a small average across many window means is insufficient. Apply and
document the intentional load-correlated stress bias separately. This audit
does not establish the real-world regulatory requirements of a particular market.

The deliberate `beta = 0.5` stress intervention is not itself this defect.

### SIM-02: PCC voltage omits the battery's actual net-power contribution

**Status:** model-coupling limitation. **Origin:** original code; retained.
**Evidence:** R1 in the [physical audit ledger (local-only)][physical-ledger];
[voltage calculation](c2g_env/physics/electrical.py#L327).

Voltage drop uses gross facility power and facility reactive demand, rather than
the battery-adjusted net exchange at the declared PCC. An audited same-state
comparison with zero versus 5 MW actual battery discharge changed net import
and tracking response, but did not change the reported voltage.

**Important boundary:** corrected metered grid import already subtracts actual
battery output; tracking and electricity expenditure use that metered boundary.
See [grid-import telemetry](c2g_env/env_low_level.py#L763). Frequency coupling
also includes battery response. This is not a finding that the whole grid model,
the reported PCC MW, or the electricity bill ignores the battery.

**Why it matters:** voltage observations, voltage penalties and voltage-related
safety decisions may differ after repair. Their effect on the reported policy
comparisons is not isolated by the existing audit.

**Required resolution:** define the battery's connection/topology and consistent
active/reactive-power boundary. Use actual SOC-limited power, not requested
dispatch. Test charging, discharging, zero output, clipping and safety thresholds;
then perform coupled-policy sensitivity evaluation.

### SIM-03: Corrected macro observation Box has stale sampling metadata

**Status:** open software defect. **Origin:** introduced during the corrections.
**Evidence:** C01 in the [macro audit ledger (local-only)][macro-ledger];
[Box bound mutation](c2g_env/env_high_level.py#L209);
[checkpoint preflight (local-only)][model-preflight].

The corrected environment sets observation coordinate 8's upper bound to
infinity after constructing the Box. The cached `bounded_above` flag remains
true. Random observation-space sampling can consequently attempt an invalid
finite-interval uniform draw and fail. Saved high-level checkpoint metadata
also contains this inconsistency.

**Impact boundary:** finite real observations and normal MLP policy prediction
can still work. The checked producing paths behind the PID comparison use
actual simulator observations, not this observation-space sampler. Direct
sampling calls found in the registered sources concerned action spaces or
replay buffers. The installed vector-policy prediction path was also checked.
There is no identified contamination of those results from this defect.

An infinite midpoint of an unbounded interval is not independent proof of
damaged policy weights. Action-space sampling is a different operation.

**Required resolution:** construct a consistent Box and address already-saved
spaces through explicit, versioned compatibility handling. Test sampling,
membership, model loading and prediction equivalence. Do not silently overwrite
historical model ZIPs; changing serialized metadata changes artifact hashes even
without retraining.

### SIM-04: RegD timebase/calibration still assumes five-second ticks

**Status:** remaining timebase limitation. **Origin:** inherited.
**Evidence:** M13 in the [macro audit ledger (local-only)][macro-ledger].

Window tick counts and AR-process calibration retain five-second assumptions.
For example, 180 ticks at a ten-second step span 1,800 seconds, not 900 seconds.
Corrected macro settlement timing does not repair this separate signal generator.

**Scope:** not an explanation of the reported default five-second NYC runs.
It matters for nondefault clocks and claims of general timestep support.
Validate each market's declared window, not an assumed universal window.

**Required resolution:** derive/calibrate the signal process from elapsed time,
or explicitly reject unsupported timebases. Test nondefault and partial intervals.

### SIM-05: Declared clearing-temperature configuration is ignored

**Status:** open configuration limitation. **Origin:** inherited.
**Evidence:** M15 in the [macro audit ledger (local-only)][macro-ledger].

The YAML advertises `handshake.sigmoid_k_p`, but clearing uses a local literal
`k_p = 0.15`. Changing the YAML value alone therefore does not change that
implementation. This does not mean that no sigmoid is used.

The default value matches the literal; no effect on a published default run has
been established. Wire the declared setting through and test a nondefault value,
or remove/document the unsupported setting in a new version.

### SIM-06: Battery charging-limit telemetry overstates available charge power

**Status:** open diagnostic inconsistency. **Origin:** inherited.
**Evidence:** R2 in the [physical audit ledger (local-only)][physical-ledger];
[battery implementation](c2g_env/physics/bess.py).

The reported `max_charge_mw` and actual charge derating use different SOC windows
(0.05 versus 0.10). At SOC 0.90, the reported available limit is approximately
5 MW while an actual charge request is limited to approximately 2.5 MW.

The audited tracking/reward path uses actual battery power, not this diagnostic;
no direct effect on those reported metrics was established. Controllers or
tools that trust the limit field could nevertheless receive misleading data.
Use one consistent limit calculation and test the SOC boundaries.

### SIM-07: Optional/legacy battery backends are not interchangeable

**Status:** conditional backend/interface risk. **Origin:** inherited.
**Evidence:** R4 in the [physical audit ledger (local-only)][physical-ledger].

The legacy selector can choose PySAM based on package availability. The adapters
declare different capacities: simple 15 MWh/5 MW versus PySAM-labelled
150 MWh/50 MW. The PySAM adapter also lacks the `_age_frac` attribute consumed
by the inspected low-level information construction.

The optional backend was not executed in this audit; its complete native-unit
compatibility and the historical paper's backend were not established.
Corrected `physical_v1` explicitly selects the simple backend, avoiding this
optional path for the audited studies.

Before advertising interchangeable backends, make selection explicit, reconcile
units/capacities/interfaces and test both. Do not attribute this conditional
legacy risk to corrected SAC results without runtime evidence.

## 3. Remaining modeling and objective-design concerns

| ID | Concern and scope | Required interpretation or follow-up |
|---|---|---|
| MODEL-01 | Work is an energy-equivalent dynamic-power proxy, not executed job counts, FLOPs, deadlines or application QoS. | Do not equate work-kWh conservation with demonstrated real computing-service preservation. Validate a richer service model before making those claims. See R3 in the [physical ledger (local-only)][physical-ledger]. |
| MODEL-02 | Queue state resets between episodes; a per-step backlog penalty is not terminal repayment. | Within-episode conservation alone does not close debt. The two-hour audited recovery addresses its declared closure conditions, not every possible long-run service obligation. Preserve service-end failures even if later recovery closes. |
| MODEL-03 | The original low observation clips backlog and does not expose every relevant clock, work, or reference state. | Distinct physical situations can share an observation. Consider causal history or explicitly declared state features; do not silently grant future information or privileged reference access. See [observation construction](c2g_env/env_low_level.py#L822). |
| MODEL-04 | A fast-controller discount of 0.99 at five seconds has an effective discounted horizon near 8.3 minutes. | This can be short relative to storage/work/recovery obligations. It is a design concern, not an SAC update bug, and must not be generalized to macro policies with different step durations or discounts. |
| MODEL-05 | Served-work rewards and actual PCC-dollar cost replaced paper-defined throttle/normalized-power proxies while outer coefficients could remain unchanged. | Reward meaning, scale and relative incentives changed. Compare rewards within a declared version; calibrate objectives explicitly rather than calling every difference a repaired formula. See the [audit clarification (local-only)][audit]. |
| MODEL-06 | Relief is against an independent simulated nominal reference, with specified operating points. | This is not an externally validated utility/customer baseline. Changing reference controls changes the estimand; document the reference alongside every relief claim. |
| MODEL-07 | Grid cash excludes business income, capital cost and a monetized aging charge; the electricity model uses its declared LMP settlement assumptions. | Do not call grid cash total business profit or assume a validated real import/export tariff. Battery wear and tracking remain separate trade-offs. |
| MODEL-08 | Frequency/voltage and plant models are simplified, not a demonstrated real deployment or full network validation. | Results support the stated simulated task. Physical calibration, topology and deployment robustness require separate evidence. Passing internal accounting checks does not supply that validation. |
| MODEL-09 | Frozen data combine NYC 2023 load with month/day-aligned 2024 weather, and workload phases repeat. | Do not describe this as a contemporaneous replay of all real measurements from one historical day. Generalization and joint load/weather realism are limited by this composition. See [the comparison's scope discussion][pid-source] and [release documentation][release-guide]. |

## 4. Generic evaluator and callback issues still requiring care

These are not a claim that every published study used the problematic path.
The new Table 1 collector independently handles full fast horizons, reset-to-end
aging, explicit fixed hardware, fresh controllers, per-case random seeds and
unambiguous metric keys. The PID/economic studies have their own registered
runners and audits. Fixing an isolated collector does not repair every generic
entrypoint.

Source detail: E02-E09/E12 in the [evaluation ledger (local-only)][evaluation-ledger],
TRAIN-006 in the [training ledger (local-only)][training-ledger], and the current
[generic benchmark](evaluation/run_benchmark.py) and
[metrics callback](baselines/metrics_callback.py).

| ID | Remaining issue | Scope and resolution |
|---|---|---|
| EVAL-01 | Generic aging subtracts the first post-step age rather than reset age, losing the first increment. | Use reset-to-service-end state. The separate corrected Table 1 collector already does so. |
| EVAL-02 | Generic no-inner-controller paths do not collect the same fast-level metrics. | Explicitly instrument fixed hardware; distinguish uncollected data from physical zero. The new Table 1 no-agent row uses an explicit fixed-action callback. |
| EVAL-03 | Some hardware SD exports use double-underscore keys such as `tracking_rmse__std`. | Declare a versioned schema and test all consumers. Do not silently omit SDs or relabel SD as SEM/CI. |
| EVAL-04 | A random controller can sample a construction environment's seed-0 action-space stream rather than an episode-local stream. | Seed the actual policy action space per case. Global Python/NumPy/Torch seeds alone are insufficient. |
| EVAL-05 | Generic controller reuse/reset does not guarantee independent initial state for every controller. | PID is reset when its method is available; bang-bang has no reset method in the inspected implementation. Prefer fresh case-local construction or a tested reset contract. |
| EVAL-06 | Generic macro survival can be inferred from call count rather than full fast completion. | Check actual elapsed ticks, completed windows and terminal/fault flags, including failure in the last macro call. |
| EVAL-07 | The generic PPO normalizer loader returns no normalizer if its file is absent. | If a checkpoint requires normalization, absence must fail explicitly. Not every PPO necessarily requires one. Choosing the proper macro/fast environment class was repaired; that is distinct from enforcing the required normalization artifact. See [the loader](evaluation/run_benchmark.py#L308). |
| EVAL-08 | LLM warm-up filtering changes the evaluated cohort, while requested counts can remain unchanged; excessive skipping can fall back to all cases. | Define the cohort before execution, preserve denominators and fail clearly for invalid settings. The new corrected Table 1 made no LLM calls. |
| EVAL-09 | Thermal-warning boundaries differ (`>= 33 C` versus `> 33 C`) and paths use normalized observations versus actual temperatures. | Declare a shared metric contract or label the difference. Warning-tick fraction is also not the fraction of episodes with any warning. |
| EVAL-10 | The low training callback retains its `>= 287` survival fallback when planned-horizon fields are absent. | Corrected physical infos supply the real horizon and completion flags, so that path is fixed. Other/legacy infos remain a conditional risk; require the fields rather than treating the fallback as universal. See [the callback](baselines/metrics_callback.py#L351). |
| EVAL-11 | Generic callback aggregation maps empty metric lists to zero, and some missing info fields have zero defaults. | This can make uncollected metrics look favorable when a wrong info schema is supplied. The audited hierarchical SAC schema mismatch was fixed, but no universal audit of all other callback users is claimed. Require/validate metric schemas or mark missing values explicitly. See [aggregation helpers](baselines/metrics_callback.py#L337). |

The corrected physical fast-RMSE aggregation, causal interface changes and
incomplete-comparison rejection should not be listed as still-unfixed merely
because `legacy_v0` preserves old behavior. Explicit mode/version labels remain
essential when generic tools or historical artifacts are reused.

## 5. Result validity, interpretation and learning concerns

### RESULT-01: Robustness to the remaining simulator repairs is unmeasured

The [PID comparison PDF][pid-pdf] is an audited, versioned record of its frozen
implementation. That is different from proving the same conclusions under
repaired RegD centering and voltage coupling.

The source-path review identified no observation-Box sampling contamination of
those producing runs. It did not establish that the RegD or voltage limitations
are harmless. Shared defects need not affect controllers equally. Do not
silently promote conditional frozen-model evidence into an unqualified
physical-model or real-world validity claim.

A versioned, paired sensitivity study remains necessary before asserting that
all reported rankings, cash gains and qualification outcomes survive those
repairs. The existing PDF's unchanged bytes should be preserved; its earlier
limitations discussion is not a substitute for this later issue disclosure.

### RESULT-02: Cross-version improvements are not one-factor causal estimates

The old-to-new pipeline changed physics/accounting, replay, reward definitions,
training coverage, checkpoints and evaluation/recovery. The corrected SAC/PID
comparison is matched within its declared protocol, but the entire historical
change is not an algorithm-only or one-bug-only improvement.

The PID reproduction recovers 6.7385 versus 5.6408 MW and an observed cooling/
remaining-facility decomposition. It does not causally allocate the difference
among individual fixes. Electricity accounting alone does not explain that MW
change. See [the PID study record][pid-readme].

### RESULT-03: Architectures, references and qualification gates differ

Keep distinct:

- Rule-based versus full-residual zero-ask macro bidding.
- Native hardware freedom versus full-work overrides.
- Analytic execution versus direct learned hardware control.
- Extra-response allocation versus original market-bidding policies.
- Nominal-operation relief baselines, own-zero-bias revenue changes, and
  panel-specific cash comparators.
- Physical/recovery gates, paid-service floors and economic credit guards.

Gains across these changes are end-to-end effects, not automatically isolated
SAC effects. A 23-observation/one-action economic allocator is not a replacement
for a 19-observation/two-action original-bidding macro.

**Important existing metadata clarification:** the full-residual historical
tracking counts in Q.5 and the first five Q.6 rows already enforce their recorded
90% paid-service/capacity floor. The PDF's abbreviated historical-gate wording
is incomplete. Rule-macro counts do not enforce that floor. The published
[Q.5/Q.6 erratum][gate-erratum] clarifies metadata without changing any number
or admission outcome. Do not repeat the oversimplification that every tracking
row lacks a service floor.

### RESULT-04: Completion, service reserve and recovery are different outcomes

- Both new PID variants completed all 220 cases and recovered, but each passed
  only 177/220 complete gate sets. Their identical 43 failures concerned
  service-end SOC. Recovery does not erase those failures.
- The new Table 1 study completed all 35 service/recovery cases, with 31/35
  supplemental passes.
- Means from complete but disqualified cohorts remain diagnostic and use all
  planned complete cases. They are not survivor-only averages.
- An incomplete case must not be converted into a full-horizon primary result
  by averaging prefixes or dropping failures.
- Some PDF means use bias 0.5 while the adjacent pass counts cover both biases.
  Their denominators are not interchangeable.

These distinctions are already recorded; the remaining concern is preserving
them in every paper table, chart, export and subsequent comparison.

### RESULT-05: The new nominal Table 1 is not an across-the-board SAC win

Under the registered default/day-0/bias-0 panel:

- Rule macro + retrained SAC has 272.6 +/- 32.9 kW RMSE versus PID's
  394.7 +/- 230.8 kW: 30.9% lower mean, with four of five paired cases better.
  Their actual macro contracts match.
- SAC has $27.74 less service regulation revenue, 12.6% more modeled service
  battery aging, and $129.69 less 26-hour grid cash than PID.
- The matched original-bidding macro SAC accepts 0/480 bids and passes the
  supplemental gates in 1/5 cases. The other four end service below SOC 0.20;
  all recover afterward.
- Recorded training/native configuration hashes match. No static market,
  plant or reward-scalar mismatch was found; a complete reconstruction of every
  historical runtime argument is not available. No causal mechanism rescuing
  the macro policy has been established.

These are limitations of the observed result, not newly diagnosed optimizer
bugs. Do not hide them by post-result checkpoint replacement or substituting
the economically successful fractional policy for the bidding policy.
See [the corrected Table 1 report (local-only)][table1-results] and
[configuration comparison (local-only)][configuration-comparison].

### RESULT-06: Learning stability and superiority remain conditional

Vanilla SAC often improved early and degraded later; completing a larger budget
does not make the final weights the selected best policy. The displayed
retrained low SAC is seed 1007 at 250,002 transitions, not its final
2,000,016-transition weights.

BC and analytic control remain strong tracking comparators. Union SAC's
retention improvement is not demonstrated superiority over BC. Native low-level
PPO and SAC development qualification counts were 0/96 and 88/96, and neither
passed the required all-seed BC comparison. Failed gates stay failed.

On the frozen economic primary panel, SAC with credit potential improved the
predeclared full-fraction comparator by about 0.866 MW and $43.44/26h, but this
is not universal dominance: the best observed primary-panel cash script earned
more cash with less relief. These comparisons have different objectives and
do not establish that one controller is best on every metric.
Evidence: [the PDF's explicit trade-off discussion][pid-source].

### RESULT-07: Generalization, selection and uncertainty have limited scope

- Five evaluation seeds are not five independent SAC training fits.
- A smaller evaluation SD is not proof of more stable training.
- A four-checkpoint mean of results is not a deployed action-averaging ensemble
  or a guarantee for a typical seed.
- The reported confirmation intervals condition on fixed dates/checkpoints.
  They are not universal population or deployment guarantees.
- Day 0 is in training-date support for the Table 1 follow-up; this is not a
  fresh calendar holdout. That alone also does not prove exact-cell or
  trajectory-level training exposure.
- Historical peaks and already-opened panels are consumed regression evidence,
  not untouched tests for another round of selection.
- Do not select a new winner using sensitivity results and then call those same
  results an independent confirmation.

### RESULT-08: Metric names must not overstate physical or economic benefits

SPDR is a stress-average proxy, not actual peak reduction. Facility relief is
not identical to PCC relief. Aging fraction is not battery-throughput MWh.
Lower absolute tracking error or wear under a smaller commitment does not prove
equal-service superiority. Positive IT-energy curtailment or throttle below one
does not alone prove that an equivalent amount of job work was deferred.

The new Table 1 supplemental gates contain no minimum tracking-performance or
maximum-capacity service floor. Random can therefore pass physical gates while
tracking very poorly. Less-negative grid cash alone is not task success.

## 6. Historical provenance and reproducibility gaps

| ID | Remaining gap | What is and is not established |
|---|---|---|
| PROV-01 | Original Table 1 producing records are incomplete. | Exact commands, episode manifests, matching complete result CSVs and full checkpoint mappings were not recovered. Source-default reconstruction is not an exact reproduction of all original runs. The archive reports macro SAC RMSE SD 626 while an older README reports 26; do not silently adjudicate without producing evidence. |
| PROV-02 | Original artifact labels and macro training partner are ambiguous. | The preserved low checkpoint directory says `s100` while its stored seed is 200. Its selected 215,000 steps do not prove an incomplete 400,000-step training budget. Original macro trainer source used rule hardware, but the old macro ZIP's actual training partner is unproven. New fingerprinted checkpoint pairing does not repair missing historical provenance. |
| PROV-03 | Corrected original-bidding PPO and paper-conformant LLM rows remain unestablished. | The new Table 1 has seven corrected rows, not nine. No compatible original-bidding PPO was established and no new training/LLM execution was authorized. Do not insert archival numbers or economic PPO into the corrected panel. |
| PROV-04 | Large evidence and exact datasets are not fully available through the public Git release alone. | Local compressed evidence parts exist; the distributed location-map template contains null URLs. Input redistribution permission was not established, so datasets were excluded from new public history. User uploads/permissions and working external locations still require confirmation. Hashes prove identity, not licensing rights. |
| PROV-05 | Independent external fresh-clone reproduction is not completed in the recorded validation. | Local clean-export checks, remote refs/API and release assets were verified; that is not the same as a completed independent external fresh-clone run. Do not advertise the latter as passed. |
| PROV-06 | The latest Table 1 study is separate from the published PID release. | Its result snapshot is local commit `f477cdea`, tag `result/table1-corrected-existing-20261102`; its 105 raw traces remain local, with hashes and episode receipts tracked. It was not pushed or inserted into the earlier evidence bundles. The public release branch/tag must be used instead of assuming default `main` contains every correction or new result. |

Sources: [training provenance ledger (local-only)][training-ledger],
[original Table 1 extraction (local-only)][original-table1],
[checkpoint readiness (local-only)][checkpoint-readiness],
[public release guide][release-guide],
[artifact-location template][artifact-locations], and the
[protected PID-comparison release][public-release].

The source/PDF audit is bounded: verified source defects do not prove that every
unknown historical producing run exercised each defect.

## 7. Recommended repair and validation sequence

This is a proposed follow-up sequence, not a record that these repairs or
sensitivity experiments have already been performed.

1. **Preserve the old evidence.** Keep prior PDFs, checkpoints, raw trajectories,
   failed cases, registrations and tags immutable. Use a new version/branch
   for implementation changes and a new result directory for reevaluation.
2. **Agree on the intended semantics before changing behavior.** In particular,
   specify RegD neutrality, timebase calibration and the battery/PCC boundary.
   Do not choose a definition because it benefits a favored controller.
3. **Repair Box construction and saved-space compatibility separately.** Test
   membership/sampling and deterministic prediction equivalence on finite real
   observations. A metadata-only repair should not be presented as RL training
   progress.
4. **Test signal and electrical invariants directly.** Cover every declared
   neutrality window, clipping, intentional bias and nondefault clocks. Cover
   actual SOC-limited charging/discharging and voltage thresholds at the chosen
   PCC. Verify that metering and accounting remain consistent.
5. **Repair generic metric and schema hazards without rewriting legacy records.**
   Test the first aging tick, last-window failures, absent required metrics,
   random seeds, controller resets, normalization artifacts and warning boundaries.
6. **Run paired frozen-policy sensitivity evaluations first.** Retain the same
   existing checkpoints and baselines; compare declared old/new versions and,
   where useful, isolated versus combined repairs. Match exogenous inputs
   within each version, keep both bias strata and recovery, and retain every
   planned case. Do not replace the original admission rules after observing
   outcomes. The generator repair necessarily changes its emitted signals;
   equal seed labels alone do not prove equal cross-version trajectories.
7. **Inspect changes in qualifications and all key metrics, not just mean relief.**
   Include tracking, paid service, SOC/work/thermal closure, voltage/frequency,
   battery use/aging, actual peaks, cash and cohort denominators. A repair can
   change rankings even if both controllers previously shared the defect.
8. **Separate robustness from retraining.** Retrain only under a separately
   declared protocol if needed. Do not mix frozen-checkpoint robustness tests
   with post-fix tuning or new checkpoint selection.
9. **Update publication claims and release access honestly.** Publish an explicit
   old/new comparison and updated limitations. Carry the Q.5/Q.6 gate clarification,
   complete lawful artifact access, and perform independent reproduction before
   claiming those gaps are closed.

## 8. Findings that must not be relabelled as open SAC bugs

- No faulty Stable-Baselines3 SAC actor/critic update equation was established.
- Original simple-BESS tracking already used actual SOC-limited output.
- The original fast reward used the observed RegD signal; the confirmed
  transition-telemetry mismatch was a separate issue and was corrected in
  `physical_v1`.
- The paper explicitly used `alpha * throttle` and a normalized facility-power
  cost proxy. Replacing these is an objective/measurement change, not proof
  that the old implementation contradicted the paper.
- No tested sixfold update-to-data, replay/timeout corruption, or NaN mechanism
  was established as the explanation for the observed SAC instability.
- Wider training coverage, multiple fits, validation checkpoint selection and
  additional recovery are experimental choices, not all original bug fixes.
- Audited accounting and reproducibility are valuable evidence, but do not
  prove that every physical-model assumption is correct.

**Bottom line:** retain the existing numbers as the recorded outcomes of their
declared versions. The Box metadata defect has no identified producing-path
impact in the checked PID comparison. The effect of repairing inherited RegD
centering and voltage coupling on the reported comparisons remains unresolved.
Neither blanket invalidation nor blanket assurance of harmlessness is supported.

<a id="local-only-supporting-evidence"></a>

## 9. Local-only supporting evidence

The following records are identified for provenance but are **not included in
this public addendum or the frozen PID release**. Their paths are local research
identifiers, not public download links. The reported observations that rely on
them are disclosures from the local audit, not a claim that the supporting
Table 1 package has already been independently reproduced from this release.

All paths in the table are relative to the local research study directory
`outputs/table1_corrected_existing_20261102/`.

| Supporting record | Local relative path |
|---|---|
| Detailed original/corrected source audit | `ORIGINAL_SAC_AUDIT.md` |
| Macro/market/clock audit, including M12-M15 and C01 | `reports/original_macro_audit.json` |
| Physical/work/battery audit, including R1-R4 | `reports/original_physical_audit.json` |
| Training integration and provenance audit | `reports/original_training_audit.json` |
| Evaluation/reporting audit | `reports/original_evaluation_audit.json` |
| Saved-checkpoint observation-space preflight | `reports/model_preflight.json` |
| Corrected seven-row Table 1 results and case summaries | `reports/table1_report.json` |
| Training/evaluation configuration comparison | `reports/training_configuration_comparison.json` |
| Original Table 1 extraction from the private paper archive | `original_table1.json` |
| Checkpoint readiness and exposure inventory | `reports/original_checkpoint_readiness_audit.json` |

Public simulator source links in the register remain usable. References to the
PID comparison's prose point to the approved [comparison PDF][pid-pdf], rather
than to excluded manuscript source. This addendum does not upload the original
paper archive, full manuscripts, raw trajectories, input datasets, model files,
or private Git history.

[audit]: #local-only-supporting-evidence
[macro-ledger]: #local-only-supporting-evidence
[physical-ledger]: #local-only-supporting-evidence
[training-ledger]: #local-only-supporting-evidence
[evaluation-ledger]: #local-only-supporting-evidence
[model-preflight]: #local-only-supporting-evidence
[pid-pdf]: outputs/pid_comparison_extract_20261102/publication/pid_progress_comparison.pdf
[pid-source]: outputs/pid_comparison_extract_20261102/publication/pid_progress_comparison.pdf#page=9
[pid-readme]: outputs/pid_comparison_extract_20261102/README.md
[table1-results]: #local-only-supporting-evidence
[configuration-comparison]: #local-only-supporting-evidence
[original-table1]: #local-only-supporting-evidence
[checkpoint-readiness]: #local-only-supporting-evidence
[release-guide]: https://github.com/HewlettPackard/dc2grid/blob/246d60e49e2478ededba81e39e60c0141fec41fa/REPRODUCIBILITY.md
[gate-erratum]: https://github.com/HewlettPackard/dc2grid/blob/246d60e49e2478ededba81e39e60c0141fec41fa/provenance/metadata_erratum_q5_q6.json
[artifact-locations]: https://github.com/HewlettPackard/dc2grid/blob/246d60e49e2478ededba81e39e60c0141fec41fa/reproducibility/artifact_locations.example.json
[public-release]: https://github.com/HewlettPackard/dc2grid/releases/tag/repro/20261102/pid-comparison
