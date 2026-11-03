# Corrected Table 1 comparison using existing checkpoints

This is a new, separate evaluation requested after the original-code audit.
It does not overwrite the original manuscript, the published PID comparison,
any previous result directory, or the protected public release.

Read [the original-code audit](ORIGINAL_SAC_AUDIT.md) for precise source
locations, counterexamples and the distinction between bugs and intentional
reward/reference changes. In particular, the original paper explicitly used
`alpha * throttle` and a normalized facility-power cost proxy. Changing these
to served-work reward and actual PCC electricity expenditure changes the task
definition/scale; it is not a repair to SAC's actor/critic update equations.

## Fixed scope

Seven existing-model configurations, five cases each:

1. Rule macro + fixed hardware `[1, 0.7, 0.7, 0]`.
2. Rule macro + independently seeded random hardware.
3. Rule macro + bang-bang.
4. Rule macro + rule-based hardware.
5. Rule macro + native PID.
6. Rule macro + selected corrected SAC low.
7. Selected corrected SAC macro + that exact frozen SAC low.

The low checkpoint is seed 1007, selected at 250,002 physical transitions:
`5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024`.
The matching macro checkpoint is seed 1007, selected at 8,004 macro transitions:
`a1b5e5752d68086565c7c4fec545c80b2c702b50f07fe90ab283379898bdb388`.
Its recorded frozen-low identity matches the low checkpoint. Neither final
weights nor fractional/economic supervisors are substituted.

There is **no training, new checkpoint selection, LLM call or new PPO fit**.
The original-bidding PPO and paper-conformant LLM rows remain not rerun.
Their archived numbers may appear only in a clearly separate historical panel.

## Reconstructed default conditions

- Current unchanged `physical_v1`, default scenario and native configuration.
- Start day 0, zero load-correlated RegD bias, seeds 100--104.
- Five-second hardware ticks, 900-second macro windows.
- 24-hour service: 17,280 fast ticks and 96 full macro windows.
- CPU inference, deterministic SAC prediction, fresh controllers per case.
- Eight disjoint worker CPUs, one numerical thread per process.

These are documented source defaults, **not recovered proof of the exact
archival producing runs**. Day 0 is a training start date, not proof that the
exact day/bias/noise cells or resulting trajectories were seen in training.
Five evaluation episodes are not five independently trained SAC policies.

The original nominal loader also advanced source rows too quickly, so the
legacy and timestamp-corrected experiments do not share identical realized
inputs merely because both use a default day label.

## Main metrics and qualification

The six Table 1 columns use service data only:

- Physical fast-tick tracking RMSE in kW.
- Gross regulation revenue in USD.
- Native macro reward averaged over returned macro windows.
- Native low reward averaged over fast ticks.
- Fraction of actual post-step temperatures with either zone **>= 33 C**.
- BESS aging `(service_end - reset) * 10000`, including the first transition.

Means and sample SDs use all five planned cases, with `ddof=1`.
If any case fails to complete service, no full-horizon primary mean is created
from survivors or mixed-duration prefixes. Prefix diagnostics remain available
and are explicitly not ranked against complete cohorts.

Full-horizon means remain visible but diagnostic when supplemental physical
or recovery gates fail. Survival and those stronger gates are separate:

- complete physical service without native termination;
- service-end SOC >= 0.20;
- backlog <= 1% of arrivals;
- thermal-warning fraction <= 0.01;
- work conservation and the inherited two-hour recovery closure.

There is **no later economic maximum-capacity service floor, credit guard,
full-work override or bid-quantity constraint**. Macro commitment and acceptance
are choices/diagnostics in this original-bidding task.

The fixed two-hour continuation is supplemental only: no extra regulation
revenue, nominal full-work/cooling controls, initial-SOC target, and continuing
workload/weather/grid/reference state. Its cash, throughput and closure are
not mixed into the six service metrics.

## Preservation and execution

The baseline is committed at `34e01546`, tagged
`baseline/pre-table1-corrected-existing-20261102`.
The study is isolated on `study/table1-corrected-existing-20261102`.
Register and commit implementation, cases, models and input hashes before
executing any production case.

```bash
.venv/bin/python -m pytest -q outputs/table1_corrected_existing_20261102/test_table1_eval.py \
  outputs/table1_corrected_existing_20261102/test_table1_registration.py
.venv/bin/python outputs/table1_corrected_existing_20261102/prepare_table1.py prepare
# Commit the new registration and its hashed implementation before continuing.
.venv/bin/python outputs/table1_corrected_existing_20261102/prepare_table1.py verify
.venv/bin/python outputs/table1_corrected_existing_20261102/run_table1.py --first-case-only
.venv/bin/python outputs/table1_corrected_existing_20261102/run_table1.py --remaining
```

Existing attempts are not overwritten or automatically retried. Physical
failures are outcomes; implementation exceptions stop execution and retain
partial evidence. Raw service, market and recovery traces remain separate.
Known frozen RegD-neutrality, voltage-interface and workload-proxy limitations
are documented rather than silently fixed during this matched evaluation.
The corrected macro observation Box also has an inconsistent cached upper-bound
sampling flag. This newer metadata defect is separately documented in the audit;
these evaluations use actual finite observations, not observation-space sampling.

## Completed results

The registration was committed at
`f1e15376b242851f24213fafd1a06241e8a82622` before execution.
All **35/35** cases completed 24-hour service and two-hour recovery.
**31/35** passed every supplemental gate. Four learned-macro/SAC cases
(seeds 100, 101, 102 and 104) ended service below the 0.20 SOC reserve;
recovery does not erase those service-time failures.

An independent raw audit passed 8,366,075 scalar checks over 105 trace files,
including metered tracking, reward components, actual settlement duration,
electricity integration, work conservation and recovery closure.

- Rule macro + retrained SAC: **272.6 +/- 32.9 kW** RMSE, versus PID
  **394.7 +/- 230.8 kW**; 30.9% lower mean and lower RMSE in four of five cases.
  Their actual macro contracts match in every window.
- This is a trade-off, not dominance: SAC has **$27.74 less** service regulation
  revenue, **12.6% more** service aging, and **$129.69 less** 26-hour grid cash
  than PID. Cash is revenue minus metered electricity cost, not total profit.
- Learned macro + the same SAC low accepts **0/480** bids, retains the standing
  5 MW commitment, and passes the supplemental gates in only **1/5** cases.
  Its full-cohort metrics remain visible as diagnostic results.
- Recorded training/native configuration hashes match. The Table 2 recovery
  configuration changes only data-directory paths, not plant/market/reward
  scalars. No static configuration mismatch was found; this is not a causal
  diagnosis of the macro policy's behavior.
- No new training, checkpoint selection or LLM calls occurred. Original-bidding
  PPO and LLM rows remain **not rerun**. The economic fractional policies are
  different controllers and were not substituted.

Deliverables:

- [Standalone corrected/archival comparison PDF](publication/table1_corrected_comparison.pdf).
- [Audited seven-row results and all 35 cases](reports/table1_report.json).
- [Full-precision summary CSV](reports/table1_summary.csv).
- [Displayed cell values, precision and source/model identities](publication/table1_values.json).
- [Exact configuration comparison](reports/training_configuration_comparison.json).
- [Observed paired trade-offs and macro bids](reports/observed_tradeoffs.json).
- [Raw trace inventory with hashes](reports/raw_trace_inventory.csv).
- [Actual PDF/CSV and preservation validation](reports/publication_validation.json).

To re-audit and render existing results, **without rerunning simulations**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q outputs/table1_corrected_existing_20261102/test_table1_*.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python outputs/table1_corrected_existing_20261102/analyze_table1.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python outputs/table1_corrected_existing_20261102/render_table1.py
cd outputs/table1_corrected_existing_20261102/publication
pdflatex -interaction=nonstopmode -halt-on-error table1_corrected_comparison.tex
pdflatex -interaction=nonstopmode -halt-on-error table1_corrected_comparison.tex
cd ../../..
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python outputs/table1_corrected_existing_20261102/verify_table1_publication.py
```

The result snapshot keeps episode receipts and trace hashes in Git. The 105
compressed raw traces remain in the local case directories; they are not added
to the earlier published evidence bundles or remote release.

See [the evaluator interface](EVALUATOR_INTERFACE.md). No automatic publication
or manuscript replacement is authorized by this study.
