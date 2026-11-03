# Corrected Table 1: clean export of the frozen existing-checkpoint study

This branch exports the scientific snapshot from local commit
`f477cdea7a1a7461d5d67fa27832c36d45d82388`, originally tagged
`result/table1-corrected-existing-20261102`.

**It is a clean, equivalent export, not that literal Git commit or its ancestry.**
The original paper ZIP, full manuscripts, input datasets, private history and
raw trace binaries are not uploaded. The public commit therefore has a different
hash. The [file-by-file source mapping](reproducibility/table1/source_export.json)
records exactly what was copied and what was transformed.

## What is included

- **69 byte-preserved producer files**, including the frozen evaluator,
  analysis/rendering source, original registrations, audit reports, execution
  receipts, 35 episode receipts, scalar results, numeric views and comparison PDF.
- One **numeric-only derivative** of the original Table 1 extraction: only its
  embedded original `table_tex` field is removed. All nine archival rows,
  numerical values and original archive/member identities are preserved in
  [original_table1.numeric.json](outputs/table1_corrected_existing_20261102/original_table1.numeric.json).
- The exact matching high-level SAC checkpoint missing from the earlier public
  release. It was retained outside the source Git commit; its bytes match the
  SHA256 pinned by that commit's registration and executed case receipts.
- A new portable verification/reconstruction/replay entrypoint. It calls the
  unchanged scientific evaluator with explicitly path-rebased metadata, not
  patched simulator functions or substituted controllers.

All 61 native runtime Python files already match the producer byte-for-byte in
the clean release base. The known RegD, voltage and Box limitations remain;
this publication does not repair physics, retrain policies or select checkpoints.

## Results and exact scope

- [Four-page Table 1 comparison PDF](outputs/table1_corrected_existing_20261102/publication/table1_corrected_comparison.pdf).
- [Full-precision summary CSV](outputs/table1_corrected_existing_20261102/reports/table1_summary.csv).
- [All 35 audited case summaries](outputs/table1_corrected_existing_20261102/reports/table1_report.json).
- [All displayed numeric cells](outputs/table1_corrected_existing_20261102/publication/table1_cells.csv).
- [Original-code audit](outputs/table1_corrected_existing_20261102/ORIGINAL_SAC_AUDIT.md).
- [Public export validation](reproducibility/table1/validation.json): 31 focused
  tests plus four subtests, exact numeric/PDF checks and all eight portable replays.
- [Raw trace inventory](outputs/table1_corrected_existing_20261102/reports/raw_trace_inventory.csv);
  the 105 compressed traces remain local, not bundled in Git.

The corrected panel has seven rows, default scenario, `physical_v1`, start day 0,
bias 0, and seeds 100--104. All 35 cases complete 24-hour service and two-hour
recovery; 31 pass the supplemental gates. Four learned-macro/SAC cases fail the
service-end SOC reserve. Those failures stay in the complete-cohort means.
PPO and LLM are **not rerun**; their archived values appear only in a separate
historical panel.

The selected low SAC has 272.6 +/- 32.9 kW tracking RMSE versus PID's
394.7 +/- 230.8 kW, but slightly lower revenue/cash and greater modeled aging.
The learned bidding macro accepts 0/480 bids and is not an improved bidding
result. This panel is not the peak-date, bias-0.5 Table 2 experiment, and day 0
is not a fresh calendar holdout.

## Checkpoints shared with, or additional to, Table 2

| Role | Checkpoint | Selection | SHA256 |
|---|---|---|---|
| Low SAC, shared with Table 2 | [best_model.zip](outputs/corrected_sac_8gpu_20261030/seed_1007/low/best_model/best_model.zip) | Seed 1007, step 250002 | `5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024` |
| Additional original-bidding macro SAC | [best_model.zip](outputs/corrected_sac_8gpu_20261030/seed_1007/high/best_model/best_model.zip) | Seed 1007, step 8004 | `a1b5e5752d68086565c7c4fec545c80b2c702b50f07fe90ab283379898bdb388` |

The high checkpoint records the exact low checkpoint's fingerprint. The low
file is 3,227,384 bytes; the high file is 3,212,503 bytes. Neither is replaced
by final training weights or by a fractional economic policy.

The dedicated [Table 2 branch](https://github.com/HewlettPackard/dc2grid/tree/results/table2-pid-comparison-20261102)
provides the nine-family Table 2 manifest and all 17 relevant checkpoint links.
Only its retrained low SAC is shared with these Table 1 SAC rows; the high
checkpoint above is additionally required here.

## Public entrypoints

Use Python 3.11 and the recorded runtime dependencies from
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). No GPU or new training is required.

```bash
# Original release bytes, then the new Table 1 source/model/results export.
python -m reproducibility verify --code-only
python -m reproducibility.table1_release verify

# Also check all 93 nonmissing mean/SD cells against the actual PDF.
# Requires the existing Poppler pdftotext tool, not input datasets.
python -m reproducibility.table1_release verify --check-pdf

# Recompute the 96-cell CSV and numeric-only TeX/JSON from all five cases/row.
python -m reproducibility.table1_release tables --output repro-results/table1
python -m unittest -v reproducibility.tests.test_table1_release
```

The numeric command uses the frozen aggregation/display helpers and verifies
that the regenerated CSV exactly matches the source snapshot. It needs no
original paper archive, embedded table TeX, raw trajectories or private Git
objects. The approved PDF is preserved byte-for-byte; rendering numeric tables
does not claim to regenerate that PDF's layout or bytes.

### Replay an existing case

Supply lawful copies of the five exact CSV inputs using the existing importer.
They remain ignored by Git and are not part of this publication.

```bash
python -m reproducibility inputs import --source-root /path/to/authorized-inputs
python -m reproducibility.table1_release verify --with-inputs

python -m reproducibility.table1_release evaluate \
  --policy rule_macro+sac --seed 100 \
  --output outputs/table1_corrected_existing_20261102/replays/low_sac_100

# Reproduce a recorded reserve-gate failure, not a survivor-only replacement.
python -m reproducibility.table1_release evaluate \
  --policy sac_macro+sac --seed 100 \
  --output outputs/table1_corrected_existing_20261102/replays/macro_sac_100
```

Only the seven original row IDs and seeds 100--104 are accepted. Day, bias,
physics, models, action restrictions and gates cannot be overridden.
Outputs must be new directories under the study's `replays/` directory, because
the unchanged evaluator constrains writes to its own study root.

Each replay launches an isolated `-I -B` CPU worker from the public checkout.
Only the configuration, model and recovery-source filesystem locators are
rebased; all scientific fields and file hashes are unchanged. The derived
execution registration records its original producer registration ID. Its
own ID can differ across checkout locations because it includes resolved paths.
No original-path fallback or private-commit ancestry is used.

Outputs include `runtime/execution_protocol.json`, `runtime/worker.log`,
`episode/episode.json`, the new raw traces, and `replay_receipt.json`.
On a worker/comparison failure, evidence and an explicit failure receipt remain;
the adapter does not retry or select another model.

The adapter compares primary metrics, all service/recovery accounting, initial
and terminal states, completion and qualification against the exact recorded
case. Booleans, counts and missing values match exactly. Floating tolerance is
`rtol=1e-9`, `atol=1e-6` (USD fields `1e-5`). Checkout paths, host metadata,
cross-run state-digest strings, gzip headers and elapsed time are not compared;
the state-preservation identity is enforced within each recovery extension.
A verified replay may correctly have `qualified_supplemental=false`.

## Historical producer records versus the public adapter

The retained producer scripts, tests and receipts describe the original local
execution. Some reference private Git ancestry, the excluded original paper
record, or then-local paths. **Do not run the old prepare/launch/main-verifier
workflow as the public replay entrypoint.** Use the commands above.

The original 38 focused tests and raw/PDF audits belong to that producing
snapshot. The public adapter has its own focused tests and replay validation.
Historical statements such as "no remote publication performed" describe the
original run, not the availability of this later clean export.

One concrete nonportable producer test is
`test_native_default_reset_contract_and_inherited_recovery_configuration`:
it reads the intentionally excluded private parent
`outputs/physical_recovery_eval_20261031/heldout_registration.json`.
That historical test is retained unchanged, not claimed to pass in this export.
The public test suite checks the same recovery/panel contract against the
published, byte-identical Table 1 registration instead. Actual native reset,
backend and initial-state behavior are checked in the complete portable replays;
no fake or partial replacement registration is installed.

The [portable replay receipt](reproducibility/table1/replay_validation.json)
covers all seven rows at seed 100, plus the learned macro's qualifying seed 103.
All eight replays match their original outcomes, including the seed-100 macro
reserve failure. The [full-trace comparison](reproducibility/table1/full_trace_validation.json)
also verifies that all 24 corresponding service, macro and recovery CSV payloads
are byte-identical after gzip decompression. Raw validation traces remain local,
not part of Git. These are consumed-case reproduction checks, not new held-out
performance evidence.

The original registration ID remains
`ed5313c9b6566c816769176f0a99eac2bd9566af1f402e4cfd087437005a1971`.
No original private tag is assigned to a different public commit. The source
mapping, selected checkpoint hashes and frozen results provide the correspondence.
