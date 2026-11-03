# Corrected-existing Table 1: main-based reproduction integration

This independent sibling of the Table 2 integration reproduces the
[frozen Table 1 clean export](https://github.com/HewlettPackard/dc2grid/tree/927c527adf19e5240caaf5580579646f4381d290).
That export maps the original local result `f477cdea7a1a7461d5d67fa27832c36d45d82388`
without importing private Git ancestry. This PR targets
`integration/grid-relief-core`, not the orphan release branch or the Table 2 PR.
No frozen result, checkpoint or tag is rewritten.

## Included implementation and source boundary

The unchanged service/recovery evaluator and numeric/PDF checking helpers are
reviewable source files. Their historical relative paths under `outputs/` are
retained for provenance, not to import the full research output tree. A new
[integration source/asset lock](reproducibility/table1/integration.json) binds
these files, the native shared core, and the portable adapter.

The original registrations, all 35 case receipts, result tables, numeric-only
archival Table 1, approved PDF and exact two SAC ZIPs are hash-checked downloads
from the immutable public export. They remain ignored by Git. No input dataset,
raw trajectory, original paper ZIP, full manuscript or private history is added.

Compared with the producer, native source differs only by the shared core's
already-reviewed **Box-construction metadata repair**. New execution protocols
explicitly record this source amendment and path rebasing, have their own IDs,
and retain the producer registration ID. The original registration is never
rewritten. Numeric bounds, physics, policy bytes, observations, rewards and
scientific gates are unchanged. Historical checkpoint metadata is not migrated.
See [simulator semantics](docs/simulator_semantics.md).

Use the supported entrypoints below. Historical preparation functions are
retained as dependencies/provenance, not as supported commands that reconstruct
private Git history or reopen the original manuscript archive.

## Commands

Use Python 3.11 and the original runtime: NumPy 2.4.4, PyTorch 2.6.0,
stable-baselines3 2.7.1, Gymnasium 1.2.3, cloudpickle 3.1.2, pandas 3.1.0,
PyYAML 6.0.3 and Matplotlib 3.11.2. CPU inference needs no GPU or new training.
Numerical thread counts are capped at one. PDF checks use the existing Poppler
`pdftotext` executable.

```bash
python -m reproducibility.assets verify table1 --sources-only
python -m reproducibility.assets fetch table1
# Or use identical existing public evidence, without network:
# python -m reproducibility.assets fetch table1 --source-root /path/to/snapshot

python -m reproducibility.table1_release verify --check-pdf
python -m reproducibility.table1_release tables --output outputs/reproduced-table1
python -m pytest -q tests/test_reproducibility_assets.py \
  reproducibility/tests/test_table1_release.py
```

This recomputes all-five-case means/sample SDs, verifies the **96-cell CSV**
byte-for-byte and checks all **93 nonmissing mean/SD cells** against the preserved
four-page PDF. The numeric-only historical panel remains separate from the seven
corrected rows; embedded manuscript TeX is not required or opened. The approved
PDF is verified, not regenerated.

### Replay

The original main history already contains processed inputs; this PR neither
adds nor downloads them. Verify exact hashes and supply lawful copies if needed.
The original clean-release input inventory is preserved as provenance; its
non-redistribution statement describes that release, not all of main's history.

```bash
python -m reproducibility.assets inputs status
# Only if needed:
# python -m reproducibility.assets inputs import --source-root /authorized/source
python -m reproducibility.table1_release verify --with-inputs

python -m reproducibility.table1_release evaluate \
  --policy rule_macro+sac --seed 100 \
  --output outputs/table1_corrected_existing_20261102/replays/low-sac-100

# Reproduce a recorded supplemental reserve failure without relabeling it.
python -m reproducibility.table1_release evaluate \
  --policy sac_macro+sac --seed 100 \
  --output outputs/table1_corrected_existing_20261102/replays/high-sac-100
```

Allowed policies are `rule_macro+none`, `rule_macro+random`,
`rule_macro+bang_bang`, `rule_macro+rule_based`, `rule_macro+pid`, `rule_macro+sac`
and `sac_macro+sac`; each accepts seeds 100--104. Replaying all 35 cases is the
Cartesian product of this fixed roster and those five seeds. Use a distinct,
new output directory below the study's `replays/` directory for every case.
No day, bias, model, macro restriction, plant or gate override is offered.

Every replay runs an isolated CPU worker using this checkout's native sources.
Source/model/config/input identities are checked before and after the run;
actual imported simulator origins are checked by the preserved evaluator.
Service/recovery accounting, initial/terminal states, completion and qualification
are compared with the recorded case. Counts/booleans are exact; floats use
`rtol=1e-9`, `atol=1e-6` (USD `1e-5`). Checkout paths, gzip headers, wall time and
cross-run state-digest strings are not compared; recovery state preservation is
enforced within each run. Failure evidence remains and mismatches fail explicitly.

## Validation and version locking

The [integration validation record](reproducibility/table1/validation.json) records
**32 focused tests plus four subtests**, every displayed PDF/CSV value, and
**eight full service/recovery replays**: all seven rows at seed 100 plus the
learned-macro passing case at seed 103. The seed-100 reserve failure remains failed.
All **24 entire decompressed CSV payloads** match the original traces byte-for-byte.
This is not a new full 35-case experiment.

Future edits to fingerprinted native or adapter sources deliberately fail
verification. Use the reviewed integration commit for historical reproduction;
do not merely replace expected hashes after changing the scientific model. The
core's separately documented pre-existing missing-HA-driver test failures remain;
these checks do not claim a green full-repository suite.

## Exact experiment, not a new improvement claim

- Default scenario, `physical_v1`, day 0, bias 0, seeds 100--104.
- Primary metrics cover **24-hour service only**. The fixed two-hour physical
  continuation is supplemental, without plant reset or extra regulation payment.
- **35/35** cases complete service; **31/35** pass supplemental gates. Four learned
  macro/SAC cases fail service-end SOC >=0.20. No failed case is dropped.
- No later economic capacity/service floor, credit guard or full-work action
  override is imposed on the seven native controllers.
- Low SAC is the same seed-1007/step-250002 policy used in Table 2:
  SHA256 `5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024`.
- The learned-bidding macro additionally uses high seed 1007, step 8004:
  SHA256 `a1b5e5752d68086565c7c4fec545c80b2c702b50f07fe90ab283379898bdb388`.
  It is not a fractional economic policy.
- Rule macro + SAC tracking is **272.6 +/- 32.9 kW**, versus PID
  **394.7 +/- 230.8 kW**. It has slightly lower revenue/cash and greater modeled
  aging; this is not dominance across objectives. The learned macro accepts
  **0/480** bids and is not an improved bidding result.
- PPO and LLM are **not rerun**. Day 0 overlaps the training calendar; five
  environment seeds are not five independent training fits. Table 1's original
  producing-run identities remain unresolved; the panel is reconstructed from
  source defaults. This is not the peak-date/bias-0.5 Table 2 experiment.

RegD centering and voltage/PCC coupling remain unresolved. Shared defects need
not cancel between controllers, and robustness after physics repair is unmeasured.
The [frozen issues register](https://github.com/HewlettPackard/dc2grid/blob/927c527adf19e5240caaf5580579646f4381d290/REMAINING_ISSUES_AND_CONCERNS.md)
retains the broader limitations.
