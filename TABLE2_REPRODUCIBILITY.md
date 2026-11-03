# Updated Table 2: main-based reproduction integration

This is the focused implementation follow-up to the
[immutable Table 2 export](https://github.com/HewlettPackard/dc2grid/tree/82acb0072804dd27d11890bcbf9ebdc3d0a8cb7e).
It descends from the shared-core integration, not the orphan publication history.
Its PR targets `integration/grid-relief-core`; the separate Table 1 PR is a sibling,
not a prerequisite. Nothing here changes the frozen result branches or tags.

## What this integration adds

- The Table 2 inference/evaluation dependencies, retaining their original relative
  paths for traceability. The small historical study modules under `outputs/` are
  source code, not a bulk import of the output tree.
- A closed replay interface for the nine Table 2 families: Rule, PID, Bang-bang,
  retrained SAC, Analytic, BC, Guided SAC, Union SAC + shaping, and Union SAC.
- Aggregation of the original scalar case records and exact reconstruction of
  Table 2(a)/(b): **18 displayed rows, 234 numeric entries**.
- A [file-by-file source/asset lock](reproducibility/table2/integration.json).
  Checkpoints, JSON result evidence and the approved comparison PDF are fetched
  from the exact public commit, checked by SHA256 and size, and remain Git-ignored.
  No new checkpoint binary, dataset, manuscript, training corpus or raw trajectory
  is introduced by this PR.

The shared native `c2g_env` and `baselines` implementations are used; a nested
second simulator is not installed. Compared with the frozen producer, the shared
core has its already-reviewed **Box-construction metadata repair**. Three study
bootstrap locators now select this native core. The source lock and replay
receipts disclose these differences. Physics, numeric observation bounds,
checkpoint bytes, actions, rewards, macro policy and scientific gates are not
redefined. See [the semantics guide](docs/simulator_semantics.md).

Use the supported entrypoints below. Historical preparation/training functions
remain in the imported modules as dependencies/provenance; this PR does not
reintroduce their original private-history orchestration as a supported command.

## Setup and evidence

Run from this checkout with Python 3.11. The frozen model runtime is NumPy 2.4.4,
PyTorch 2.6.0 (CPU or CUDA build), stable-baselines3 2.7.1, Gymnasium 1.2.3 and
cloudpickle 3.1.2. The recorded analysis environment also used pandas 3.1.0,
PyYAML 6.0.3 and Matplotlib 3.11.2. Replay checks the model-runtime versions and
runs on CPU with numerical thread counts capped at one.

```bash
# Inspect the source binding before any network fetch.
python -m reproducibility.assets verify table2 --sources-only
python -m reproducibility.assets fetch table2
python -m reproducibility.assets verify table2

# Alternatively import identical evidence from an existing public snapshot checkout:
# python -m reproducibility.assets fetch table2 --source-root /path/to/snapshot

python -m reproducibility.table2_snapshot --verify
python -m reproducibility.table2_snapshot --output outputs/reproduced-table2
python -m pytest -q tests/test_reproducibility_assets.py \
  reproducibility/tests/test_table2_integration.py
```

Numeric reconstruction needs no input CSVs, simulator rollout or manuscript
archive. It recalculates the nine complete-cohort aggregates from scalar case
records, verifies their means/SDs and gates, then checks the regenerated CSV
byte-for-byte against the published Table 2 CSV. The generated JSON is a new,
explicitly labeled integration receipt, not a replacement producer registration.
The approved PDF is preserved, not re-typeset.

Downloads are explicit. Existing different files are never overwritten; missing
or changed evidence fails rather than falling back to another model. The same
fetcher/input helpers are shared by both sibling PRs in a common ancestor commit,
so merging either table does not create a conflicting duplicate implementation.

## Replay a consumed case

The original main history already contains processed inputs. These PRs do not
add, replace or automatically download them. Their presence does not establish
redistribution rights: verify the exact five CSV hashes and obtain lawful copies
if needed. The input inventory is preserved from the clean publication, whose
statement that inputs were not distributed refers to that publication.

```bash
python -m reproducibility.assets inputs status
# Only if needed:
# python -m reproducibility.assets inputs import --source-root /authorized/source

python -m reproducibility.tracking.tracking_release evaluate \
  --root . --snapshot table2_regression --policy rule__old_sac \
  --day 207 --seed 100 --bias 0.5 \
  --output outputs/table2-replays/retrained-sac-d207-s100-b05

python -m reproducibility.tracking.tracking_release evaluate \
  --root . --snapshot value_peak_regression --policy rule__union_conditioned_1100 \
  --day 207 --seed 100 --bias 0.5 \
  --output outputs/table2-replays/union-shaped1100-d207-s100-b05
```

The reconstructed `replay_cases.jsonl` enumerates **all 630 allowed selectors**:
21 fixed policies, days 207/248/249, seeds 100--104, and biases 0/0.5. Use its exact
snapshot/policy/day/seed/bias tuples for other cases; no checkpoint override,
development panel, economic policy, full-residual macro or new calendar is exposed.
Every output directory must be new. A worker failure or metric mismatch preserves
an explicit failure record and returns a nonzero exit; it never silently retries.

Replay compares service/recovery metrics and exact qualification outcomes to the
recorded case. Floating tolerances are `rtol=1e-9`, `atol=1e-6`, or `1e-5` for USD.
Model/source/data hashes and discrete values are exact. Gzip headers, wall time
and checkout-specific paths are not claimed to be identical. Receipts record the
actual module origins and source-binding changes.

## Validation and version locking

The [integration validation record](reproducibility/table2/validation.json) records
**31 focused tests**, exact verification of all **234 numbers in their actual
PDF rows**, and **11 full service/recovery replays**. These cover every family,
the known PID reserve failure and a zero-bias SAC case. All **33 entire decompressed
CSV payloads** match the original traces byte-for-byte, not only their summaries.
These are representative checks, not a new simulation of all 630 selectors.

Reproduction is version-locked. Future edits to fingerprinted native or adapter
sources deliberately fail verification. Use the reviewed integration commit for
historical reproduction; do not silently loosen the checks after changing physics.
The core's separately documented pre-existing missing-HA-driver test failures are
not repaired or represented as a green full-repository suite here.

## Scientific scope and caveats

- Means use bias 0.5 and 15 calendar/noise cases. Qualification counts use both
  biases and 30 cases per fixed checkpoint.
- Four-fit families average checkpoints within each case before computing the
  case SD. They are **not** action-averaging ensembles or fresh independent
  training repetitions.
- Rule macro retains its 80%-RMCP ask. All cases retain the original 24-hour
  service plus inherited two-hour physical recovery.
- Native Rule/PID/Bang-bang/retrained SAC keep their four controls. Analytic,
  BC, Guided and Union retain their full-work execution restriction. No later
  economic 90% paid-capacity floor or credit guard is added to these rule-macro rows.
- PID's **26/30** qualification remains in the displayed complete-cohort result;
  reproducing its reserve failure is a successful reproduction, not an admission.
- All 17 checkpoints remain the exact selected/mandatory snapshots. Retrained
  low SAC is seed 1007, step 250002, SHA256
  `5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024`.
  Table 1 shares this low policy; no learned macro SAC is used in Table 2.
- RegD centering and voltage/PCC coupling remain unresolved. The constructor
  repair does not rewrite historical ZIP metadata. Ranking robustness after
  future physics repairs is unmeasured; historical cells are already consumed.

The full disclosures remain in the
[frozen remaining-issues register](https://github.com/HewlettPackard/dc2grid/blob/900f42f452fda930f37936c9aa494511d21629ae/REMAINING_ISSUES_AND_CONCERNS.md).
Economic ablations and confirmation tables are separate work, not new Table 2
results in this PR. No retraining or new scientific selection occurs here.
