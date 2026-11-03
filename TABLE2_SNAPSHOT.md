# Updated Table 2: frozen grid-relief snapshot

This branch is dedicated to **the updated main Table 2, both parts, all nine
controller families**, on page 1 of
[pid_progress_comparison.pdf](outputs/pid_comparison_extract_20261102/publication/pid_progress_comparison.pdf).
It is not a new experiment, a retraining campaign, or a claim that the remaining
simulator limitations were repaired.

The branch shares the clean release's common runtime and history. Its manifest
isolates the Table 2 subset; inherited economic/other-table files are not extra
Table 2 candidates. No original private Git history, paper archive, full
manuscript, or input dataset is introduced.

## Snapshot and criteria

- Frozen public scientific snapshot: `246d60e49e2478ededba81e39e60c0141fec41fa`,
  tag `repro/20261102/pid-comparison`.
- Scientific source stages: `repro/20261102/tracking-baseline` and
  `repro/20261102/union-tracking`.
- Macro: the original rule-based macro, including its 80%-RMCP ask.
- Dates: 2023-07-27, 2023-09-06 and 2023-09-07 (start days 207, 248 and 249).
- Evaluation seeds: 100--104.
- Displayed means: bias 0.5, 15 cases per fixed policy.
- Qualification counts: biases 0.0 and 0.5, 30 cases per fixed policy.
- Service: 24 hours; inherited physical recovery: two hours.
- Four-fit families average checkpoints within each case before the displayed
  case SD. They are not action-averaging ensembles.
- The rule-macro physical/recovery gates are retained. No later economic
  paid-capacity floor, credit guard, or full-work override is added to native
  Rule/PID/Bang-bang/retrained-SAC controls.

The nine families instantiate 21 fixed policies and 630 consumed policy/case
combinations across both biases. PID's 26/30 qualification count remains a
diagnostic complete cohort; no failure is removed from the mean.

## Results, checkpoints and replay inventory

- [Table 2 manifest](reproducibility/table2/table2_snapshot.json): source bindings,
  panel, gates, all checkpoint identities, and immutable direct download URLs.
- [Table 2 CSV](reproducibility/table2/table2_cells.csv).
- [Table 2 HTML](reproducibility/table2/table2_numeric.html).
- [Numeric TeX fragment](reproducibility/table2/table2_numeric.tex).
- [All 630 consumed replay selectors](reproducibility/table2/replay_cases.jsonl).

**All 17 checkpoint files are already stored in public Git and are included in
this branch's tree** (16,364,591 bytes in total). The manifest verifies every
file's SHA256 and records the selected step/update, not just a filename.

| Family | Checkpoints | Selection |
|---|---:|---|
| Retrained SAC | 1 | Seed 1007, physical step 250002 |
| BC only | 4 | Seeds 1100--1103, the recorded per-seed selected supervised update |
| Guided SAC | 4 | Selected steps 4096 / 8192 / 4096 / 4096 |
| Union SAC + shaping | 4 | Mandatory step 552960 |
| Union SAC | 4 | Mandatory step 552960 |

Rule, PID, Bang-bang and Analytic are scripted controllers, not missing neural
checkpoints. The analytic/BC/Guided/Union execution restrictions remain those
of their source studies.

The exact shared low SAC is
[best_model.zip](outputs/corrected_sac_8gpu_20261030/seed_1007/low/best_model/best_model.zip):

```text
5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024
```

This is also the low checkpoint used in the corrected Table 1 comparison.
Table 2 does **not** use a learned macro SAC. Table 1's learned-macro row
additionally needs its matching high1007 checkpoint, published with that
separate Table 1 export. Do not substitute an economic fractional policy.

## Verify or rebuild the Table 2 numeric views

Run from the checkout using Python 3.11. These commands do not load a policy,
train, simulate, read input datasets, or open a new test panel.

```bash
python -m reproducibility verify --code-only
python -m reproducibility.table2_snapshot --verify
python -m unittest -v reproducibility.tests.test_table2_snapshot
python -m reproducibility.table2_snapshot --output repro-results/table2
```

The new subset builder reuses the release's hash-checked canonical numeric
renderer. It validates the complete parent numeric reconstruction, then emits
only Table 2(a)/(b), preserving every displayed mean, SD and gate count.
It does not regenerate the frozen PDF's prose/layout.

## Replay an already-consumed case

First supply lawful copies of the five exact CSV inputs using the existing
[input procedure](REPRODUCIBILITY.md). They remain excluded from public Git.
Install the recorded runtime dependencies as described in that guide. No
training data corpus or GPU is needed for frozen policy inference.

```bash
python -m reproducibility inputs import --source-root /path/to/authorized-inputs

# Retrained SAC: original rule macro, original peak, biased signal.
python -m reproducibility tracking \
  --snapshot table2_regression --policy rule__old_sac \
  --day 207 --seed 100 --bias 0.5 \
  --output repro-results/table2-sac-d207-s100-b05

# Union SAC + shaping: select one exact trained fit, not a four-policy ensemble.
python -m reproducibility tracking \
  --snapshot value_peak_regression --policy rule__union_conditioned_1100 \
  --day 207 --seed 100 --bias 0.5 \
  --output repro-results/table2-union1100-d207-s100-b05
```

Use the snapshot/policy/cell tuples in the manifest or replay-selector file for
the other cases. Output directories must be new. A reproduced qualification
failure is a successful reproduction of that failure, not an admitted result.
The [tracking interface](reproducibility/tracking/INTERFACE.txt) documents source,
model, data and scalar validation.

## Limitations and publication boundary

The [remaining-issues disclosure](REMAINING_ISSUES_AND_CONCERNS.md) applies.
RegD centering and voltage coupling remain unresolved in this frozen simulator;
no claim of robustness to future repairs is made here. Historical regression
cases are already consumed, not a new held-out confirmation. The archived
legacy table and later economic/confirmation tables are outside this branch's
Table 2 manifest scope.

The older release inventory and protected result tags are not rewritten.
Its verifier covers the original release bytes; the separate Table 2 verifier
covers these added views and checkpoint bindings. This branch is intended for
a PR into `release/grid-relief-reproducibility-20261102`, not an unrelated-history
merge into `main`.
