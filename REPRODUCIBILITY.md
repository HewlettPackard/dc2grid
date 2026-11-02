# Frozen grid-relief comparison: code, criteria and evidence

This release makes the numerical results in the
[nine-page comparison PDF](outputs/pid_comparison_extract_20261102/publication/pid_progress_comparison.pdf)
traceable to the corresponding source, controller, checkpoint, conditions and
evaluation rules. It does not retrain policies or replace a failed experiment.
The PDF SHA-256 is
`2c31e19948811377bbf68594698573c1aac9498c97f499452861c9330655a595`.

## What is and is not distributed

| Resource | Availability |
|---|---|
| Approved code, configurations, selected checkpoints and scalar results | In this Git release |
| Per-table/per-row scientific provenance | [Row map](provenance/table_row_provenance.json) and [controller protocols](provenance/controller_protocol_provenance.json) |
| Original paper archive, full manuscript, private reviewer notes | Excluded |
| Processed energy, weather and workload input datasets | Excluded; exact hashes and explicit local import are provided |
| Original raw trajectories and required training evidence | Prepared as local compressed parts; not uploaded to GitHub |
| OneDrive download URLs | Pending the author's upload and permission decisions; no working URL is implied |

The original repository/history is untouched. The public source snapshots form
a new **orphan** chain, so even older full papers in the original upstream
ancestry are not inherited. Source commits in the
[snapshot map](reproducibility/snapshot_map.json) are provenance identifiers,
not parents of the public commits. Scientific source and checkpoint bytes are
preserved; clean-export documentation and replay adapters are explicitly new.

No new blanket project, model or dataset license is granted. Input-source terms
and attribution remain separate from software publication. A checksum is not
a redistribution permission.

## Separate tracking and economic tasks

- **Tracking:** [the analytic/BC/Guided study](outputs/table2_distress_rl_20261031)
  and [the Union study](outputs/rl_value_conditioning_20261101) retain their
  original hardware rewards, action restrictions and selection rules.
- **Economics:** [absolute-MW allocation](outputs/rl_grid_cash_ppo_20261101/track_b)
  and [fractional allocation](outputs/rl_grid_cash_ppo_20261101/track_b_fraction)
  retain their economic objective, common analytic executor and credit/service
  constraints. This is not learned market bidding or the original low-level SAC.
- Both reuse their recorded physical simulator. Do not fork the plant model
  merely to separate the objectives, or silently apply one study's gates to another.

The internal identifier `old_sac` means the selected **corrected/retrained**
SAC checkpoint: seed 1007, step 250002. It is not the archival SAC reported at
0.21 MW. BC is supervised imitation, not online RL. A four-checkpoint mean in a
table is not a deployed action-averaging ensemble.

## Table and snapshot index

All 12 numbered tables, 13 tabular parts and 118 data rows have explicit source
bindings. The numeric views contain 1,026 entries, including SDs and pass counts.

| PDF content | Source/result stages |
|---|---|
| Table 2: corrected rule-macro comparison | `tracking-baseline`, `union-tracking` |
| Q.1: original reported comparison | Archival numeric record; unknown execution metadata remains unknown |
| Q.2-Q.3: PID semantics reproduction | `pid-explanation`, with the separately recorded probe/source identity |
| Q.4-Q.5: tracking replication and macro-only comparison | `union-tracking` |
| Q.6: historical economic progression | Tracking references plus `absolute-economics`, `fractional-economics`, `economic-regression`, and later fixed scripts |
| Q.7-Q.10: new-calendar/new-noise comparison | `frozen-confirmation`, followed by `pid-references` |
| Q.11: PID resource/reserve diagnostics | `pid-references` |

Tag names have the prefix `repro/20261102/`. The source chain additionally
includes `confirmation-protocol`, `confirmation-amendment`, and `pid-protocol`.
The complete portable release is tagged `repro/20261102/pid-comparison`.

Result-packaging commits are not automatically training/execution commits.
Where the historical execution commit was not recorded, source-content
fingerprints and checkpoint hashes are retained and the commit stays null.
Only the narrow archival PID relief mean/SD have an independent reproduction;
the other archival mean/SD pairs remain reported-only.

### Criteria and aggregation must stay explicit

- Means, SDs and qualification counts can have different scopes. Some displayed
  means use bias 0.5 while their pass counts include both biases.
- Historical rule-macro tracking gates are not the newer economic admission test.
- **Q.5/Q.6 clarification:** the displayed historical full-residual tracking
  rows already enforce their recorded 90% capacity-service floor. The PDF's
  abbreviated gate wording must not be read as removing that floor. See the
  [metadata erratum](provenance/metadata_erratum_q5_q6.json); no number or gate
  was changed.
- New PID means retain all complete cases, including reserve failures. Closing
  SOC recovery at 26 hours does not erase a failed 24-hour service-end reserve.
- Extra economic relief can trade against tracking accuracy. Cash is metered
  simulator cash, not business profit.
- Reproducing PID's earlier/corrected power balance is not a one-fix-at-a-time
  causal attribution. Same date/seed labels do not imply identical inputs
  across legacy and corrected modes.

## 1. Check public bytes and rebuild numeric tables

Use Python 3.11. The numeric table tools need only the standard library:

```bash
git clone --single-branch --branch release/grid-relief-reproducibility-20261102 \
  https://github.com/HewlettPackard/dc2grid.git dc2grid-repro
cd dc2grid-repro
python3.11 -m reproducibility verify --code-only
python3.11 -m reproducibility tables --output repro-results/tables
```

The new renderer produces CSV, HTML and a numeric TeX fragment, checking all
118 rows and 1,026 entries against preserved reports and scalar case records.
It does not read the original archive, original assembly blob, manuscript
source, input datasets, model weights or raw trajectories.

The approved PDF is preserved as the frozen view. Numeric reconstruction does
**not** claim to regenerate that document's prose, layout or PDF bytes.

## 2. Supply lawful, exact input data

The five required processed CSVs and hashes are listed in the
[input-data manifest](reproducibility/input_data_manifest.json). The release
does not infer redistribution rights from an existing public copy, and does not
silently fetch a mirror or substitute synthetic inputs.

If you already hold authorized copies in the declared directory structure:

```bash
python3.11 -m reproducibility inputs status
python3.11 -m reproducibility inputs import --source-root /path/to/authorized-copy
python3.11 -m reproducibility verify
```

The source root contains `data/processed/...` as declared in the manifest.
Missing or mismatched inputs fail explicitly. Existing different files are
never overwritten. Imported inputs and replay outputs are ignored by Git.

Source credit belongs to the New York Independent System Operator, NOAA NCEI,
and the Alibaba Cluster Trace Program (2023 GPU sharing, 2025 DLRM and 2026
GenTD26). The manifest links the identified providers and records unresolved
acquisition/version metadata honestly rather than inventing a license or date.
The recorded 2024 weather is calendar-matched to 2023 load, not contemporaneous
2023 weather. Consult each provider's terms before acquiring or sharing data.

## 3. Replay frozen policies, not old private registration workflows

Create a separate Python 3.11 environment and install the
[recorded runtime requirements](reproducibility/economic/requirements-runtime.txt).
CPU inference is sufficient. CPU Torch 2.6.0 wheels are accepted; the recorded
validation environment used Torch 2.6.0+cu124 with inference on CPU. Exact
observed package versions are in
[validation_runtime.json](reproducibility/validation_runtime.json).

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r reproducibility/economic/requirements-runtime.txt

python -m reproducibility tracking \
  --snapshot table2_regression --policy rule__old_sac \
  --day 207 --seed 100 --bias 0.5 --output replay_checks/retrained-sac

python -m reproducibility economic \
  --case-id 'confirmation/model__potential_sac_1200:calendar:d7_s207829280_b0p5' \
  --output replay_checks/economic-sac-phi

python -m reproducibility pid-semantics \
  --mode legacy_v0 --day 207 --seed 100 \
  --output replay_checks/pid-legacy-pair
```

Only already-consumed policy/cell slots are accepted. There is no training,
new-panel opening, gain tuning, retry-until-pass or survivor-only replacement.
Each run uses a new output directory and checks source, model and input bytes
before and after execution. Each independent replay has an isolated process.

Tracking/economic cases retain 24-hour service plus two-hour continuing
recovery. The PID semantics diagnostic is instead an original 24-hour
PID/reference pair, without invented recovery or newer gates. A faithful
replay of a recorded physical failure is a successful reproduction of that
failure, not a newly qualified policy.

See the [tracking interface](reproducibility/tracking/INTERFACE.txt),
[economic interface](reproducibility/economic/INTEGRATION.md), and
[PID diagnostic interface](reproducibility/tracking/PID_INTERFACE.txt).
Historical prepare/register/manuscript scripts are preserved where in scope,
but their private-origin workflows are not the portable release entrypoints.
Training source/configurations and outcomes are retained; this release does
not claim a new training campaign or bitwise reproduction of retraining.

## 4. Original evidence and pending OneDrive locations

Large evidence is prepared as independently compressed `.tar.gz` parts, each
strictly below 1,900,000,000 bytes. It is not in Git and was not uploaded to
GitHub. The [artifact manifest](reproducibility/artifact_manifest.json) binds
each object and restored path by SHA-256; OneDrive URLs remain explicitly null.

```bash
python -m reproducibility artifacts plan --tier audit
python -m reproducibility artifacts restore --tier audit \
  --local-bundles /path/to/prepared-parts --cache .repro-cache
python -m reproducibility artifacts verify --tier audit
```

Other tiers are `tables`, `reevaluate` and `train`; use repeated `--tier`
arguments to combine them. Numeric table rebuilding and new frozen replay do
not require the original bulk raw archive. The input-data tier is not a
permission to redistribute the provider's CSVs.

After the author uploads permitted parts to OneDrive, copy the example
location map to the ignored `artifact_locations.json` and fill direct HTTPS
download URLs. Pass `--locations artifact_locations.json` when restoring.
The immutable checksums do not change when storage locations are added.
Until then a download attempt fails with an explicit pending-upload message.

The archive preserves labelled interrupted prefixes, including empty files;
it does not repair or relabel them as complete trajectories. Private session
trees, conversations, reviewer notes and original manuscript archives are not
part of the local numeric evidence package.
