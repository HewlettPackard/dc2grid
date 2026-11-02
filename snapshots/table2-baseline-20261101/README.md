# Verified Table 2 baseline snapshot

This baseline was captured before starting the next controller-improvement study.
It records the current implementation and results, not a new experiment.

Baseline tag: `baseline/table2-verified-20261101`.
Parent commit before capture: `7550013331a865fa45684cb7cd973279312d4678`.

## What Git stores

- Current simulator, controller, training, evaluation, configuration and test code.
- Registered experiment source snapshots, protocols, selection receipts, audit
  records, readable reports and per-case summary metadata from the completed studies.
- The current manuscript, its updated archive, supporting figures and PDF previews.
- The nine checkpoint files used by the published Table 2 comparisons: four BC
  initializers, four selected Guided-SAC policies and the prior retrained SAC model.
- The existing processed datasets and the frozen-source data symlink.
- [The SHA-256 inventory](manifest.json) of 26,802 retained file/link paths,
  including approximately 59.72 GiB of repository data and generated artifacts.

The directory ignore rules remain in force. Selected generated evidence was added
explicitly; **do not force-add the entire outputs directory** in future commits.

## What is retained on disk, not stored in this Git commit

Raw rollout CSV archives, demonstration arrays, replay buffers, intermediate model
checkpoints, and routine logs remain in their original locations. The inventory
records their relative paths, sizes and SHA-256 digests; entries with
`git_payload: false` are not backed up by this commit.

**A checksum is not a backup.** Copy or archive these bulk files separately before
moving the workspace or deleting ignored artifacts. Do not use `git clean -fdx`
without first protecting the bulk evidence.

Caches, virtual environments, Git internals and IDE state are excluded. Symlinks
are recorded without traversal. The frozen-source `data` symlink points to the
original workspace's processed data directory and is preserved as-is. Existing
registered configurations also retain their original absolute paths: if moving
to a different checkout location, create a new derived run configuration rather
than silently editing the historical registrations.

## Validation at snapshot

- Corrected physical/SAC/parallel/grid-relief contract tests: **34 passed**.
- Table 2 controller, finite-task, statistics and formatting tests: **55 passed**.
- Existing CRLF line endings were preserved. The whitespace check passed with
  `core.whitespace=cr-at-eol`; no source-format cleanup was performed.
- A local credential-pattern scan checked 8,603 selected files and 80 text members
  inside archives, reporting no matches. No data was sent to an external service.
  This is a practical screening check, not a guarantee against every secret format.
- Bulk hashing used four bounded workers and checked that files did not change
  between inventory and hashing.

The experiment findings and their limits remain in the
[verified results report](../../outputs/table2_distress_rl_20261031/reports/REPORT.md).
Both final evaluation panels are consumed; neither is an untouched test for a
subsequently tuned controller. No controller behavior, model weights or
experimental results were changed while making this Git snapshot.

## Comparing future work

```bash
git show --no-patch baseline/table2-verified-20261101
git diff --stat baseline/table2-verified-20261101..HEAD
git diff baseline/table2-verified-20261101 -- baselines c2g_env conf evaluation scripts tests
```

Use separate commits and new registered output directories for future experiments.
Keep code changes, protocols, results and selected checkpoint identities together.
This snapshot is local; no push is implied.
