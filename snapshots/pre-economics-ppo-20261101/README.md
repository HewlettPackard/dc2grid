# Pre-economics/PPO baseline snapshot

This snapshot preserves the latest verified results before any new experiment
code changes. The source/result commit is
`053b52752e9f9e926d12fe3cb65e836b73637081`, already tagged
`results/table2-union-verified-20261101`. The snapshot metadata commit is tagged
`baseline/pre-economics-ppo-20261101`.

The worktree was clean before capture. All 24,074 files in the two latest study
inventories were rehashed using four bounded workers: 38,685,313,944 bytes,
with no missing files, digest mismatches, or files changing during verification.
The [snapshot receipt](snapshot.json) records the source tree and publication
hashes. Git contains the experiment code, protocols, selected evidence, eight
final union checkpoints, and eight union pilot checkpoints.

The existing lean-storage arrangement is unchanged. Bulk replay buffers, raw
traces, and intermediate checkpoints remain at their original locations and
are inventoried, not all copied into Git. **Checksums are not a backup.**
The earlier repository-wide inventory is referenced but its bulk entries were
not rehashed as part of this snapshot. No remote push is implied.

## Scientific boundary

The [verified report](../../outputs/rl_value_conditioning_20261101/reports/REPORT.md)
establishes improved SAC tracking and battery throughput, not a new overall
grid-relief/cash winner. No learned high-level replacement was promoted.
All previous peak and six-date panels, including Final-B, are consumed.
New optimization must not relabel those outcomes as fresh validation.

Further work belongs on a separate branch and in a new registered study.
Frozen simulator semantics, earlier result directories, original manuscript,
and read-only external study materials must remain unchanged.

## Inspecting and comparing

```bash
git show --no-patch baseline/pre-economics-ppo-20261101
git diff --stat baseline/pre-economics-ppo-20261101..HEAD
git diff baseline/pre-economics-ppo-20261101 -- c2g_env baselines evaluation scripts tests
```

The snapshot only adds this documentation and its receipt; it changes no
controller behavior, model weights, simulator semantics, or reported results.
