# Isolated online SAC stability pilot

This pilot does not edit, stop, or replace the original eight-GPU campaign.
Its source is a separate copy of the campaign's archived source, with the
online-continuation runner and scheduler added here. The data link is used
read-only. Its registered protocol and input hashes must not change once
the scheduler starts.

## Design

- Four selected parent checkpoints: seeds 1000, 1001, 1006, 1007 at 1,000,008
  original transitions. These represent different observed failure patterns.
- Three nested continuation streams per parent and three treatments: **36 runs**.
- Sham retains the late critic/target/Adam state. Optimizer reset keeps the late
  critics and initializes fresh critic Adam. Restoration loads the same parent's
  500,004-step critics and targets and initializes fresh critic Adam.
- All arms preserve the late actor, actor optimizer, entropy coefficient and
  entropy optimizer initially. All receive **500 critic-only calibration updates**.
  Sham is therefore a matched-calibration control, not a literal replay of
  the original training schedule.
- Every run then collects **103,680 new online transitions**, inserting them into
  its loaded replay buffer while performing one normal SAC update per transition.
  This is one complete 24-hour episode per worker when no episode ends early.
  Initial replay and update/condition seeds are matched within each parent/stream
  block. Realized actions, trajectories, and subsequent replay contents may differ.
- The reward, physics, action/observation definitions, training conditions, four
  validation cells, and all safety/service gates remain unchanged.
- Evaluation occurs at online step 0 and every 25,920 new transitions. The final
  evaluation/checkpoint is taken **after all final-budget optimizer updates**.
  Mid-run callback snapshots retain the documented one-vector-step replay lag.
- Final-budget outcomes for every registered run are primary. Best-qualified
  models are secondary. All failed jobs and unqualified/incomplete cells remain
  recorded; there are no silent retries or survivor-only summaries.

Raw and normalized tracking errors, all seven reward components, actual
commitments, queue saturation/debt, thermal warnings and terminal SOC are
recorded. Qualification does not itself mean better tracking. Restoring earlier
critics is not a random network reset and does not independently isolate targets.
Fresh Adam includes its startup transient.

The four parent policies were selected after the original degradation was
observed. Three continuations of a parent are **not three independent
from-scratch seeds**. This is an exploratory recovery pilot, not a confirmatory
population claim, an experiment on the onset of instability, or evidence for
changing the paper's results. The three peak test dates are not evaluated here.

## Resource safety

The scheduler waits until an original baseline slot is explicitly terminal
(`completed`, `blocked`, `failed`, or interrupted) **and** its final learner PID
has exited. It then reuses that slot's GPU and disjoint seven-core CPU assignment:
one learner, six CPU environments, one compute thread each. Validation workers
reuse the same cores while training environments are idle. At most eight pilot
learners run concurrently. A slot is also deferred when GPU free memory is below
4 GiB or instantaneous GPU utilization is 90% or more.

Baseline macro training has priority: a completed low-level process does not
release a slot while that baseline seed is proceeding to its macro stage.
Stopping this pilot supervisor terminates only its own children, not the baseline.

## Launch and monitor

Use the original project's Python 3.11 environment, but run from this isolated
`source` directory. Do not use `uv run`, which could synchronize the shared
environment while the baseline is active.

```bash
/workspace/IBM/temp/dc2grid-repos/dc2grid/.venv/bin/python \
  -m scripts.launch_online_stability --register

/workspace/IBM/temp/dc2grid-repos/dc2grid/.venv/bin/python \
  -m scripts.launch_online_stability
```

The second command is foreground; use an approved persistent process manager
to leave it running. Registration refuses overwrites and freezes source hashes,
protocol, checkpoint/replay hashes, randomized job order and resource assignments.
The scheduler refuses to silently relaunch an existing campaign.

Monitor `status.json` here, then `runs/<job-id>/status.json`, per-job logs,
`validation/online_<new-steps>.json`, and `initialization.json`. Initialization
records actor/entropy preservation, treatment hashes, calibration and RNG seeds.
Completion verifies the exact new transition/update budget and replay insertion
position. Model and replay snapshots are recoverable; simulator/RNG continuation
is intentionally fresh, not a bit-exact resume of the parent episode.

## Validation

From `source`, with native thread caps set to one:

```bash
/workspace/IBM/temp/dc2grid-repos/dc2grid/.venv/bin/python \
  -m pytest -q tests/test_online_stability.py
```

Tests cover all three real online continuations, final-update accounting,
actor/entropy invariance, treatment differences, telemetry transparency,
paired condition streams, unchanged input files and resource-release gating.
