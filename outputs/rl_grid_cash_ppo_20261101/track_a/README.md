# Track A: matched native PPO and SAC

This preserves the original 24-hour low-level reward, 24 causal observations,
full-work pump/HVAC/BESS commands, and rule-based macro. Every evaluation adds
the unchanged two-hour physical recovery. This is separate from Track B's
economic-dispatch objective.

Four matched BC initializers are used for each algorithm. Their deterministic
actors and initial physical action distributions match. PPO keeps its Gaussian
samples in pre-tanh coordinates and transforms them only at the actuator
boundary, avoiding unstable inverse-tanh likelihood reconstruction. Deployment
uses the same transform once, with physical commands in the history features.
PPO checkpoints use the ordinary SB3 policy/save/load implementation.

Both arms use 1,024 policy-collected warmup transitions and 256 critic-only
optimizer updates with the actor frozen. PPO calibrates its value function
on two on-policy rollouts; SAC calibrates its Q functions from warmup replay.
The critics, exploration after initialization, entropy regularizers, optimizer
counts, and computation are not claimed to match.

The full budget is **552,960 physical transitions per fit**, four environments
per fit, four seeds per algorithm. The 4,096 and 69,120 checkpoints are retained
but cannot replace the required final checkpoint. The matching quantity is
environment transitions, not gradient updates or wall time. PPO's KL-based
within-rollout early stopping is counted using actual optimizer hooks.

Qualification and comparison use all 24 consumed Dev-B cells. Tracking
non-inferiority means the mean RMSE in each bias stratum is no worse than the
same seed's BC initializer; individual-case regressions are also listed.
An overall improvement additionally needs joint facility relief, PCC relief,
and cash gains against the strongest qualified matched reference families.
Missing model seeds cannot produce a survivor-only family mean.

New calendar-combination conditions are reserved before training, but no new
final outcomes are opened by these commands. They are not an unseen-workload
phase or p90-tail holdout. Both tracks must lock candidates before any future
shared opening.

```bash
.venv/bin/python -m pytest outputs/rl_grid_cash_ppo_20261101/track_a -q
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/track_a/study_low.py prepare
# Commit the source, reservation, and protocol before launching.
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/track_a/study_low.py smoke
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/track_a/study_low.py full
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/track_a/evaluate_low.py baselines
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/track_a/evaluate_low.py full
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/reports/finish_low.py
```

The eight fits use one GPU and five disjoint physical cores each, with numerical
threads capped at one. Evaluation uses the separately reserved CPU group.
All failed runs, incomplete cases, and non-admitted model slots remain visible.

The registered report entry point assumed a BC identifier prefix that differs
from the inherited frozen names. The separate audited collector above matches
family, macro, and initializer metadata instead. It leaves the registered
source, episodes, models, thresholds, and cohorts unchanged and records the
original lookup failure in its output.
