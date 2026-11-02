# Grid relief, economics, and PPO follow-up

The completed union-SAC results were committed and independently snapshotted
before this study; see the
[baseline receipt](../../snapshots/pre-economics-ppo-20261101/README.md).

The user approved two separate tracks: benchmark-preserving tracking comparisons
and an explicitly labelled economic-dispatch variant. Neither track changes the
frozen simulator, workload service, settlement, or two-hour physical recovery.
Prior result directories and all external read-only study materials stay intact.

## Pre-learning headroom experiment

The first experiment uses nonlearned, causal residual controllers around the
existing analytic controller. It measures deliberate 0.5/1 MW overdelivery and a
separate battery-inventory diagnostic under the existing full-residual,
zero-ask macro. These are not PPO/SAC training results or replacements for the
original Table 2 regulation benchmark.

The 24 previously consumed Dev-B cells are used openly for development. The
complete frozen 22-policy readiness panel supplies matched reference families.
All new episodes retain raw service, market, and recovery traces. Qualification
requires every physical/recovery gate and the actual 90% capacity-service floor.

Facility relief, PCC relief, signed tracking deviation, and battery reallocation
are reported separately. A battery-induced increase in facility-only relief at
unchanged PCC power cannot establish a grid-relief gain. Constant efficiency
offsets are not represented as time shifting.

```bash
.venv/bin/python -m pytest outputs/rl_grid_cash_ppo_20261101/test_economy.py -q
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/headroom.py prepare
# Commit the registered source and protocol before executing the experiment.
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/headroom.py run
.venv/bin/python outputs/rl_grid_cash_ppo_20261101/headroom_report.py
```

The runner uses eight distinct physical cores and one numerical thread per
process. No GPU is needed for these initial nonlearned probes. PPO/SAC budgets
and model-selection rules require their own subsequent registrations.

## Verified outcome

All 144 headroom episodes qualified, and 36,720 raw scalar/array checks passed.
The disclosed 1 MW economic-dispatch probe improves biased development cash
by USD 269.83 and exceeds the strongest matched frozen facility/PCC relief
families by 0.491/0.581 MW. It does so with approximately 1 MW tracking RMSE,
not improved regulation accuracy. See the
[headroom report](reports/HEADROOM.md) for the full comparison and limitations.
