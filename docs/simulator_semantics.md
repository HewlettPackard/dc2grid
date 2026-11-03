# Versioned simulator semantics and checkpoint compatibility

The shared runtime supports two explicit contracts:

| Mode | Purpose |
|---|---|
| `physical_v1` | Current default for new experiments: physically metered response/work accounting, timestamp-based replay and causal market interfaces. |
| `legacy_v0` | Historical formulas and observation/replay behavior retained for explicitly labelled legacy evaluation. |

This integration carries the shared core from public snapshot
`246d60e49e2478ededba81e39e60c0141fec41fa`
(`repro/20261102/pid-comparison`). It does not import the large experiment-output
tree, checkpoints, private research history, new datasets or manuscripts.
Existing files already present on public `main` are not removed merely because
the clean reproducibility export omitted them.

The existing historical result tables in the main README are preserved as
reported. They are not relabelled as results of this integration. Exact producing
records for every archival number have not been recovered.

## Choosing a mode

The environment defaults are in [c2g_env/config.yaml](../c2g_env/config.yaml).
Hydra training defaults are in [conf/config.yaml](../conf/config.yaml).
The change of default to `physical_v1` is intentional and should be considered
when reviewing or migrating an existing experiment.

```python
from c2g_env import C2GFastEnv, C2GMacroEnv

physical = C2GFastEnv(
    simulation_mode="physical_v1",
    start_day=207,
    regd_load_bias=0.5,
)
legacy = C2GFastEnv(simulation_mode="legacy_v0")
macro = C2GMacroEnv(simulation_mode="physical_v1", start_day=0)
```

For the shared benchmark CLI:

```bash
# New physical accounting with a classical controller.
python evaluation/run_benchmark.py --agents rule_macro+pid \
  --scenarios default --n_episodes 1 --seed 100 \
  --simulation-mode physical_v1 --start-day 0 --regd-load-bias 0.0

# Explicitly evaluate a bundled historical, unmarked SAC checkpoint.
python evaluation/run_benchmark.py --agents rule_macro+sac \
  --scenarios default --n_episodes 1 --seed 100 \
  --simulation-mode legacy_v0 \
  --hw-model-dir trained_models/BestCheckpoints/sac_default_s100
```

The grid-relief and dedicated SAC-high/SAC-low CLIs also expose
`--simulation-mode` and `--allow-semantic-mismatch`; inspect their `--help`
for their own case/model arguments. For Hydra SAC training, use
`environment.simulation_mode=legacy_v0` only when intentionally training the
legacy task.

## What physical_v1 changes

### Power, work and tracking

- Flexible server idle draw remains electrical/thermal load, not deferrable work.
- Dynamic work arrivals, actual service, backlog and electrical draw are
  accounted consistently. Reducing a throttle command does not earn flex
  response credit if actual service/power did not decrease.
- Thermal heat and electrical IT draw use consistent physical quantities.
- A continuing nominal thermal/electrical reference uses throttle 1, pump/HVAC
  0.7 and zero battery dispatch. Actual response is the difference between
  reference power and battery-adjusted grid import.
- Tracking uses actual SOC-limited battery output. Facility relief and PCC
  relief remain different quantities.

The built-in nameplate is not increased by this change: the default simple BESS
remains 15 MWh / 5 MW, and the facility remains the configured 250 MW IT model.
Work is still an energy-equivalent proxy, not measured job/FLOP completion.

### Rewards and monetary accounting

The old throughput term `alpha * throttle` is replaced by served dynamic work
normalized by dynamic service capacity. The reported `throughput_ratio` is a
different quantity: served / (served + remaining backlog), with 1 for an empty
queue. Do not equate its denominator with the reward denominator.

Macro electricity expenditure integrates actual signed grid-import MW times
the contemporaneous LMP and executed duration. It does not bill a normalized
power observation or use the product of independent window averages.
Early-ended windows use their actual settlement duration.

The original paper explicitly specified the throttle reward and normalized
facility-power cost proxy. Replacing them changes the objective and reward
scale; these are not repairs to SAC's actor/critic update equations or proof
that the old code disagreed with the paper. Reward totals across modes are not
interchangeable, even where outer coefficients are unchanged.

### Time, observations and randomness

- Calendar replay uses source timestamps rather than advancing a five-minute
  row every five seconds.
- Calendar start and deliberate RegD load bias apply before the reset
  observation, including workload phase.
- The displayed market offer is the offer used for clearing, and the hardware
  policy sees the current commitment before acting.
- Exogenous random streams are separated from other market/grid calls.
- Physical macro RMSE aggregates fast-tick squared errors, not RMS of
  macro-window mean errors.

These are defined changes to the task. Matching an old/new seed or day label
alone does not establish identical cross-version input trajectories.

## Checkpoints and staged SAC

The shared loader checks checkpoint semantics. Unmarked historical checkpoints
are treated as `legacy_v0` and are rejected by default in a physical-mode run.
This is an intentional failure, not a reason to silently substitute a rule
controller or a different checkpoint.

`--allow-semantic-mismatch` is an explicit frozen-policy diagnostic override.
It does not turn a legacy checkpoint into a correctly trained physical policy.
Record that override when it is used; do not mix the result into a matched
retrained-policy comparison without its qualification.

The SAC macro trainer requires the actual selected frozen hardware checkpoint,
or an explicitly selected classical-inner baseline:

```bash
python -m baselines.train_hierarchical_sac algo=sac \
  +hrl.inner_checkpoint=/path/to/selected-physical-low.zip

python -m baselines.train_hierarchical_sac algo=sac \
  +hrl.inner_controller=rule_based
```

The corrected SAC entrypoints record source/context and the actual model seed;
hierarchical training records the inner checkpoint fingerprint. Device and
plant-profile options are forwarded rather than discarded. Dedicated macro
metrics use the macro info schema and fast-step weighting.

The resource-bounded campaign is defined by
[corrected_sac.yaml](../conf/retraining/corrected_sac.yaml) and
[retrain_corrected_sac.py](../scripts/retrain_corrected_sac.py). It uses explicit
train/validation/held-out condition sets, capped threads, process/CPU allocation,
and checkpoint selection only after complete validation suites. The supplied
GPU/CPU allocation is a campaign configuration, not portable hardware discovery.
Review it before launching. The launcher starts training jobs; `--help` is the
appropriate way to inspect its interface without launching a campaign.

NumPy 2.0 or later is required by the imported core. Published checkpoint replay
also depends on its recorded Torch/SB3/Gymnasium/runtime versions, not just the
broad lower bounds in the project dependency list.

## Focused validation and existing test-suite boundaries

From an environment with the declared dependencies installed, the focused
shared-core checks are:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q -rs \
  tests/test_physical_contract.py tests/test_sac_contracts.py \
  tests/test_parallel_sac.py tests/test_baselines.py \
  tests/test_critical_bug_fixes.py tests/test_grid_relief.py tests/test_gym_api.py
```

Some optional tests require PySAM or private/unshipped helper modules and report
explicit skips. The repository also retains five pre-existing tests that target
the absent `evaluation/run_ha_benchmark.py`. The missing driver and unchanged
test calls were verified on public main, and representative failures were
reproduced there. Those obsolete-entrypoint tests are not repaired by this core port.
Several older logging smoke tests invoke `uv run` internally and can create
their own environment/lock file. Use the focused checks above when validating
the core without an implicit environment sync.

These boundaries must remain visible: passing the focused tests is not a claim
that every pre-existing repository test passes. The shared `run_benchmark.py`
entrypoint, not the removed `run_ha_benchmark.py`, is documented above.

## Macro observation-space metadata repair in this integration

The frozen physical runtime changed macro observation bound 8 to infinity after
constructing its Gymnasium Box. Gymnasium cached that coordinate as bounded,
so random observation-space sampling could fail.

The integration constructs the Box only after all numeric bounds are finalized.
The low/high arrays, observations, actions and reward/physics equations are not
changed by this metadata repair. Tests cover both modes, sampling/membership,
the semantic-mode attribute and new SAC checkpoint save/reload.

Existing published checkpoint ZIPs are not rewritten. Their serialized Box
metadata can still contain the historical defect if an application samples the
saved observation space directly. Normal prediction on finite real observations
is a different operation. A versioned migration of old model metadata, if needed,
must preserve original artifacts and declare new hashes.

## Metrics and remaining limits

This shared-core integration is not a declaration that all benchmark/model
limitations are resolved:

- The RegD generator's last-sample mean correction does not enforce its
  advertised per-window energy neutrality. Deliberate stress bias is separate.
- PCC voltage still uses gross facility demand rather than the battery-adjusted
  boundary. Metered grid-import accounting is a separate, corrected path.
- RegD window/calibration assumptions remain tied to the declared five-second
  setup; alternative clocks need their own validation.
- The declared clearing-temperature setting and battery charge-limit telemetry
  retain the limitations documented in the source audit.
- Generic benchmark/callback paths retain some aging, survival, missing-metric,
  cohort and export-schema limitations. A generic CSV is not automatically an
  audited Table 1 or Table 2 reproduction.

The frozen publication uses its separately registered runners, gates, recovery
and aggregation rules. Keep those workflows and the original failures distinct
from generic diagnostics. Source-specific recovery is not automatically added
to every core environment episode.

For exact historical reproduction, consult the existing published snapshots:

- [Table 2 guide and checkpoint map](https://github.com/HewlettPackard/dc2grid/blob/82acb0072804dd27d11890bcbf9ebdc3d0a8cb7e/TABLE2_SNAPSHOT.md).
- [Table 1 clean source/result export](https://github.com/HewlettPackard/dc2grid/blob/927c527adf19e5240caaf5580579646f4381d290/TABLE1_SNAPSHOT.md).
- [Remaining issues and evidence boundaries for the frozen snapshots](https://github.com/HewlettPackard/dc2grid/blob/927c527adf19e5240caaf5580579646f4381d290/REMAINING_ISSUES_AND_CONCERNS.md).

Those immutable evidence references do not imply that this core integration
contains their full experiment packages or that the remaining physical-model
repairs have been evaluated. New physics changes require a separate versioned
comparison, rather than silently replacing published numbers.
