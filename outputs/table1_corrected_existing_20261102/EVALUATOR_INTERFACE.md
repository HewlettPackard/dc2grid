# Evaluator interface

`table1_eval.run_case(protocol, policy, seed, destination)` runs one authorized
case and returns the dictionary also written to `episode.json`. Run from the
repository root. `destination` must be a **new** directory under this study.
Existing directories are never overwritten, resumed, or retried.

The parent creates and freezes the registration. Required protocol keys:

- `registration_id`: parent registration identity.
- `panel`: exactly exported `PANEL_SPEC`.
- `policies`: exactly exported `ROW_SPECS`, in order.
- `models`: exactly exported `MODEL_SPECS`.
- `config`: `{path, sha256}` for native `c2g_env/config.yaml`.
- `source_files`: mapping of every relative path from
  `runtime_source_paths()` to its SHA-256.
- `data_files`: mapping of every relative path in `DATA_PATHS` to its SHA-256.
- `evaluator_sha256`: hash of `table1_eval.py`.
- `recovery_ops`: exactly `RECOVERY_OPS_SPEC`.
- `recovery`: exactly `RECOVERY_SPEC`.
- `service_gate`: exactly `SERVICE_GATE_SPEC`.

The module exports `file_hash`, `file_receipt`, and `identity` for integration;
it does not create registrations or launch a panel. A policy can be a row ID or
a dictionary matching its `id`, `macro`, and `hardware` fields. Seeds are
100–104. There are seven rows, 35 cases, and no PPO, LLM, fitting, optimizer,
checkpoint-selection, or economic-policy substitution.

## Interpretation and metrics

This is a **new corrected-physics selected-checkpoint comparison**.
Default scenario, day 0, zero bias, and seeds 100–104 reconstruct documented
source defaults; they do not prove the original published producing conditions.
Day 0 was in these models' training panel. Five episodes do not represent five
independently trained policies.

Service is exactly 17,280 five-second ticks / 96 complete 900-second windows.
Every row uses one native macro environment and records both levels from the
same episode. No-inner means explicit `[1, 0.7, 0.7, 0]`. Controllers are fresh
per case, including bang-bang hysteresis. Random actions use a private action
space explicitly seeded with the case seed. SAC inference is deterministic CPU
inference using only the two exact selected seed-1007 checkpoint hashes.

`service.observed_service_metrics` always describes the observed service
prefix. `service.table1_service_metrics` contains full-service values only
when the horizon completed without termination; otherwise its six metric
values are null and `survived` is zero. No missing duration is imputed.

- `tracking_rmse_kw`: RMS over fast ticks of demand minus
  `1000 * (p_reference_mw - p_grid_import_mw)`. Native reported response/error
  must agree with this metered definition.
- `regulation_revenue_usd`: sum of actual macro settlements.
- `macro_mean_reward`: sum of native macro rewards / actual returned macro calls.
- `low_mean_reward`: sum of native low rewards / actual fast ticks.
- `thermal_warning_fraction`: fraction of post-step fast states where **either
  temperature is >= 33°C**; not percent and not episode incidence. Native
  warning flags are retained separately in raw rows.
- `bess_aging_x1e4`: `(service_end_age - reset_age) * 1e4`, including the first
  tick; this is the simulator aging accumulator, not dollars.
- `survived`: exactly the complete fast horizon, 96 full macro windows,
  truncation, no termination, and no recorded native faults.

Aggregate across all five registered cases using sample SD (`ddof=1`).
Do not silently remove failed cases or average only non-null full-service
values. An execution failure writes `failure.json`, retains raw partial
traces, and raises; the launcher must retain it rather than retry.

Supplemental gates require physical completion, service-end SOC >= 0.20,
backlog <= 1% of service arrivals, warning fraction <= 0.01, work conservation,
and inherited two-hour recovery closure. A completed full-cohort estimate
remains available if a supplemental gate fails, but must be labeled diagnostic.
There is **no maximum-capacity admission floor, credit guard, or service-action
override**. Macro commitment and acceptance statistics are descriptive only.

## Supplemental continuation and raw evidence

The existing pinned `recovery_ops.py` functions are executed unchanged.
Only their function definitions are loaded, avoiding that file's historical
import bootstrap, which would otherwise redirect native imports to a different
source tree. All functions use current registered native classes.

Healthy service continues for the existing fixed two-hour recovery:
zero commitment/revenue, throttle 1, pump/HVAC 0.7, battery dispatch targeting
initial SOC, and continuing weather/work/grid/thermal/RNG state. It does not
reset the plant or stop when closure is first achieved. Native fault termination
remains active. Closure, work conservation, metered electricity cost, and AC
battery throughput are supplemental only. Recovery never contributes to
Table 1's service rewards, tracking, revenue, thermal fraction, or aging.

Output files:

- `attempt.json`: pre-execution identities and closed-case specification.
- `service_hardware.csv.gz`: every fast info dictionary, pre/post observations,
  requested action, physical tracking reconstruction, actual commitment,
  explicit warning convention, and all native fault flags.
- `service_macro.csv.gz`: every actual market settlement, macro reward,
  pre/post observations, requested bid, and interval length.
- `recovery_hardware.csv.gz`: separate continuing physical recovery rows.
- `episode.json`: service metrics, supplemental recovery/accounting, runtime
  versions/backend/noise policy, hashes, reset/end inventories, and faults.
- `failure.json`: exception, phase, partial row counts/hashes, and any completed
  service summary. Empty-phase CSVs are absent, never fabricated.

The evaluator deliberately retains the published `physical_v1` behavior,
including its known remaining RegD/voltage limitations. It does not change
the plant, replay, rewards, fault thresholds, or stochastic-input mechanisms.
