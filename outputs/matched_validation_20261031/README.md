# Matched validation-only controller matrix

This evaluation uses existing checkpoints and the frozen `physical_v1` source
from the completed online stability pilot. It does not train policies, modify
the underlying simulator, or evaluate the three held-out peak days.

## Registered scope

- 32 policy entries: four classical references, eight original selected hardware
  slots, eight original selected hierarchy slots, and twelve restored final
  hardware policies.
- 29 entries are executable. Original low seed1000 and hierarchy seeds1000/1005
  have no qualified selected model; their cells remain unavailable with reasons.
  No final checkpoint is substituted.
- All twelve restored final policies are evaluated, including five previously
  unqualified policies explicitly labeled diagnostic-only.
- Learned macros are paired only with the exact hardware checkpoint hash recorded
  during their training. No macro has been trained against the restored policies,
  so that surface is declared not attempted rather than fabricated.
- Eight cells: days196/256; seeds90001/90002 on day196 and90003/90004 on day256;
  each at RegD load bias0 and0.5. Four are the original validation cells and four
  are new counterparts that allow a same-seed bias comparison.
- Thus232 rollouts are executed and24 unavailable policy-cell entries retained.
  CPU execution is capped at four workers with one native/PyTorch thread each.

This is a selected-candidate validation inventory, not a fresh algorithm ranking.
Old models were selected by qualified validation reward; restored models are the
online pilot's prespecified final checkpoints. Their selection procedures differ.
Continuation streams share parent policies and are not independent training
seeds. The two validation dates share a cyclic workload phase; grid/weather and
RegD conditions still differ. None of these results is a held-out paper result.

## Comparisons and signs

1. Every available controller pairing versus **rule macro + nominal fixed
   hardware** `(throttle=1,pump=0.7,HVAC=0.7,BESS=0)`.
2. Each learned macro versus rule macro with **the same hardware weights**.
3. Restored hardware versus its original selected healthy parent, when available.
4. Each pairing under bias0.5 versus its own bias0, matched by day and seed.

Nominal refers only to the fixed-hardware pairing. PID, rule-based hardware and
bang-bang are additional comparators, not alternative denominators for the
paper-style relief ratios. Holding rule macro fixed means fixed **logic**, not
identical commitments: its state-dependent bids may change with hardware behavior.

Positive facility relief is `B_facility-A_facility`. PCC relief additionally
includes `A_battery_discharge-B_battery_discharge`, with charging negative.
Component relief is calculated from actual logged electrical/cooling components
in both trajectories, never from assumed35MW/1.05MW constants. All power-component,
work-conservation, and electricity-settlement identities are asserted.

The stress mask is the top decile of the complete **exogenous zonal load**
trajectory. `SPDR_proxy_pct` follows the paper's mean-stress-relief divided by
modeled system-peak denominator; it is not the actual change in maximum load.
Actual facility-boundary and PCC-boundary system-peak differences are separate.
Paper-style ratios are reported only versus nominal fixed hardware; other
comparisons use absolute MW, MWh and currency differences.

Daily mean flexible-power reduction can be zero despite substantial intra-day
deferral and repayment. Stress-window shifted work, non-stress service changes,
queue exposure, and terminal backlog are reported separately.

## Economics and incomplete accounting

`cashflow_delta_before_recovery_usd` is:

```
(A regulation revenue - B regulation revenue)
    - (A metered electricity cost - B metered electricity cost)
```

This is **not net profit**, a sustainable-service value, or an assertion about
real ISO compensation. The simulator uses a benchmark price/quantity/performance
settlement proxy, under which nominal fixed hardware can still earn revenue.
Gross contract revenue and tracking discounts are reported separately.

No recovery rollout or degradation-cost valuation is performed. Initial/final
nominal stored battery energy, the distinct usable-energy property, battery
throughput/aging, queue energy and final temperatures are retained as obligations.
Different terminal inventories prevent an unconditional net-value ranking.
Observed LMP extrema describe the evaluated day, not guaranteed future recharge
prices. Thermal `C*dT/COP` and queued-work proxies are not silently subtracted
from relief MWh as though they were metered recovery electricity.

Grid prices are exogenous. System-wide price-suppression/avoided-cost proxies are
excluded. Synthetic RegD is not guaranteed window-neutral; same-seed bias tests
verify the exact clipped signal construction.

## Qualification and artifacts

Prior checkpoint eligibility, full-horizon completion, and current per-cell
constraint qualification are separate fields. Early-terminated/incomplete pairs
have no full-horizon numeric comparison. Their records remain present; other
valid pairs are not discarded. No survivor-only cohort estimate is implied.

The inventory was independently checked, including missing models and exact
frozen-inner hashes. An adversarial accounting review and invariant tests
preceded execution. `registration.json` freezes the runner, source, inputs,
policy inventory, review and runtime versions.

Each rollout retains compressed hardware and macro traces plus `episode.json`.
Reports retain all planned episode rows and explicit pairwise comparisons,
including unavailable comparisons. No manuscript values are changed by this task.

Run with the project's Python environment from the frozen source directory:

```bash
python /absolute/path/to/matched_validation_20261031/run_matrix.py --register
python /absolute/path/to/matched_validation_20261031/run_matrix.py
```

The second invocation refuses to overwrite a prior execution. A read-only
re-analysis uses `--summarize-only` with the same registered runner.
