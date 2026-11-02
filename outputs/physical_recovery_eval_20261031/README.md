# Locked controller evaluation with physical recovery

This experiment implements the user-approved common two-hour recovery phase.
No policy is retrained, no checkpoint weights are changed, and the manuscript,
original archives, previous registrations and frozen simulator source stay intact.

## Service and recovery contracts

The service period remains exactly 24 hours, at five-second resolution, under
`physical_v1`. Its trajectories, bias normalization, market handshake, and
settlement are unchanged. Development cases reconstruct the previously recorded
hardware actions, infer the original macro actions, and require byte-identical
uncompressed hardware and macro CSVs before beginning recovery.

Only a healthy time-limit truncation can continue. A service termination is never
revived. At that boundary the evaluator appends real calendar load and matched
weather-climatology data, retaining all live plant, queue and RNG states.
The 24-hour normalization of the biased signal is preserved, including during
the continuation. This is not a 26-hour renormalization of the original task.

For the next **fixed two hours**, every controller uses:

- Full workload service: throttle 1.0.
- Nominal pump and HVAC settings: 0.7 and 0.7.
- No regulation commitment, bids, or revenue, including no standing DR fallback.
- Bounded battery dispatch toward that episode's initial SOC. A one-step model
  solve respects the existing efficiency, power, SOC and float32 action limits.
  Trial battery copies do not mutate the live battery or its age.
- The same ongoing arrivals, weather, load and proxy LMP as its matched baseline.

Recovery lasts the full two hours even if inventories settle sooner. Existing
thermal, frequency and voltage termination rules stay active. Neither service
nor recovery failures receive reset states, extended horizons, or replacement
models. Nominal fixed hardware receives the same continuation.

Closure requires a complete recovery without physical termination, absolute SOC
error at most `1e-6`, backlog at most `1e-6 kWh`, and absolute temperature
differences from the continuing nominal reference at most `0.1 C` in each zone.
Work conservation over the entire 26 hours must hold strictly within `1e-4 kWh`.
These are declared numerical/engineering tolerances, not exact inventory
identity. Residuals, usable-energy differences, warning ticks, queue exposure
and irreversible battery age remain reported. The service qualification gates
remain unchanged; recovery does not erase a service violation.

## Stages and selection

1. Development recovery: the eight existing matched-bias cells on days 196/256.
   Four classical comparators, eight original low-policy slots and eight original
   hierarchy slots are retained. Sixteen policies passed all eight existing
   service gates, giving 128 executions and 32 explicitly unavailable/non-admitted
   policy-cells. Macro1002 is not admitted after its two thermal terminations.
2. Freeze selection using development results only:
   - Hardware: among fully qualified and closed original low checkpoints, maximize
     biased stress-window facility relief versus nominal.
   - Macro: among fully qualified and closed original pairs with equally qualified
     matched rule-hardware references, maximize incremental cashflow after recovery.
   - Both rules require all eight cells; ties use policy ID.
3. Lock the checkpoint hashes and run fresh deterministic inference on days
   207/248/249, seeds 100--104, and biases 0/0.5: 30 cells per policy, 600 planned
   policy-cells, 480 executions and 120 unavailable/non-admitted entries.
   No held-out selection, retraining, fallback or survivor-only aggregate is allowed.

The primary hardware comparison holds rule-based macro logic fixed. The learned
hierarchy is a separate comparison, including incremental macro benefit against
the **exact same hardware weights** used in macro training. Restored policies stay
stability ablations; old macros are never paired with restored weights.

The independent [macro1002 diagnosis](macro1002_diagnosis.json) finds both faults
under rejected-bid 5 MW fallback commitments, with near-zero HVAC and full battery
discharge. The matched rule-macro runs do not fail. This supports a control-envelope
concern, not a claim that large accepted bids or insufficient installed cooling
alone explain the failures. Simply capping bids is not an established remedy.

The seed slots retain training/selection failures, but this is a selected-checkpoint
evaluation, not a fresh multi-seed algorithm ranking. The two development dates
share a workload phase; the three test dates have phases 27/8/9 of the cyclic
30-day tape. Evaluation-case standard deviations are not training uncertainty.

## Physical and financial reporting

The original 24-hour stress mask and battery-exclusive facility relief remain
separate from battery-inclusive PCC relief, actual maximum-load differences,
and recovery energy. Table-2-shaped summaries use the fifteen biased test cases
and matched unbiased revenue counterparts; they do not overwrite the manuscript.
Cooling and IT contributions are measured separately.

Cashflow after the fixed recovery phase is service regulation revenue minus
metered electricity cost over all 26 hours. A **closed-recovery** comparison
additionally requires closure in both trajectories. No full-cohort estimate is
produced when any required value is unavailable, incomplete or unclosed.

Recovery occurs at 00:00--02:00 of the **following** calendar day. The measured
time-weighted service/recovery LMP means and energy-weighted battery charging/
discharging prices are reported separately. An overnight price regime can change
the value of ending charged versus discharged; the results do not establish the
same economic value for recovery at another time or under other prices.

Wear sensitivity subtracts `price * absolute AC battery throughput`, with
illustrative prices 0/5/10/25/50 USD per MWh and an explicit break-even price/slope.
Those are valuation assumptions, not measured replacement costs or a calibrated
degradation model. Recharging does not reset aging. No arbitrary thermal
`C*dT/COP` or queued-work value is subtracted as though it were measured electricity.
Work completion does not erase latency/SLA differences. Total business profit
and real ISO payment accuracy are not claimed.

NYC load is from 2023; weather is calendar-matched 2024 climatology. Load and weather
use the frozen causal hold/maximum-age rules. LMP is the simulator's exogenous load
proxy. Synthetic RegD is still not guaranteed window-neutral. None of these
limitations is repaired by the recovery procedure.

## Execution and evidence

Use the project Python environment. The script resolves data paths to the frozen
source, caps native/PyTorch threads at one, and uses eight distinct physical CPU
cores. No GPU or training optimizer is used.

```bash
python run_evaluation.py register-development
python run_evaluation.py run-development
python run_evaluation.py lock-heldout
python run_evaluation.py run-heldout
```

Registrations freeze code, configuration, runtime, inputs, selection rules and
checkpoint hashes. Re-execution refuses to overwrite an existing run. Every
attempt retains a completion or failure record and compressed interval traces.
Source, checkpoint, matched-evidence, continuation-state, physical accounting,
paired exogenous and complete-cohort invariants are checked.
This README is the frozen protocol; post-hoc figures and interpretation belong in
the derived reports, not edits to the registered documentation.
