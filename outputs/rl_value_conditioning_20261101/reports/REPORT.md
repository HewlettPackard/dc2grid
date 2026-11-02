# Audited union-prior SAC grid-relief results

## Outcome

The new SAC controllers substantially improve tracking and reduce battery throughput relative
to the previously published Guided SAC while retaining physical/recovery
qualification throughout the locked evaluation. The improvements are not
dominance on every metric, and they do not establish a general advantage over
behavior cloning.

The headline, **Conditioned Union SAC**, was chosen using final-step development
RMSE before Final-B was opened. Both union variants are displayed in both tables.

### Headline comparison with prior Guided SAC

All numbers below use the rule-based macro and biased service cases
(`regd_load_bias = 0.5`). Battery throughput includes the two-hour recovery.
This is a comparison of different training pipelines and selection rules:
prior Guided SAC completed 153,600 training transitions but its displayed
checkpoints were selected at 4,096 or 8,192 steps on an earlier development
panel; the new methods use the mandatory final 552,960-step checkpoints.

| Panel | Prior Guided RMSE, kW | New RMSE, kW | RMSE reduction | Prior/new BESS throughput, MWh |
|---|---:|---:|---:|---:|
| Original three peak dates, consumed regression | 382.691 | 295.268 | 22.84% | 0.719 / 0.134 |
| New six-date Final-B | 181.377 | 58.270 | 67.87% | 0.834 / 0.142 |

On Final-B, the headline controller delivers **5.831 MW** of facility relief,
versus **5.103 MW** for matched PID: **14.27% more relief**. Its tracking RMSE is
58.270 kW versus PID's 715.112 kW. It qualifies in **144/144** model/case pairs
under each macro, including both bias conditions.
This is not a relief lead over the stronger alternatives on that panel:
BC gives 5.838 MW, the analytic controller 5.852 MW, and prior Guided SAC
5.869 MW.

On the original peak dates, its relief is 5.567 MW versus PID's 5.641 MW.
That small reduction is reported rather than hidden. Cash relative to PID is
+\$364 per 26-hour biased regression case and +\$236 on biased Final-B cases;
prior Guided SAC has somewhat higher cash and relief, but substantially worse
tracking and battery throughput.
The headline net modeled cashflows themselves are -\$203,062 and -\$118,733
per biased case, respectively. The positive cash deltas are about 0.2% of these
six-figure energy-cost-dominated amounts, not large operating profits.

In the rule-macro headline scope, **BC is numerically better on all four core
metrics on both panels**:

| Panel | BC / headline RMSE, kW | BC / headline relief, MW | BC / headline BESS, MWh | BC / headline net cash, USD |
|---|---:|---:|---:|---:|
| Consumed peak regression | 290.506 / 295.268 | 5.583 / 5.567 | 0.109 / 0.134 | -203,038 / -203,062 |
| Final-B | 53.311 / 58.270 | 5.838 / 5.831 | 0.092 / 0.142 | -118,726 / -118,733 |

The analytic controller is the tracking leader: 50.155 kW on peak regression
and 0.156 kW on Final-B. The central result is **stable, full-budget SAC**,
not a demonstrated incremental benefit over imitation or analytic control.
The lower SAC throughput is a physical cycling measure, not an independently
monetized wear saving: for example, 0.134 versus 0.719 MWh is about 0.9% versus
4.8% of the 15 MWh battery nameplate in absolute AC throughput.

### Conditioning ablation

The potential is not established as the active ingredient. Conditioned versus
unconditioned union RMSE is 63.164 versus 63.259 kW on development (a 0.15%
selection margin), 295.268 versus 294.409 kW on peak regression (conditioned is
worse), and 58.270 versus 59.053 kW on Final-B (conditioned is better).
These small, mixed-direction differences do not establish a material
conditioning advantage or statistical equivalence. The pre-final headline
choice is retained, and both methods are shown.

## What changed

- The policy prior is evaluated on an equal mixture of fixed demonstration
  states and the **same current replay batch already drawn for SAC**.
- Total Gaussian KL weight remains 0.1; each state source contributes 0.05.
  The reference standard deviation 0.01 is a design prior, not an uncertainty
  estimate.
- The conditioned ablation applies a frozen, causal state potential only when
  sampling replay rewards. Original rewards remain in storage and are used at
  evaluation. True-terminal masking preserves the discounted objective.
- Deployment remains an ordinary neural SAC actor with 24 causal features and
  three commands: pump, HVAC and BESS. Full work service is fixed. There is no
  analytic controller, frozen reference actor, potential, or rescue projection
  at deployment.
- The final checkpoint is mandatory. No early-best checkpoint was substituted.

The first single-source KL pilot was not promoted: fixed-state regularization
had three thermal terminations in one development condition, while one
replay-state fit regressed substantially. Those attempts and decisions remain
committed and in the final inventory.

## Training and selection

Each of the eight final fits completed:

- **552,960 online transitions**;
- **551,936 SAC actor updates**;
- **552,192 critic updates**, including 256 critic-only calibration updates;
- **32 complete training episodes**, covering all 30 allowed training dates.

All eight fits retained qualification at every registered development milestone.
The extension repeated the original pilot prefixes from the same initializers
and seeds. Actor and critic parameter hashes matched at all repeated milestones;
these repetitions are counted as computation, not as independent replications.

The final 44 admitted policy IDs and 36 Final-B cells were locked and committed
before one-shot evaluation. Historical non-admitted methods were retained as
explicit slots rather than averaged away.

## High-level result

The higher-level incumbent remains **full admissible residual quantity at zero
ask**, not a new learned high-level SAC policy.

With the same headline hardware controller on biased Final-B cases:

| Macro | Mean committed, MW | Relief, MW | Absolute RMSE, kW | Mean paid performance |
|---|---:|---:|---:|---:|
| Rule | 11.490 | 5.831 | 58.270 | 99.55% |
| Full residual | 15.169 | 7.957 | 133.053 | 99.36% |

Full residual changes cash by **+\$3,743.65 per 26-hour case**, relief by
+2.126 MW, and absolute RMSE by +74.783 kW. The larger absolute error occurs
under larger and differently distributed commitments; this is not a
like-for-like hardware-accuracy comparison. Full-residual relief is still below
BC's 7.977 MW and prior Guided SAC's 7.994 MW on this panel.
This is a paired comparison of two existing macro policies, not a learned
planning gain or proof of global quantity optimality. Bidding changes dominate
the modeled cash delta; the new hardware contribution is chiefly tracking
stability and lower battery throughput.

Two additional causal forecast-based macro studies were tested on the known
development panel, averaging both bias conditions:

- Prefix-credit quantity control: **+\$2.953/case**.
- Expected-terminal-surplus quantity control: **+\$4.244/case**.

Both passed all physical/recovery/service-floor checks, but both missed the
unchanged **+\$5/case** advancement threshold. Neither was promoted to Final-B.
Forecast credit was never booked as actual delivered service.
All measured gains were in the biased stratum: +\$5.905 and +\$8.489 per
biased case, respectively, with exactly zero gain in the unbiased stratum.
The loan inherited the ridge/zero-threshold configuration that was the best
of three earlier development candidates on the same cells; this adaptive
two-stage development selection is not a fresh validation.

## Verification and limitations

- **2,904 publication episodes** completed 24 hours of service plus two hours of
  physical recovery.
- **957,264 scalar-or-array raw-trace checks** passed.
- All **1,056 new-union-policy publication episodes** qualified; comparator
  failures remain in the report.
- Qualification means physical/recovery gates for rule-macro rows and adds the
  actual 90% capacity-service floor for non-nominal full-residual rows. The
  full-residual-reference ratio on rule rows is diagnostic, not an enforced
  quota, and can exceed one because that rule can bid above residual capacity.
- Exogenous and nominal-reference trajectories match across policies.
  Full-residual commitment, price, duration and admission schedules also match.
- The original three peak dates are consumed regression evidence. Final-B is
  a new calendar-combination panel, not an unseen-workload-phase test.
- Dates and noise seeds are crossed and correlated. Reported SDs are
  descriptive, not confidence intervals or claims of statistical significance.
- Stronger tracking/battery performance does not imply better cash, relief, or
  actual system-maximum reduction on every panel.
- The original archive, earlier verified publication, frozen physics and
  read-only AAAI workspace were not modified.

## Deliverables

- [Readable Table 2 and Final-B preview](../publication/build/table2_preview.pdf)
- [New camera-ready archive](../publication/camera_ready_table2_union_updated.zip)
- [Updated camera-ready source](../publication/camera_ready.tex)
- [Audited statistics and inventory](./final_report.json)
- [All case-level metrics](./final_episode_metrics.csv)
- [Complete learning curves](./extended_learning_final.png)
- [Publication validation receipt](../publication/validation_receipt.json)
- [Final validation summary](./completion_receipt.json)
- [Bulk-result SHA-256 inventory](./result_inventory.json)

The table and exact changed-section previews compile successfully with shell
escape disabled. The full
manuscript build is blocked by the pre-existing missing `algorithm.sty`
dependency; that limitation was not bypassed.
