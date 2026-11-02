# Final camera-ready grid-relief confirmation

Scientific registration: `628973aa3ebcb14ea8d5bd1f32543dfc773de57e8298cfac30e7f0e8617fe5d7`.
Technical amendment: `e386a6314a06cbf18e107ca6dbcaaec51cb759fdfc4693a3527f615cdf5e6c7a`.

## Outcome

**supported on the prespecified conditional panel.**

This study did not train or select another checkpoint. It confirmed a fixed four-checkpoint economic SAC candidate against a prespecified same-coordinate full-fraction rule, while reporting all 27 scripted controls and all four PPO/SAC families.

The economic supervisor is not a new learned bidder or a pure SAC low-level controller. Full-residual zero-ask bidding and analytic hardware execution are common to every row.

## Prespecified paired comparison

Intervals below resample stochastic noise within fixed dates and hold the four checkpoints fixed. They are not calendar-population or training-seed-population confidence bounds.

| Panel | Bias | PCC relief change (MW), 95% interval | 26 h cash change (USD), 95% interval |
|---|---:|---:|---:|
| Eight new calendar combinations | 0.0 | +1.040 [+1.012, +1.069] | +47.7 [+45.4, +50.1] |
| Eight new calendar combinations | 0.5 | +0.866 [+0.836, +0.896] | +43.4 [+41.2, +45.7] |
| Three known peak dates, new stochastic streams | 0.0 | +0.108 [+0.094, +0.122] | +150.7 [+144.5, +156.9] |
| Three known peak dates, new stochastic streams | 0.5 | +0.053 [+0.050, +0.057] | +71.5 [+67.2, +76.1] |

The sole primary stratum is the eight-date, bias-0.5 panel. Its two point estimates and lower bounds must exceed +0.001 MW and +5 USD; the bias-0 mean cash contrast must be nonnegative. These small cutoffs are not claims of commercial materiality. All candidate and primary comparator cases across both panels must qualify.

Primary-gate qualification: 1100/1100.

## Strong scripted controls and the quality trade-off

The tables include the fixed baseline set and representative observed best-cash/best-relief scripts. These descriptive extreme points do not replace the prespecified primary comparator. All 31 family-level rows per stratum are in [the complete CSV](./final_summary.csv).

### Eight new calendar combinations, bias 0.5

| Controller | Facility / PCC relief (MW) | Cash vs no-extra (USD) | RMSE (kW) | BESS (MWh) | PCC peak reduction, 26 h (MW) | Qualified |
|---|---:|---:|---:|---:|---:|---:|
| Analytic, no extra | 7.083 / 7.083 | 0.0 | 0.2 | 0.000 | 4.737 | 80/80 |
| Full-fraction rule | 8.307 / 8.307 | 191.2 | 1189.2 | 0.000 | 5.964 | 80/80 |
| f=1.00, margin 10 | 8.080 / 8.080 | 290.0 | 1861.4 | 0.000 | 5.751 | 80/80 |
| Deadline margin | 8.390 / 8.390 | 316.5 | 2101.9 | 0.000 | 6.068 | 80/80 |
| Credit threshold | 8.273 / 8.273 | 195.7 | 2595.6 | 0.000 | 4.963 | 80/80 |
| PPO | 8.196 / 8.196 | 173.3 | 1005.2 | 0.000 | 5.852 | 320/320 |
| SAC | 8.237 / 8.237 | 184.3 | 1160.3 | 0.000 | 5.882 | 320/320 |
| PPO + credit potential | 7.977 / 7.977 | 132.9 | 769.8 | 0.000 | 5.634 | 320/320 |
| SAC + credit potential | 9.172 / 9.172 | 234.7 | 1612.4 | 0.000 | 6.887 | 320/320 |

Observed family-mean relief/cash frontier: `potential_sac`, `script_deadline_margin10`.

This is a descriptive frontier of the tested families, not proof of global optimality, statistical nondominance, or a deployable action-averaging ensemble.

### Three known peak dates, new stochastic streams, bias 0.5

| Controller | Facility / PCC relief (MW) | Cash vs no-extra (USD) | RMSE (kW) | BESS (MWh) | PCC peak reduction, 26 h (MW) | Qualified |
|---|---:|---:|---:|---:|---:|---:|
| Analytic, no extra | 8.579 / 8.579 | 0.0 | 0.2 | 0.208 | 5.822 | 30/30 |
| Full-fraction rule | 10.010 / 10.010 | 1440.1 | 1355.1 | 0.000 | 7.255 | 30/30 |
| f=1.00, margin 10 | 10.010 / 10.010 | 1446.2 | 1390.3 | 0.002 | 7.255 | 30/30 |
| Deadline margin | 10.010 / 10.010 | 1444.2 | 1376.4 | 0.001 | 7.255 | 30/30 |
| Credit threshold | 10.117 / 10.117 | 1439.5 | 2747.3 | 0.148 | 7.493 | 30/30 |
| f=0.50, margin 20 | 10.009 / 10.009 | 1489.5 | 1473.6 | 0.086 | 7.254 | 30/30 |
| PPO | 9.917 / 9.917 | 1265.7 | 1170.7 | 0.050 | 7.162 | 120/120 |
| SAC | 10.001 / 10.001 | 1382.9 | 1321.9 | 0.000 | 7.243 | 120/120 |
| PPO + credit potential | 9.626 / 9.626 | 952.6 | 881.9 | 0.083 | 6.873 | 120/120 |
| SAC + credit potential | 10.064 / 10.064 | 1511.6 | 1462.2 | 0.017 | 7.308 | 120/120 |

Observed family-mean relief/cash frontier: `potential_sac`, `script_credit_one`.

This is a descriptive frontier of the tested families, not proof of global optimality, statistical nondominance, or a deployable action-averaging ensemble.

## All four frozen candidate checkpoints

| Training seed | Primary PCC contrast (MW) | Primary cash contrast (USD) |
|---|---:|---:|
| 1200 | +0.417 | +30.3 |
| 1201 | +0.264 | +13.1 |
| 1202 | +0.416 | +46.9 |
| 1203 | +2.367 | +83.5 |

Leave-one-date-out label: **date-robust** under the prespecified descriptive rule.

## Scope and previous results

- Primary: eight new, below-p90 calendar combinations, ten unique noise streams per date, both bias strata. Two dates are in the p75-p90 band.
- Secondary: three previously studied peak dates with ten new streams each; this is conditional stochastic replication, not unseen-calendar evidence.
- Load is NYC 2023; weather is calendar-aligned NYC 2024. Workload phases repeat and are not unseen.
- Mean stress-window relief is not maximum-system-peak reduction. Both 24 h and recovery-inclusive 26 h maxima are independently audited.
- The actual 90% service floor and physical closure are measured. The common 92% guard is not credited as learned safety.
- Earlier failed low-level PPO/SAC and composite economic promotion gates remain failed. Their old reserved panel remains unopened.
- In the earlier rule-macro Final-B tracking study, BC (53.3 kW) tracked better than conditioned Union SAC (58.3 kW); analytic execution was stronger still. Economic improvement is not a tracking win.

## Verification and artifacts

- 10,918 new evaluation episodes; 864 exact reused neural records.
- 11,782 audited model/case records; 11,782 qualified.
- 12,924,854 raw scalar-or-array checks, with exogenous/reference and stress-mask equality verified.
- 20 focused confirmation tests and 5 publication tests passed. Five independent adversarial reviewers checked the design and implementation.
- A narrowly scoped technical amendment preserved all 483 completed known-panel cases, corrected the endogenous-frequency pairing assertion, and made restart non-destructive. Completed physical failures are never re-rolled.
- The original author snapshot and manuscript archive are unchanged. The new copy preserves the unrelated original benchmark table.
- The author-approved unused algorithm imports were removed after checking the complete included dependency chain. No packages were downloaded.
- The archive's 76 bibliography entries were preserved; seven verified missing entries were added. ICC metadata is primary-source verified, but its gated full text was not directly inspected.
- The public repository and its grid-relief evaluator were verified at `7550013331a865fa45684cb7cd973279312d4678`. This local confirmation branch was not pushed; new results are not falsely described as already on public main.

[New manuscript PDF](../publication/full_build/camera_ready.pdf) | [Table 2 preview](../publication/build/table2_preview.pdf) | [New manuscript source](../publication/camera_ready.tex)

Bulk raw traces remain in the study's evaluation directories. Their hashes and byte sizes are inventoried separately rather than silently claimed to be backed up as Git blobs.
