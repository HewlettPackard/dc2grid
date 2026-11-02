# Fractional economic-budget PPO/SAC follow-up

**Adaptive development results, not the original Table 2 and not a
fresh final test.** This follow-up was registered after observing V1
on the same consumed Dev-B cells. All 16 fits and all four model seeds
per family are retained; only mandatory final checkpoints are used.

## What changed

- The action chooses a fraction of an allowance frozen at the first
  actual hardware tick after the contract handshake. The requested MW
  stays fixed within that window; the original per-tick guard remains.
- The second arm adds training-only observed-credit potential
  `Phi = 10 USD/MWh * credit_MWh / 100 USD`, with gamma 1 and true-terminal
  next-potential zero. It preserves complete and invalid episode returns.
- The value 10 came from the previously tested heuristic, not discovery.
- No physics, bidding, observation, work, SOC target or recovery change
  is introduced. Tracking deviations remain an explicit economic trade-off.

Four native smoke fits passed. For each algorithm, both reward arms had
identical physical warmup states/actions/terminals; their reward
difference matched the potential formula. The task/integration tests
also verified 26-hour behavior and true-terminal telescoping.

## Matched biased development results

All rows use the existing full-residual, zero-ask macro. Cash is net
modeled 26-hour cash after physical recovery; less negative is better.
Qualification includes both bias strata and the actual 90% service floor.

| Family | Models | All gates | Facility MW | PCC MW | RMSE kW | Net cash USD | Minimum service |
|---|---:|---|---:|---:|---:|---:|---:|
| cooling_first | 1 | Yes | 6.783 | 6.783 | 0.184 | -97,170.44 | 100.00% |
| constant_one | 1 | Yes | 7.783 | 7.783 | 1,000.000 | -96,900.61 | 93.02% |
| greedy_budget | 1 | Yes | 8.019 | 8.019 | 1,209.691 | -96,841.49 | 92.01% |
| margin_ten | 1 | Yes | 7.817 | 7.817 | 1,937.785 | -96,734.30 | 92.01% |
| economic_ppo | 4 | Yes | 7.936 | 7.936 | 1,094.005 | -96,865.57 | 92.01% |
| economic_sac | 4 | Yes | 8.013 | 8.013 | 1,133.407 | -96,860.21 | 92.01% |
| fraction_ppo | 4 | Yes | 7.912 | 7.912 | 1,024.860 | -96,879.16 | 92.30% |
| fraction_sac | 4 | Yes | 7.987 | 7.987 | 1,178.352 | -96,856.22 | 92.01% |
| potential_ppo | 4 | Yes | 7.692 | 7.692 | 783.358 | -96,949.75 | 93.71% |
| potential_sac | 4 | Yes | 8.870 | 8.870 | 1,573.948 | -96,786.90 | 92.01% |

## Predetermined promotion gate

The comparator is a **composite per-metric envelope**, not a single
achievable incumbent: strongest qualified family for cash, facility
relief and PCC relief separately. It includes the new scripted greedy
and price-margin controllers as well as earlier learned/frozen controls.
A family must pass all four seeds' physical/service/recovery cases,
improve both relief metrics, and improve cash without a negative mean
cash contrast in either bias stratum. No threshold was relaxed.

- **fraction_ppo: joint improvement = False.**
- **fraction_sac: joint improvement = False.**
- **potential_ppo: joint improvement = False.**
- **potential_sac: joint improvement = False.**

## Training accounting

Completed final models: **16/16** at the registered
24,576 macro transitions, two environments and 256 critic-only updates.
PPO/SAC are matched on transitions, not optimizer counts or computation.
Individual completion receipts record actual optimizer operations,
episode qualification, conditions and retained transition arrays.

Raw evaluation primary checks: **77,184**.
The underlying physical and recovery ledgers, true contract target,
credit limits, raw cash and deployment without Phi were retained.

## Scope and persistence

- V1 was snapshotted before this follow-up at
  `5e2bf2855a3da4ac8cfea1763df8e9d8b28e9f82`.
- The adaptive-attempt ledger and one-further-campaign cap were frozen
  before training. Reused Dev-B is not relabeled as fresh validation.
- The shared six-date reservation has no new workload phases or p90-tail
  coverage. Native 26-hour staleness checks passed for all six starts.
- Constant offsets and battery-boundary effects are not called load
  shifting or free regulation accuracy.

## Evidence

- [Audited outcomes](../track_b_fraction/development_report.json)
- [Factorial registration](../track_b_fraction/registration.json)
- [Exact warmup parity](../track_b_fraction/smoke_parity_receipt.json)
- [Machine-readable comparison](fraction_summary.csv)
- [Previous full-budget results](REPORT.md)
- [Native calendar coverage](calendar_coverage_verification.json)
- [Development economic frontier](economic_frontier.png)

**No family earned promotion under the registered gate.** The fresh
calendar reservation remains unopened; do not substitute a different
seed, earlier checkpoint, relaxed gate or repeated final opening.
