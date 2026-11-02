# PID-inclusive progress and trade-off extract

This separate deliverable contains only **Section 4.5, Destressing the Grid**,
and **Appendix Q**. The original archive's `camera_ready.tex` is the historical
starting point. That original source ends at Appendix P; Q was added during the
follow-up and is used here for the comparison supplement.

The original archived Table 2, corrected rule-macro results, later tracking
studies, and economic-dispatch studies are kept in distinct panels. Differences
in simulator semantics, macro bidding, workload constraints, training seeds or
evaluation noise must not be presented as one uninterrupted learning curve.

## Additional matched PID references

The user explicitly authorized evaluating PID on the newest confirmation cells.
This is a **post-confirmation descriptive addition**, not a change to the original
43-policy test, its candidate, its statistics, or its frozen checkpoints.

- **PID (native):** unchanged controller defaults and all four original actions.
  It may defer batch work; that behavior is measured and disclosed.
- **PID (full work):** exactly the same controller with only executed throttle
  fixed to 1. This matches the newer full-work restriction; all other commands
  and gains are unchanged. It is not silently called the original PID.

Both use the same full-residual zero-ask macro, registered calendar/noise/bias
cells, frozen `physical_v1` simulator, independent nominal reference, 24-hour
service and two-hour continuing recovery. No gains are tuned and no learning
is performed. The economic credit-allocation guard is **not** retrofitted to
PID. Actual paid-service ratio and physical/recovery qualification are measured.
Thus this is an end-to-end reference comparison, not a pure supervisor ablation.

The two references each run all 220 cells: 160 new-calendar-confirmation cells
and 60 known-peak/new-noise cells, totaling **440 additional episodes**. Those
cells are now consumed; no new blinded-validation claim is made.

All incomplete and disqualified cases remain in the record. An all-complete but
disqualified cohort may have a clearly marked diagnostic mean, never an
admissible mean or a survivor-only average. No incomplete full-horizon metric is
fabricated. Original failed training/promotion gates remain failed.

The evaluator reuses the already-frozen generic physical runner through its
explicit hardware-factory argument, without modifying or monkeypatching any
frozen task. A historical native-PID case must reproduce its original values;
new complete-case inputs must match the already-audited confirmation inputs.

## Completed comparison extract

[Open the nine-page PDF](publication/pid_progress_comparison.pdf).
It contains only Section 4.5 and the newly assembled comparison Appendix Q:

- Page 1: the updated original rule-macro comparison, with PID and the best
  low-level follow-ups visible.
- Pages 2--3: the exact archived Table 2 body and the independently reproduced
  explanation of PID's 6.74 to 5.641 MW change.
- Pages 4--5: Final-B tracking, macro-only changes and historical matched
  economic-policy results.
- Pages 6--7: the latest matched PID/SAC/PPO/scripted comparisons, separated by
  calendar panel and signal bias.
- Pages 8--9: the relief/cash plot, PID work/SOC diagnostics,
  gains/failures/limitations and audit trail.

All **440 PID episodes completed and recovered**. Each variant qualified in
177/220 cases. The identical 43 failures per variant concern only the **24-hour
service-end SOC reserve**, not the 26-hour recovery closure or the 90% paid
service floor. Every displayed mean uses all complete planned cases (80 or 30),
including these reserve failures, and is marked diagnostic where appropriate.
There is no survivor-only PID mean.

The original finalized manuscript, original archive, 43-policy confirmation
and its primary statistical decision remain unchanged. The new comparison is
descriptive and does not attribute executor/overlay changes entirely to RL.
Thirteen focused tests passed. The new PID study performed 98,120 raw checks;
an additional read-only audit verified 42,240 window fee calculations using
the actual 100 kW normalization floor and added work/SOC diagnostics.

The added semantics explanation reuses a completed **60-trajectory
reproduction**: 15 PID/nominal-reference pairs in each of `legacy_v0` and
`physical_v1`. Both reported relief values reproduce, and all trajectories
complete the full 24-hour service window. The 1.0977 MW decrease decomposes into
-1.0984 MW cooling-related relief and +0.0007 MW remaining facility terms.
This is an observed power balance, **not an individual-fix causal attribution**.
The same integer date/seed labels generate different realized inputs across
the two replay modes; no PID gain change, new training, or publication-result
replacement is involved.

The verified summary, source snapshot and trace inventory are retained in
[the diagnostic record](reports/pid_semantics_reproduction.json),
[the exact probe source](diagnostics/probe_pid_relief_semantics.py), and
[the trace inventory](reports/pid_semantics_trace_inventory.csv).
