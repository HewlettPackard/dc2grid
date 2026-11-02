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
