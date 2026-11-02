# Economic and newest PID frozen replay

This directory is a **release reevaluation adapter**, not a new experiment,
preregistration, or opening. It runs only exact policy/cell slots that already
have saved outcomes. No training, gain tuning, source rewriting, monkeypatching,
historical `verify()` bypass by replacement, or admission-gate changes occur.
The historical verification functions are simply not the release entry point:
the separate release manifest verifies the actual scientific runtime instead.

## Deliverables and export contract

| File | Purpose |
|---|---|
| `adapter.py` | Portable `evaluate(root, policy_spec, cell, output)` and CLI |
| `runtime_manifest.json` | Explicit recorded-absolute-path → checkout-relative-path map; SHA-256 checks for **66 source files, 2 configs, 4 original protocols, 5 data files, and 24 final models** |
| `policies.json` | 73 immutable policy specifications, deployment semantics, evaluator selection |
| `cases.json` | 12,894 consumed policy/cell slots, original record hashes, expected metrics |
| `representative_cases.json` | 11 fixed portability cases, including both failed PID examples |
| `failures.json` | All **86 disqualified PID cases**, counts, and runtime-failure inventory |
| `comparison_scopes.json` | Same-action-space versus end-to-end comparisons; unchanged scientific gates and analysis specification |
| `extraction_provenance.json` | Saved reports/manifests used to derive this catalog; these are **not runtime requirements** |
| `requirements-runtime.txt` | Recorded Python-package versions; Python 3.11, CPU inference |
| `test_adapter.py`, `test_runtime_imports.py` | Episode-free checks; the latter optionally probes relocated sources and original protocol projection |
| `build_metadata.py` | Maintainer-only extraction source; its bytes are required by the extraction attestation, but it is never executed during public replay |

Copy this directory into the release, for example `reproducibility/economic/`.
Export every `checkout_relative` item in the manifest's `sources`, `configs`,
`protocols`, `data`, and `models` arrays **byte-for-byte at the same relative
location**. Do not copy caches, `.venv`, training-transition buffers, or legacy
training artifacts merely because a historical protocol mentions them.

The original absolute path is only a provenance identifier. The adapter never
falls back to it. Symlinks escaping the checkout are rejected. The frozen
`source/data` symlink is unnecessary: the three environment data directories are
explicitly relocated to the release's `data/processed/...`.

The four original protocol JSONs retain their original bytes and historical
IDs. They contain provenance references to files not needed for inference;
those referenced files are **not** release dependencies. In particular:

- No author attachment, manuscript, benchmark ZIP, AAAI-demo, old checkpoint
  lineage, pre-amendment raw-result inventory, Git ancestry, or original
  preregistration opening is needed.
- Hash validation covers every exported scientific source, complete protocol,
  configuration, required data file, and all 24 frozen model archives.
- All 66 source hashes also match the original recorded scientific manifests.
- Model deserialization occurs only after verification, in a fresh process.

## API and commands

Use the frozen-compatible Python 3.11 environment. `torch==2.6.0` CPU wheels are
accepted in place of the recorded `2.6.0+cu124` build; inference remains on CPU.
Other listed package versions are checked exactly. `physical_v1` explicitly
uses the original simple BESS implementation, irrespective of optional PySAM
availability. No dependency installation is performed by the adapter.

```bash
# Files-only verification needs no simulator imports.
python -B reproducibility/economic/adapter.py verify \
  --root /absolute/path/to/new-checkout --files-only

# Also check Python and package versions.
python -B reproducibility/economic/adapter.py verify \
  --root /absolute/path/to/new-checkout

# One ALREADY CONSUMED case, not an experiment/panel opening.
python -B reproducibility/economic/adapter.py evaluate \
  --root /absolute/path/to/new-checkout \
  --case-id 'confirmation/script_zero:calendar:d7_s207829280_b0p5' \
  --output /absolute/path/to/new-checkout/release-results/economic/zero
```

In Python (with the new checkout on `sys.path`):

```python
from pathlib import Path
from reproducibility.economic.adapter import evaluate

receipt = evaluate(
    root=Path("/absolute/path/to/new-checkout"),
    policy_spec={"policy_key": "pid/pid_full_work", "panel": "calendar"},
    cell={"start_day": 107, "seed": 3002708602, "regd_load_bias": 0.0},
    output=Path("/absolute/path/to/new-checkout/release-results/pid-failed-soc"),
)
assert receipt["matches_saved_case"]
assert receipt["qualified"] is False
```

`policy_spec` is an immutable catalog key plus panel, **not an editable policy
dictionary**. Model paths, controller settings, fractions, and gates cannot be
overridden. Unknown or unconsumed policy/cell pairs fail before execution.
Use `representative_cases.json` for the eleven exact requests; do not construct
new seeds. Each request requires a new output directory. There is no automatic
retry, seed replacement, overwrite, or selection among successful attempts.

Keep outputs outside the frozen `outputs/` tree and outside this adapter
directory. The original source tree is explicitly refused as an evaluation
root. `verify` may safely read it for integrity checks without running anything.

## What executes unchanged

| Backend | Frozen evaluator and deployment |
|---|---|
| Absolute economic | `track_b/evaluate_macro.evaluate_case`; original `EconomicDispatchTask`, 0–6 MW absolute request |
| Fractional development | `track_b_fraction/evaluate_fraction.evaluate_case`; original `FractionalDispatchTask` |
| Confirmation | `confirmation_eval.evaluate_case`, `controls.make_environment`, `controls.load_model`, original scripts |
| Native/full-work PID | `economy_eval.evaluate_episode(..., hardware_factory=pid_reference.hardware_factory)` |

The original raw auditors also run: `confirmation_audit.audit_record`,
`audit_pid.audit_episode`, or `evaluate_low.audit_primary`. The manifest covers
their transitive imports, including shared bootstrap modules. Importing training
modules to reuse `MacroLatentEnv` does **not** run their training functions.

Only runtime copies of the environment/protocol are path-projected. The three
data-directory strings, config path, model path, and confirmation's derived
task-file path/hash are relocated; every scientific field stays unchanged.
Original protocol IDs in evaluator output are provenance, not a new opening.
`runtime_inputs/relocation_receipt.json` records both original and derived file
hashes and the complete explicit path map.

Plotting and Python caches are initialized in the new result directory before
historical bootstrap imports. The evaluators therefore need not write caches or
bytecode beside the exported frozen sources.

## Semantics and failure preservation

- Service is **24 h / 17,280 five-second ticks**; recovery continues the same
  physical/random/workload state for **2 h / 1,440 ticks**. Fractional and absolute
  learned tasks keep their finite terminal/no-timeout-bootstrap semantics.
- The **service-end SOC floor is 0.20**. Recovery targets the initial **0.50 SOC**
  with the original `1e-6` tolerance. Closing recovery does not erase a failed
  service-end gate.
- All 86 saved PID disqualifications remain in the catalog. For example,
  `d107_s3002708602_b0` ends service at SOC `0.1636191738621983` and then closes
  recovery. Both native and full-work PID must still be **unqualified**.
- Complete-but-disqualified cash remains a diagnostic. Incomplete 26-hour
  outcomes retain null cash and failed status. No survivor-only aggregate or
  repaired pairing is produced by this adapter.
- The actual paid-service floor remains **0.90**. Economic dispatch keeps the
  original **0.92** credit target and **0.02 MWh** reserve. PID receives no new
  credit-allocation guard.
- Fractional SAC, SAC+creditPhi, and scripts share the native fraction action
  space and fixed analytic executor. CreditPhi is training-only. PPO's original
  latent wrapper applies the transform exactly once.
- Absolute-MW models are never relabeled as fractional models. PID comparisons
  are end-to-end, not same-action-space claims. Full-work PID changes only the
  executed throttle, through the existing explicit factory.
- Stress-window mean relief, the 24-hour difference of boundary maxima, and the
  26-hour difference including recovery remain separate metrics. Cash is metered
  simulator cash, not business profit, temporal shifting, or peaker displacement.
- The 864 previously reused neural records are explicitly marked
  `historical_flat_schema`. Release inference uses the current frozen evaluator
  format while comparing their original scientific outcomes and audited metrics.

Every successful run writes:

1. `release_request.json`, `runtime_inputs/`, `worker.log`;
2. unchanged frozen-evaluator `episode.json` and raw traces under
   `frozen_evaluator/`;
3. `frozen_audit.json`, `release_metrics.json`, `golden_comparison.json`,
   `release_receipt.json`.

Failures keep all existing raw/partial files and add `release_failure.json`.
Golden comparisons require exact gate booleans and use `rtol=0`, `atol=1e-6`
for currency and `1e-8` for other floating metrics, matching the frozen audit's
numerical scale. These are replay checks, **not relaxed scientific gates**.
Elapsed time, gzip header timestamps, and path-dependent serialized record bytes
are not required to equal the historical archive's hashes. Frozen input/model/
source hashes remain exact; newly generated trace hashes are recorded separately.

The parent's `reproducibility/runtime.py` may additionally run
`verify_files(root)` against its overall `release_manifest.json` before importing
this adapter. This adapter is intentionally self-contained and adds its own
scientific-runtime checks. Its explicit map is compatible with the parent's
`resolve_recorded(root, absolute_old_path)` prefix mapping but rejects unlisted
runtime paths rather than requiring arbitrary historical references to resolve.
Calling it from an already-fresh parent replay process is safe; it still uses a
fresh internal worker so frozen short module names cannot leak between studies.
No historical Git tag or private ancestry is needed at runtime.

## Episode-free tests and remaining validation

Run from this directory so every test-created fixture stays here:

```bash
python -B -m unittest -q test_adapter.py

# Optional: copies only sources/configs/protocols into a test-owned checkout,
# imports them, and checks script/PPO/PID controller contracts.
# It never resets an environment, loads rollout data, trains, or runs an episode.
DC2GRID_FROZEN_ROOT=/absolute/path/to/exported-checkout \
  python -B -m unittest -q test_adapter.py test_runtime_imports.py
```

The implementation was verified with episode-free unit, relocation, source
origin, and controller probes, plus full read-only dependency hashing. **No
simulator episode was run in this work.** The parent integration must run the
eleven representative cases in its clean checkout before claiming full rollout
reproducibility. They are coverage examples, not a new inferential cohort.

There are no known missing frozen source/model/data dependencies. Outstanding
integration blockers are (1) export the manifest's exact relative layout and
bytes, (2) provide the matching Python/package runtime, and (3) complete the
parent-owned clean-checkout episode validation. Publication/pushing is outside
this directory's scope.
