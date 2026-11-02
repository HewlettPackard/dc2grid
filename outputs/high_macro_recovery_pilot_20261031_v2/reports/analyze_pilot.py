"""Post-hoc presentation and paired-initialization audit; no policy execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isort: off
from study_common import file_hash, load_json, write_json  # noqa: E402
from register import verify_registration  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
# isort: on


def main():
    protocol = load_json(ROOT / "registration.json")
    verify_registration(protocol)
    output = ROOT / "reports"
    rows = load_json(output / "learning_progress.json")
    pairing = []
    for seed in protocol["macro_seeds"]:
        directories = [
            ROOT / "runs" / f"{variant}_seed_{seed}" for variant in protocol["variants"]
        ]
        initial = [load_json(path / "initialization.json") for path in directories]
        warmup = [load_json(path / "warmup_actions.json") for path in directories]
        for key in (
            "actor_tensor_digest",
            "critic_tensor_digest",
            "critic_target_tensor_digest",
        ):
            if initial[0][key] != initial[1][key]:
                raise AssertionError(f"Unmatched initial parameters: {seed}/{key}")
        np.testing.assert_array_equal(warmup[0]["actions"], warmup[1]["actions"])
        if any(value["exploration_actions"] != 52 for value in warmup):
            raise AssertionError("Incorrect warmup exploration budget")
        pairing.append(
            {
                "seed": seed,
                "initial_parameters_matched": True,
                "all_1024_warmup_actions_matched": True,
                "uniform_exploration_actions_per_arm": 52,
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    milestones = [0, *protocol["milestones"]]
    for variant, color in (("cash", "#2563eb"), ("reserve_potential", "#ea580c")):
        means, counts = [], []
        for step in milestones:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["macro_transitions"] == step
            ]
            complete = len(selected) == 4 and all(
                row.get("cash_delta_vs_strongest_qualified_baseline") is not None
                for row in selected
            )
            means.append(
                float(
                    np.mean(
                        [
                            row["cash_delta_vs_strongest_qualified_baseline"]
                            for row in selected
                        ]
                    )
                )
                if complete
                else np.nan
            )
            counts.append(sum(row.get("qualified_cells") == 24 for row in selected))
        axes[0].plot(milestones, means, marker="o", label=variant, color=color)
        axes[1].plot(milestones, counts, marker="o", label=variant, color=color)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean cash delta vs strongest baseline (USD)")
    axes[0].set_title("All four macro seeds; no qualification filtering")
    axes[1].set_ylabel("Fully qualified seeds (of 4)")
    axes[1].set_yticks(range(5))
    axes[1].set_ylim(-0.1, 4.1)
    axes[1].set_title("Every development cell must qualify")
    for axis in axes:
        axis.set_xlabel("Aggregate macro transitions per fit")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Development pilot only; step 0 is imitation initialization")
    fig.savefig(output / "learning_progress.png", dpi=170)
    plt.close(fig)
    baselines = load_json(ROOT / "baselines/summary.json")
    best = max(
        (row for row in baselines["policies"] if row["all_qualified"]),
        key=lambda row: row["cashflow"],
    )
    final_rows = [
        row for row in rows if row["macro_transitions"] == protocol["total_timesteps"]
    ]
    readback = load_json(output / "readback.json")
    adjudication = load_json(output / "completion_adjudication.json")
    raw_verification = load_json(output / "selected_raw_verification.json")
    report = {
        "registration_id": protocol["registration_id"],
        "implementation_revision": 2,
        "invalidated_v1_baseline_registration": protocol[
            "supersedes_invalidated_registration"
        ],
        "readback": readback,
        "completion_adjudication": adjudication,
        "selected_raw_verification": raw_verification,
        "forward_fix_full_lifecycle_smoke": load_json(
            ROOT.parent
            / "high_macro_recovery_pilot_20261031_v3"
            / "smoke/full_lifecycle/result.json"
        ),
        "execution_summary": {
            "training_and_all_validation_budgets_completed": adjudication[
                "adjudicated_complete_training_and_validation_runs"
            ],
            "native_clean_exit_runs": 0,
            "native_terminal_guard_failures_preserved": 8,
            "forward_fixed_trainer": adjudication["forward_fix"],
        },
        "paired_initialization_checks": pairing,
        "strongest_qualified_nonlearned_macro": best,
        "all_final_checkpoints": final_rows,
        "teacher_baseline_cash_gap_usd": (
            next(
                row["cashflow"]
                for row in baselines["policies"]
                if row["policy_id"] == "fraction_0.98"
            )
            - best["cashflow"]
        ),
        "interpretation": [
            "All budgets completed, but native terminal guard failures are preserved.",
            "The sole protocol delta was SB3 inserting its already-effective default.",
            "Separate adjudication verifies numerical validity, "
            "not clean native exits.",
            "Improving on the .98 initializer is not evidence of beating fraction1.",
            "Returning to the analytic full-residual rule is not a new RL advantage.",
            "Both learning variants share the same complete-trajectory objective.",
            "Development selection can be optimistic; the sealed panel is unexecuted.",
            "The separate one-shot selector study is not pooled with this pilot.",
        ],
        "source_inputs": {
            name: file_hash(output / name)
            for name in (
                "learning_progress.json",
                "selected_models.json",
                "arm_summary.json",
            )
        },
    }
    write_json(output / "final_report.json", report)
    pd.DataFrame(pairing).to_csv(
        output / "paired_initialization_checks.csv", index=False
    )
    print(
        json.dumps(
            {
                "paired_seeds_verified": len(pairing),
                "strongest_baseline": best["policy_id"],
                "final_models": final_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
