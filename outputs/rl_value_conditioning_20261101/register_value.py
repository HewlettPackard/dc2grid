"""Register the adaptive follow-up without rewriting any first-pilot evidence."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ctx import (
    PRIOR,
    PROJECT,
    ROOT,
    file_hash,
    identity,
    read_json,
    verify,
    write_json,
)


def main():
    destination = ROOT / "registration.json"
    if destination.exists():
        raise FileExistsError("The second study registration is immutable")
    prior = read_json(PRIOR / "registration.json")
    rejected = read_json(PRIOR / "extension_decision.json")
    if rejected["extend_both_arms"]:
        raise ValueError(
            "This follow-up is for the rejected first pilot, not its extension"
        )
    references = read_json(PRIOR / "development_references.json")
    fitted = read_json(ROOT / "potential_v1/report.json")
    potential = read_json(ROOT / "potential_v1/model.json")
    fitter_version = subprocess.run(
        [
            "git",
            "show",
            "e82a1c8:outputs/rl_value_conditioning_20261101/fit_value.py",
        ],
        cwd=PROJECT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(fitter_version).hexdigest() != fitted["fitter_sha256"]:
        raise ValueError("The actual potential fitter source is not preserved in Git")
    if file_hash(fitted["model"]["path"]) != fitted["model"]["sha256"]:
        raise ValueError("The fitted potential artifact changed")
    for path, digest in fitted["input_hashes"].items():
        if file_hash(path) != digest:
            raise ValueError("A potential training input changed")
    if (
        fitted["source_registration_id"] != prior["registration_id"]
        or not set(fitted["training_days"]) <= set(prior["train_days"])
        or fitted["date_group_state_rmse"] >= fitted["date_group_constant_rmse"]
        or potential["weights"][3] != 0.0
        or potential["alpha"] != 1.0
    ):
        raise ValueError("Frozen causal potential failed its training-support checks")
    protocol = copy.deepcopy(prior)
    protocol.pop("registration_id")
    input_paths = [
        PRIOR / "registration.json",
        PRIOR / "extension_decision.json",
        PRIOR / "development_references.json",
        Path(references["panel_summary"]["path"]),
        ROOT / "potential_v1/model.json",
        ROOT / "potential_v1/report.json",
        *[Path(path) for path in fitted["input_hashes"]],
        *[
            PRIOR / f"runs/pilot/{variant}_{seed}/{name}.json"
            for variant in prior["variants"]
            for seed in prior["seeds"]
            for name in ("completed", "initialization", "calibration")
        ],
    ]
    protocol.update(
        schema="mixed-prior-causal-value-conditioning-v1",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        parent_registration_id=prior["registration_id"],
        parent_non_advancement_id=rejected["decision_id"],
        frozen_reference_id=references["reference_id"],
        frozen_reference_path=str(PRIOR / "development_references.json"),
        variants=["union_original", "union_conditioned"],
        variant_potentials={
            "union_original": {"kind": "constant", "mean": 0.0},
            "union_conditioned": potential,
        },
        reference_prior={
            "direction": "KL(current pre-tanh Gaussian || frozen BC Gaussian)",
            "reference_weight": 0.1,
            "sigma": 0.01,
            "state_distribution": (
                "equal mixture of fixed demonstration and current replay"
            ),
            "batch": "256 fixed + the same 256 replay states already drawn for SAC",
            "loss": "0.1 * mean(KL on concatenated 512 states)",
            "effective_weight_per_source": 0.05,
            "not_twice_the_registered_total_weight": True,
            "sigma_is_design_prior_not_estimated_uncertainty": True,
        },
        conditioning={
            "reward_at_storage_and_in_environment": (
                "unchanged original physical reward"
            ),
            "reward_when_sampling_for_SAC": (
                "r + gamma*(1-done)*Phi(next_observation) - Phi(observation)"
            ),
            "potential_discount_must_equal_Bellman_discount": True,
            "true_terminal_handling": (
                "zero next potential for both healthy and fault terminals"
            ),
            "potential_inputs": "observed P_BASE, P_FLEX, and remaining clock",
            "backlog_coefficient_explicitly_zero": True,
            "potential_labels": "recorded training-only throughput returns",
            "day_is_a_grouping_label_not_a_potential_input": True,
            "potential_at_evaluation_or_deployment": False,
            "no_new_thermal_experience_claimed": True,
            "policy_invariance_does_not_guarantee_better_learning": True,
            "original_terminal_vs_continuation_preferences_are_preserved": True,
            "large_early_terminal_potential_differences_are_expected": True,
            "backlog_fit_variance": 0.0,
            "backlog_coefficient_is_fixed_not_identified_from_data": True,
            "potential_fitter_prototype_commit": "e82a1c8",
        },
        pairing={
            "initial_parameters_and_raw_warmup": (
                "must match frozen first pilot exactly"
            ),
            "original_arm_critic_calibration": (
                "must match frozen unshaped critic calibration"
            ),
            "conditioned_critic_calibration": (
                "may differ by the declared reward transform"
            ),
            "actor_unchanged_in_both_calibrations": True,
            "historical_controls_are_not_independent_replications": True,
            "new_configuration_chosen_after_observing_Dev_B": True,
            "Dev_B_selection_status": "known and reused, not newly held out",
        },
        shared_final_guard_directory=str(PRIOR),
        old_non_admitted_method_slots_must_be_retained=True,
        source_hashes={
            **prior["source_hashes"],
            **{str(path): file_hash(path) for path in ROOT.glob("*.py")},
        },
        input_hashes={
            **prior["input_hashes"],
            **{str(path): file_hash(path) for path in input_paths},
        },
    )
    protocol["registration_id"] = identity(protocol)
    verify(protocol)
    write_json(destination, protocol)
    print(protocol["registration_id"], flush=True)


if __name__ == "__main__":
    main()
