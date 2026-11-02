"""Paired, fully counted ReferenceSAC fits with final-step-only primary checkpoints."""

from __future__ import annotations

import argparse
import copy
import shutil
import signal
import traceback
from functools import partial
from pathlib import Path

# isort: off
from study import (
    ROOT,
    FEATURE_CONTRACT,
    cap_cpu,
    file_hash,
    identity,
    read_json,
    write_json,
    verify_registration,
    implementation_commit,
)
from low_core import ReferenceSAC
from low_task import make_worker
from panel import run_panel
from fit_bc import corpus, parameter_digest
from train import Journal, replay_digest
import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
# isort: on


def summarize(episodes):
    if not episodes:
        raise ValueError("Cannot summarize an empty candidate panel")
    result = {
        "cases": len(episodes),
        "qualified_cells": sum(row["qualified"] for row in episodes),
        "complete_cells": sum(row["complete"] for row in episodes),
        "all_qualified": all(row["qualified"] for row in episodes),
        "disqualification_reasons": [],
    }
    if not all(all(row["service_checks"].values()) for row in episodes):
        result["disqualification_reasons"].append("service_or_physical_gate_failure")
    if not all(row["recovery"]["closed"] for row in episodes):
        result["disqualification_reasons"].append("recovery_not_closed")
    if result["complete_cells"] != len(episodes):
        result["disqualification_reasons"].append("incomplete_cells")
        return {
            **result,
            "rmse_kw": None,
            "case_rmse_p75_p90_kw": None,
            "biased_relief_mw": None,
            "cash_usd": None,
            "battery_26h_mwh": None,
            "incomplete_cases_not_silently_averaged": True,
        }
    errors = [row["service"]["tracking_rmse_kw"] for row in episodes]
    cash = [row["cashflow_closed_recovery_usd"] for row in episodes]
    return {
        **result,
        "rmse_kw": float(np.mean(errors)),
        "case_rmse_p75_p90_kw": np.quantile(errors, [0.75, 0.9]).tolist(),
        "biased_relief_mw": float(
            np.mean(
                [
                    row["grid"]["stress_facility_relief_mw"]
                    for row in episodes
                    if row["cell"]["regd_load_bias"] == 0.5
                ]
            )
        ),
        "cash_usd": float(np.mean(cash))
        if all(value is not None for value in cash)
        else None,
        "battery_26h_mwh": float(
            np.mean(
                [
                    row["service"]["battery_throughput_mwh"]
                    + row["recovery"]["ledger"]["battery_throughput_mwh"]
                    for row in episodes
                ]
            )
        )
        if all(row["recovery"]["complete"] for row in episodes)
        else None,
    }


def assess(summary, references, protocol):
    bc, guided = references["bc"], references["guided"]
    ratios = {
        "rmse_ratio_to_bc": summary["rmse_kw"] / bc["rmse_kw"]
        if summary["rmse_kw"] is not None
        else None,
        "rmse_ratio_to_guided": summary["rmse_kw"] / guided["rmse_kw"]
        if summary["rmse_kw"] is not None
        else None,
    }
    if not summary["all_qualified"] or summary["cash_usd"] is None:
        return {
            **ratios,
            "extension_qualified": False,
            "stability_vs_guided": False,
            "increment_vs_bc": False,
        }
    extension = protocol["extension"]
    return {
        **ratios,
        "rmse_delta_to_bc_kw": summary["rmse_kw"] - bc["rmse_kw"],
        "biased_relief_delta_to_bc_mw": summary["biased_relief_mw"]
        - bc["biased_relief_mw"],
        "cash_delta_to_bc_usd": summary["cash_usd"] - bc["cash_usd"],
        "extension_qualified": bool(
            summary["rmse_kw"]
            <= extension["maximum_rmse_ratio_to_own_BC"] * bc["rmse_kw"]
            and summary["rmse_kw"]
            <= extension["maximum_rmse_ratio_to_locked_Guided_same_initializer"]
            * guided["rmse_kw"]
        ),
        "stability_vs_guided": bool(summary["rmse_kw"] <= 0.8 * guided["rmse_kw"]),
        "increment_vs_bc": bool(
            summary["rmse_kw"] <= 0.9 * bc["rmse_kw"]
            and summary["biased_relief_mw"] >= bc["biased_relief_mw"] - 1e-9
            and summary["cash_usd"] >= bc["cash_usd"] - 1e-6
        ),
    }


def fit(args):
    protocol = read_json(args.protocol)
    verify_registration(protocol)
    code_commit = implementation_commit(protocol)
    original_identity = identity(protocol)
    if args.variant not in protocol["variants"] or args.seed not in protocol["seeds"]:
        raise ValueError("Unregistered learner")
    if (
        args.stage != "smoke"
        and args.output.resolve()
        != (ROOT / f"runs/{args.stage}/{args.variant}_{args.seed}").resolve()
    ):
        raise ValueError(
            "Production fits must use their exact registered slot directories"
        )
    if len(args.cpus) != protocol["n_envs"] + 1 or len(set(args.cpus)) != len(
        args.cpus
    ):
        raise ValueError(
            "One distinct learner core plus four distinct environment cores required"
        )
    references = None
    if args.stage != "smoke":
        reference_path = ROOT / "development_references.json"
        reference = read_json(reference_path)
        body = {key: value for key, value in reference.items() if key != "reference_id"}
        if (
            identity(body) != reference["reference_id"]
            or reference["registration_id"] != protocol["registration_id"]
            or file_hash(reference["panel_summary"]["path"])
            != reference["panel_summary"]["sha256"]
        ):
            raise ValueError("Matched development reference lock changed")
        references = reference["per_initializer"][str(args.seed)]
    if args.stage == "extended":
        decision = read_json(ROOT / "extension_decision.json")
        decision_body = {
            key: value for key, value in decision.items() if key != "decision_id"
        }
        if (
            identity(decision_body) != decision["decision_id"]
            or not decision["extend_both_arms"]
            or decision["registration_id"] != protocol["registration_id"]
        ):
            raise ValueError("The registered extension criterion has not passed")
        pilot_records = read_json(
            ROOT / f"runs/pilot/{args.variant}_{args.seed}/validation_summary.json"
        )
        if [row["timesteps"] for row in pilot_records] != protocol["stage_milestones"][
            "pilot"
        ]:
            raise ValueError(
                "The complete original pilot milestone inventory is required"
            )
    cap_cpu(args.cpus[0])
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name, digest in protocol["source_hashes"].items():
        path = Path(name)
        if path.is_relative_to(ROOT):
            target = source / path.name
            shutil.copy2(path, target)
            if file_hash(target) != digest:
                raise AssertionError("Fit source snapshot changed")
    write_json(
        args.output / "manifest.json",
        {
            "registration_id": protocol["registration_id"],
            "stage": args.stage,
            "variant": args.variant,
            "initializer_seed": args.seed,
            "online_seed": protocol["online_seed_by_initializer"][str(args.seed)],
            "cpus": args.cpus,
            "source_hashes": protocol["source_hashes"],
            "references": references,
            "implementation_commit": code_commit,
        },
    )
    callback = Journal(args.output)
    online_seed = protocol["online_seed_by_initializer"][str(args.seed)]
    vector = SubprocVecEnv(
        [
            partial(make_worker, protocol, online_seed, rank, args.cpus[rank + 1])
            for rank in range(protocol["n_envs"])
        ],
        start_method="spawn",
    )
    succeeded = False
    try:
        artifact = protocol["initializers"][str(args.seed)]
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("BC initialization artifact changed")
        model = ReferenceSAC.load(
            artifact["path"],
            env=vector,
            device=args.device,
            seed=online_seed,
            **copy.deepcopy(protocol["algorithm"]),
        )
        if (
            model.c2g_context["feature_contract"] != FEATURE_CONTRACT
            or model.policy_kwargs["net_arch"] != protocol["architecture"]
            or model.learning_starts != protocol["learning_starts"]
        ):
            raise ValueError("Initializer learning or deployment contract changed")
        model.set_logger(configure(str(args.output / "metrics"), ["csv"]))
        x, y, _, _, corpus_info = corpus(Path(protocol["corpus_path"]), "history")
        model.attach_demonstrations(x, y, args.seed)
        model.freeze_reference()
        model.configure_reference(args.variant, protocol["reference_weight"])
        model.set_diagnostic_batch(args.seed)
        model.c2g_context = copy.deepcopy(
            {
                **model.c2g_context,
                "kind": "standalone_reference_policy_regularized_SAC",
                "variant": args.variant,
                "registration_id": protocol["registration_id"],
                "initialization": artifact,
                "online_seed": online_seed,
                "reference_digest": model.reference_digest,
                "reference_sigma": protocol["reference_sigma"],
                "reference_sigma_is_design_prior": True,
            }
        )
        initial = {
            "actor": parameter_digest(model.actor),
            "critic": parameter_digest(model.critic),
            "critic_target": parameter_digest(model.critic_target),
            "reference": model.reference_digest,
            "initializer": artifact,
            "corpus": corpus_info,
            "reference_weight": model.reference_weight,
            "legacy_bc_weight": model.bc_weight,
        }
        if initial["actor"] != protocol["initializer_actor_digests"][str(args.seed)]:
            raise AssertionError("Actor is not the exact registered BC initialization")
        initial_policy = model.policy_diagnostics()
        initial["policy_at_initialization"] = initial_policy
        np.testing.assert_allclose(
            initial_policy["latent_std_mean"], protocol["reference_sigma"], rtol=1e-6
        )
        model.learn(
            protocol["learning_starts"], callback=callback, reset_num_timesteps=False
        )
        if (
            model.actor_updates
            or model._n_updates
            or parameter_digest(model.actor) != initial["actor"]
        ):
            raise AssertionError("Collection-only warmup modified parameters")
        initial["warmup_replay"] = replay_digest(model.replay_buffer)
        initial["warmup_transitions"] = model.num_timesteps
        initial["diagnostic_constants"] = vector.env_method("diagnostic_constants")
        model.snapshot_warmup_probe()
        write_json(args.output / "initialization.json", initial)
        calibration = model.calibrate_critics(protocol["critic_calibration_updates"])
        write_json(
            args.output / "calibration.json",
            {
                **calibration,
                "prior_scale_probe": model.prior_scale_probe(),
                "diagnostics": model.diagnostics(),
                "reference_weight": model.reference_weight,
                "critic_digest": parameter_digest(model.critic),
                "critic_target_digest": parameter_digest(model.critic_target),
            },
        )
        model.logger.dump(step=model.num_timesteps)
        model.last_training_dump_step = model.num_timesteps
        records = []
        for target in protocol["stage_milestones"][args.stage]:
            model.learn(
                target - model.num_timesteps,
                callback=callback,
                reset_num_timesteps=False,
            )
            expected_actor = target - protocol["learning_starts"]
            if (
                model.num_timesteps != target
                or model.actor_updates != expected_actor
                or model._n_updates
                != expected_actor + protocol["critic_calibration_updates"]
                or model.bc_weight != 0
                or model.reference_weight != protocol["reference_weight"]
                or identity(protocol) != original_identity
            ):
                raise AssertionError(
                    "Transition, optimizer, prior or protocol accounting changed"
                )
            model.verify_reference()
            if model.last_training_dump_step != target:
                model.logger.dump(step=target)
                model.last_training_dump_step = target
            model.c2g_context.update(
                online_transitions=target,
                sac_actor_updates=model.actor_updates,
                critic_updates=model._n_updates,
                reference_weight=model.reference_weight,
                reference_state_distribution=model.reference_state_distribution,
            )
            checkpoint = args.output / f"step_{target}.zip"
            model.save(checkpoint)
            model.save_replay_buffer(args.output / f"step_{target}_replay.pkl")
            policy = {
                "id": f"{args.variant}_{args.seed}_step_{target}",
                "family": args.variant,
                "training_seed": args.seed,
                "hardware": "history_sac",
                "macro": "rule",
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": file_hash(checkpoint),
                },
            }
            selected_cells = protocol[
                "smoke_cells" if args.stage == "smoke" else "development_cells"
            ]
            panel = run_panel(
                protocol,
                [policy],
                selected_cells,
                args.output / f"validation_{target}",
                args.cpus[1:],
                role="training_smoke" if args.stage == "smoke" else "development",
            )
            summary = summarize(panel["episodes"])
            record = {
                "timesteps": target,
                "checkpoint": policy["checkpoint"],
                "actor_digest": parameter_digest(model.actor),
                "critic_digest": parameter_digest(model.critic),
                "diagnostics": model.diagnostics(),
                **summary,
                **(
                    assess(summary, references, protocol)
                    if references is not None
                    else {}
                ),
            }
            if (
                args.stage == "extended"
                and target in protocol["stage_milestones"]["pilot"]
            ):
                previous = [row for row in pilot_records if row["timesteps"] == target]
                if len(previous) != 1:
                    raise ValueError(
                        "Repeated prefix has no unique original pilot milestone"
                    )
                previous = previous[0]
                record["repeated_prefix_actor_exact"] = (
                    record["actor_digest"] == previous["actor_digest"]
                )
                record["repeated_prefix_critic_exact"] = (
                    record["critic_digest"] == previous["critic_digest"]
                )
            records.append(record)
            write_json(args.output / "validation_summary.json", records)
            print(args.variant, args.seed, target, summary, flush=True)
        prefixes = vector.env_method("training_prefix")
        write_json(args.output / "training_prefixes.json", prefixes)
        diagnostics = [
            vector.env_method(
                "save_diagnostics",
                str(args.output / f"training_worker_{rank}.npz"),
                indices=rank,
            )[0]
            for rank in range(protocol["n_envs"])
        ]
        if sum(item["transitions"] for item in diagnostics) != model.num_timesteps:
            raise AssertionError("Per-transition diagnostic inventory is incomplete")
        model.save_replay_buffer(args.output / "final_replay.pkl")
        verify_registration(protocol)
        final = records[-1]
        prefix = [row for row in records if "repeated_prefix_actor_exact" in row]
        divergent = [
            row["timesteps"]
            for row in prefix
            if not row["repeated_prefix_actor_exact"]
            or not row["repeated_prefix_critic_exact"]
        ]
        if args.stage == "extended" and len(prefix) != 3:
            raise AssertionError("Missing a repeated-prefix comparison")
        write_json(
            args.output / "selection.json",
            {
                "primary": final if final["all_qualified"] else None,
                "rule": protocol["acceptance"]["primary_checkpoint"],
                "all_milestones_retained": True,
            },
        )
        write_json(
            args.output / "completed.json",
            {
                "status": "completed",
                "registration_id": protocol["registration_id"],
                "stage": args.stage,
                "variant": args.variant,
                "seed": args.seed,
                "online_transitions": model.num_timesteps,
                "sac_actor_updates": model.actor_updates,
                "critic_updates": model._n_updates,
                "final": final,
                "source_hashes": protocol["source_hashes"],
                "transition_diagnostics": diagnostics,
                "smoke_only": args.stage == "smoke",
                "registration_unchanged": True,
                "repeated_prefix": {
                    "applicable": args.stage == "extended",
                    "parameter_exact": not divergent if prefix else None,
                    "first_divergent_milestone": min(divergent) if divergent else None,
                    "authoritative_prefix_results": (
                        "original pilot used for the extension decision"
                    ),
                    "no_additional_replication_claimed": True,
                },
            },
        )
        succeeded = True
    finally:
        vector.close()
        if not succeeded:
            write_json(
                args.output / "failure.json",
                {"status": "runtime_failure", "traceback": traceback.format_exc()},
            )


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise InterruptedError(f"Learner interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--stage", choices=("smoke", "pilot", "extended"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    fit(parser.parse_args())
