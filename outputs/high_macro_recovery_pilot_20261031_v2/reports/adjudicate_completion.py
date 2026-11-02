"""Read-only adjudication of the terminal SB3 default-insertion guard failure."""

from __future__ import annotations

import inspect
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isort: off
from study_common import cell_key, file_hash, load_json, write_json  # noqa: E402
from register import verify_registration  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm  # noqa: E402
import torch  # noqa: E402
# isort: on


def differences(a, b, prefix=""):
    result = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in a:
                result.append({"path": path, "operation": "added", "value": b[key]})
            elif key not in b:
                result.append({"path": path, "operation": "removed", "value": a[key]})
            else:
                result.extend(differences(a[key], b[key], path))
    elif a != b:
        result.append({"path": prefix, "operation": "changed", "old": a, "new": b})
    return result


def main():
    registered = load_json(ROOT / "registration.json")
    verify_registration(registered)
    default = inspect.signature(SAC.__init__).parameters["use_sde"].default
    if default is not False:
        raise AssertionError("The supposedly implicit default is not False")
    source = inspect.getsource(OffPolicyAlgorithm.__init__)
    if 'self.policy_kwargs["use_sde"] = self.use_sde' not in source:
        raise AssertionError("Expected constructor mutation was not found")
    expected_change = [
        {
            "path": "algorithm.policy_kwargs.use_sde",
            "operation": "added",
            "value": False,
        }
    ]
    rows = []
    expected_cells = {cell_key(cell) for cell in registered["development_cells"]}
    for variant in registered["variants"]:
        for seed in registered["macro_seeds"]:
            run = ROOT / "runs" / f"{variant}_seed_{seed}"
            failure = load_json(run / "failure.json")
            if (
                failure["error_type"] != "RuntimeError"
                or failure["error"] != "Registered macro protocol changed"
                or failure["macro_transitions"] != 20480
            ):
                raise AssertionError(f"Unexpected native failure: {run}")
            with zipfile.ZipFile(run / "final_model.zip") as archive:
                model_data = json.loads(archive.read("data"))
                actor_optimizer = torch.load(
                    io.BytesIO(archive.read("actor.optimizer.pth")),
                    map_location="cpu",
                    weights_only=True,
                )
                critic_optimizer = torch.load(
                    io.BytesIO(archive.read("critic.optimizer.pth")),
                    map_location="cpu",
                    weights_only=True,
                )
            diff = differences(registered, model_data["c2g_context"]["protocol"])
            if diff != expected_change or model_data["use_sde"] is not False:
                raise AssertionError(
                    f"Not solely an unchanged default insertion: {run}"
                )
            if (
                model_data["num_timesteps"] != 20480
                or model_data["_n_updates"] != 19968
            ):
                raise AssertionError("Saved optimizer/training budgets are incomplete")
            actor_steps = {
                int(state["step"].item()) for state in actor_optimizer["state"].values()
            }
            critic_steps = {
                int(state["step"].item())
                for state in critic_optimizer["state"].values()
            }
            if actor_steps != {19456} or critic_steps != {19968}:
                raise AssertionError("Adam states disagree with the update budget")
            validation_count = 0
            for step in [0, *registered["milestones"]]:
                report = load_json(run / f"validation/step_{step}/summary.json")
                if {
                    cell_key(e["condition"]) for e in report["cells"]
                } != expected_cells:
                    raise AssertionError("Missing or replaced development evaluation")
                if len(report["cells"]) != 24:
                    raise AssertionError("Incorrect validation cardinality")
                validation_count += 24
                if step:
                    checkpoint = run / f"checkpoints/step_{step}"
                    metadata = load_json(checkpoint / "metadata.json")
                    if (
                        not metadata["after_all_optimizer_updates"]
                        or file_hash(checkpoint / "model.zip")
                        != metadata["model_sha256"]
                        or file_hash(checkpoint / "replay.pkl")
                        != metadata["replay_sha256"]
                    ):
                        raise AssertionError("Post-update checkpoint evidence changed")
            initial_workers = load_json(run / "initial_worker_runtime.json")
            final_workers = load_json(run / "final_worker_runtime.json")
            if len(initial_workers) != 4 or len(final_workers) != 4:
                raise AssertionError("Missing worker integrity checks")
            for first, last in zip(initial_workers, final_workers):
                if (
                    first["hardware_tensor_digest"] != last["hardware_tensor_digest"]
                    or last["torch_threads"] != 1
                    or last["hardware_device"] != "cpu"
                    or last["cuda_initialized"]
                ):
                    raise AssertionError("Frozen hardware/resource invariant failed")
            selected = load_json(run / "best_model/selection.json")
            if file_hash(run / "best_model/model.zip") != selected["checkpoint_sha256"]:
                raise AssertionError("Selected checkpoint changed")
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "native_exit_record_preserved": "runtime_failure",
                    "training_and_all_validation_budgets_completed": True,
                    "macro_transitions": 20480,
                    "normal_sac_updates": 19456,
                    "critic_only_updates": 512,
                    "supervised_initialization_updates": 256,
                    "development_rollouts": validation_count,
                    "only_protocol_delta": diff,
                    "effective_sde_setting_matches_registered_default": True,
                    "frozen_hardware_workers_verified": 4,
                    "final_model_sha256": file_hash(run / "final_model.zip"),
                }
            )
    verdict = {
        "registration_id": registered["registration_id"],
        "classification": (
            "Completed numerical study with a verified terminal "
            "provenance-guard false positive"
        ),
        "native_clean_exit_runs": 0,
        "native_failure_receipts_preserved": 8,
        "adjudicated_complete_training_and_validation_runs": len(rows),
        "macro_transitions": sum(row["macro_transitions"] for row in rows),
        "normal_sac_updates": sum(row["normal_sac_updates"] for row in rows),
        "critic_only_updates": sum(row["critic_only_updates"] for row in rows),
        "supervised_initialization_updates": sum(
            row["supervised_initialization_updates"] for row in rows
        ),
        "development_rollouts": sum(row["development_rollouts"] for row in rows),
        "rows": rows,
        "cause": (
            "A shared policy_kwargs dictionary was passed to SB3. Its constructor "
            "inserted use_sde=False, the already-effective default. Only the live "
            "protocol object changed; registered files, physics and effective "
            "algorithm parameters did not. The terminal guard ran after the final "
            "optimizer block, model save, validation and worker integrity checks."
        ),
        "disposition": (
            "Numerical development results may be analyzed with this qualification. "
            "Do not relabel native failures as clean exits. No large retraining "
            "rerun or sealed evaluation was performed to repair bookkeeping."
        ),
        "forward_fix": str(
            ROOT.parent / "high_macro_recovery_pilot_20261031_v3/learn.py"
        ),
        "sealed_policy_rollouts": 0,
        "adjudicator_sha256": file_hash(__file__),
    }
    verify_registration(registered)
    write_json(ROOT / "reports/completion_adjudication.json", verdict)
    print(
        json.dumps(
            {key: value for key, value in verdict.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
