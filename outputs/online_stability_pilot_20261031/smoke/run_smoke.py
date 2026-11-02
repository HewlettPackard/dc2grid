"""Bounded CPU wiring check; not an outcome from the registered online study."""

import copy
import json
from pathlib import Path

import yaml

from baselines.online_stability import ROOT, run_job
from c2g_env.experiments.identity import configuration_id, file_fingerprint
from scripts.launch_online_stability import checkpoint_inputs


def main():
    output = Path(__file__).resolve().parent
    baseline = Path(
        "/workspace/IBM/temp/dc2grid-repos/dc2grid/outputs/corrected_sac_8gpu_20261030"
    )
    protocol = yaml.safe_load((baseline / "protocol.yaml").read_text())
    config = yaml.safe_load((ROOT / "c2g_env/config.yaml").read_text())
    config["global"]["episode_ticks"] = 12
    config_path = output / "environment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    protocol["config_path"] = str(config_path)
    registration = {
        "source_files": {},
        "protocol_file": file_fingerprint(config_path),
        "environment_protocol": copy.deepcopy(protocol),
        "scope": "SMOKE ONLY: 12-tick episodes, not a performance evaluation",
        "pilot_protocol": {
            "online_transitions": 96, "critic_calibration_steps": 4,
            "evaluation_interval": 48, "checkpoint_interval": 48,
            "status_interval": 24, "smoke_only": True,
        },
    }
    registration["registration_id"] = configuration_id(registration)
    (output / "registration.json").write_text(json.dumps(registration, indent=2))
    job = {
        "id": "actual_checkpoint_smoke", "base_seed": 1006, "stream": 0,
        "arm": "critic_restore",
        "inputs": checkpoint_inputs(baseline, 1006, 1000008, 500004),
    }
    run_job(
        registration, job, output / "run", [28, 29, 30, 31, 60, 61, 62], "cpu",
    )
    status = json.loads((output / "run/status.json").read_text())
    final = json.loads((output / "run/validation/online_96.json").read_text())
    assert status["new_online_transitions"] == 96
    assert status["new_normal_sac_updates"] == 96
    assert final["new_normal_sac_updates"] == 96
    assert len(final["cells"]) == 4
    assert status["actual_replay_position"] == status["expected_replay_position"]
    print("PASS: actual 1M checkpoint/replay, six workers, 96 new online transitions")
    print("PASS: 96 normal SAC updates and post-update final evaluation")
    print("SMOKE ONLY; no claims about policy improvement")


if __name__ == "__main__":
    main()
