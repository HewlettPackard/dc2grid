"""Isolated full-lifecycle fixture: test guards, not an efficacy registration."""

from __future__ import annotations

import copy

# isort: off
from study_common import ROOT, identity, load_json, write_json
from prepare import environment_config
import register
from learn import train
# isort: on


def main():
    fixture = ROOT / "smoke/full_lifecycle"
    fixture.mkdir(parents=True, exist_ok=False)
    protocol = copy.deepcopy(load_json(ROOT / "protocol_draft.json"))
    protocol["config_path"] = str(environment_config(fixture / "config", ticks=360))
    protocol.update(
        total_timesteps=16,
        learning_starts=8,
        milestones=[16],
        critic_calibration_updates=4,
    )
    protocol["algorithm"].update(
        train_freq=1,
        batch_size=8,
        buffer_size=128,
        policy_kwargs={"net_arch": [32, 32]},
    )
    protocol["development_cells"] = [
        {"start_day": day, "regd_load_bias": bias, "seed": 88000 + index}
        for index, (day, bias) in enumerate(
            [(16, 0.0), (16, 0.5), (19, 0.0), (19, 0.5)]
        )
    ]
    protocol["registration_id"] = identity(protocol)

    def fixture_guard(value):
        body = {key: item for key, item in value.items() if key != "registration_id"}
        if identity(body) != value["registration_id"]:
            raise AssertionError(
                "Library mutation reached the immutable fixture protocol"
            )

    # Only this isolated smoke replaces production registration/gate IO.
    register.verify_registration = fixture_guard
    register.verify_training_gate = fixture_guard
    before = copy.deepcopy(protocol)
    output = fixture / "run"
    train(protocol, "cash", 812, output, protocol["cpu_groups"][0], "cpu")
    if protocol != before:
        raise AssertionError("Full lifecycle mutated the caller's protocol")
    status = load_json(output / "status.json")
    assert status["state"] == "completed"
    assert status["macro_transitions"] == 16
    assert status["normal_sac_updates"] == 8
    assert status["calibration_updates"] == 4
    write_json(
        fixture / "result.json",
        {
            "passed": True,
            "scope": "small isolated lifecycle fixture, not an efficacy fit",
            "clean_terminal_status": status["state"],
            "caller_protocol_unchanged": True,
            "macro_transitions": 16,
            "normal_sac_updates": 8,
            "critic_only_updates": 4,
            "sealed_policy_evaluations": 0,
        },
    )
    print("Full learner lifecycle completed with an unchanged protocol")


if __name__ == "__main__":
    main()
