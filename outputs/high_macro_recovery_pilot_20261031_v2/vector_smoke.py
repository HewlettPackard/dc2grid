"""Exercise spawned one-shot workers, including an intentionally shorter fixture."""

from __future__ import annotations

import copy
from functools import partial

# isort: off
from study_common import ROOT, load_json, write_json
from prepare import environment_config
from macro_task import MacroQuantityTask
from evaluate import (
    ConstantQuantity,
    OneShotEvaluation,
    cap_worker,
    evaluate_suite,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
# isort: on


def worker(protocol, rank, cpus, shorter_config):
    cap_worker(cpus[rank + 1])
    local = copy.deepcopy(protocol)
    if rank == 0:
        local["config_path"] = shorter_config
    return OneShotEvaluation(
        MacroQuantityTask(
            local, 0.0003, cell=local["development_cells"][rank], record_trace=True
        )
    )


def main():
    protocol = copy.deepcopy(load_json(ROOT / "protocol_draft.json"))
    output = ROOT / "smoke/vector"
    output.mkdir(parents=True, exist_ok=False)
    protocol["config_path"] = str(environment_config(output / "long", ticks=360))
    shorter = str(environment_config(output / "short", ticks=180))
    protocol["development_cells"] = [
        {"start_day": day, "regd_load_bias": bias, "seed": 440051 + index}
        for index, (day, bias) in enumerate(
            [
                (16, 0.0),
                (16, 0.5),
                (19, 0.0),
                (19, 0.5),
                (16, 0.0),
                (16, 0.5),
                (19, 0.0),
                (19, 0.5),
            ]
        )
    ]
    cpus = protocol["cpu_groups"][0]
    vector = SubprocVecEnv(
        [partial(worker, protocol, rank, cpus, shorter) for rank in range(4)],
        start_method="spawn",
    )
    try:
        report = evaluate_suite(
            ConstantQuantity(0.98),
            vector,
            protocol,
            output / "evaluation",
            {"kind": "mixed_horizon_fixture_not_efficacy", "registration_id": "smoke"},
        )
        assert report["recorded_cells"] == 8
        for index, row in enumerate(report["cells"]):
            assert row["decisions"] == (1 if index % 4 == 0 else 2)
            assert row["service_complete"] and row["recovery"]["complete"]
        assert report["cells"][0]["suppressed_padding_steps"] == 1
        assert report["cells"][4]["suppressed_padding_steps"] == 1
        assert all(
            row["hardware_predictions"] == row["service"]["steps"]
            for row in report["cells"]
        )
        write_json(
            ROOT / "smoke/vector_smoke.json",
            {
                "passed": True,
                "recorded_cases": 8,
                "physical_macro_decisions": sum(
                    row["decisions"] for row in report["cells"]
                ),
                "padding_steps": sum(
                    row["suppressed_padding_steps"] for row in report["cells"]
                ),
                "hidden_extra_rollouts": 0,
                "only_training_calendar_fixtures_used": True,
            },
        )
        print("Spawned evaluation and explicit padding smoke passed")
    finally:
        vector.close()


if __name__ == "__main__":
    main()
