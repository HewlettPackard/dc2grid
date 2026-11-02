"""Fit a frozen potential from completed training trajectories, never Dev-B."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

# isort: off
from ctx import ROOT, PRIOR, read_json, write_json, file_hash
import numpy as np
# isort: on


def returns_to_go(rewards, gamma):
    result = np.zeros(len(rewards) + 1, dtype=np.float64)
    for step in range(len(rewards) - 1, -1, -1):
        result[step] = rewards[step] + gamma * result[step + 1]
    return result[:-1]


def factors(clock, gamma):
    remaining = np.clip(np.rint(np.asarray(clock, dtype=np.float64) * 17280), 0, 17280)
    return -np.expm1(remaining * np.log(gamma)) / (1 - gamma)


def fit_coefficients(features, target, factor):
    design = features * factor[:, None]
    weights, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    return weights, int(rank)


def prediction(features, factor, weights):
    return factor * np.clip(features @ weights, 0, 1)


def rmse(value, target):
    return float(np.sqrt(np.mean((value - target) ** 2)))


def fit_value(output):
    if output.exists():
        raise FileExistsError("Never replace a frozen potential fit")
    output.mkdir(parents=True)
    protocol = read_json(PRIOR / "registration.json")
    gamma = protocol["algorithm"]["gamma"]
    by_day, input_hashes = {}, {}
    duplicates = 0
    for seed in protocol["seeds"]:
        directory = PRIOR / f"runs/pilot/fixed_reference_kl_{seed}"
        complete = read_json(directory / "completed.json")
        if complete["online_transitions"] != 69120 or complete["status"] != "completed":
            raise ValueError(
                "Potential labels require the complete first training pilot"
            )
        path = directory / "final_replay.pkl"
        input_hashes[str(path)] = file_hash(path)
        with path.open("rb") as handle:
            buffer = pickle.load(handle)
        if buffer.full or buffer.pos != 17280 or buffer.n_envs != 4:
            raise ValueError("Unexpected original replay layout")
        for rank in range(4):
            path = directory / f"training_worker_{rank}.npz"
            input_hashes[str(path)] = file_hash(path)
            with np.load(path) as archive:
                columns = list(archive["columns"])
                data = archive["values"]
                alpha = float(archive["reward_alpha"])
            if alpha != 1 or len(data) != 17280:
                raise ValueError("Potential scale or original training horizon changed")
            day_values = data[:, columns.index("start_day")]
            if len(np.unique(day_values)) != 1:
                raise ValueError("Expected one complete training episode per worker")
            day = int(day_values[0])
            if day not in protocol["train_days"]:
                raise ValueError("A potential label came from outside training support")
            np.testing.assert_array_equal(
                data[:, columns.index("step")], np.arange(1, 17281)
            )
            if data[:, columns.index("physical_terminated")].any():
                raise ValueError(
                    "Faulted trajectory cannot label a healthy continuation"
                )
            np.testing.assert_array_equal(
                buffer.actions[: buffer.pos, rank].shape, (17280, 3)
            )
            np.testing.assert_array_equal(
                buffer.observations[: buffer.pos, rank, 18], np.ones(17280)
            )
            component_indices = [
                index
                for index, column in enumerate(columns)
                if column.startswith("reward_")
            ]
            original_reward = data[:, component_indices].sum(axis=1).astype(np.float32)
            np.testing.assert_array_equal(
                original_reward, buffer.rewards[: buffer.pos, rank]
            )
            observations = buffer.observations[: buffer.pos, rank]
            feature = np.column_stack(
                (
                    np.ones(17280),
                    observations[:, [3, 4, 16]],
                )
            ).astype(np.float64)
            throughput = data[:, columns.index("reward_throughput")]
            clock = observations[:, 23].astype(np.float64)
            np.testing.assert_allclose(
                clock, 1 - np.arange(17280) / 17280, atol=1e-7, rtol=0
            )
            if day in by_day:
                prior = by_day[day]
                np.testing.assert_array_equal(prior["features"], feature)
                np.testing.assert_array_equal(prior["throughput"], throughput)
                np.testing.assert_array_equal(prior["clock"], clock)
                duplicates += 1
            else:
                by_day[day] = {
                    "features": feature,
                    "throughput": throughput.copy(),
                    "clock": clock,
                }
    dates, x, target, factor, reward = [], [], [], [], []
    for day in sorted(by_day):
        row = by_day[day]
        dates.extend([day] * 17280)
        x.append(row["features"])
        reward.append(row["throughput"])
        target.append(returns_to_go(row["throughput"], gamma))
        factor.append(factors(row["clock"], gamma))
    dates = np.asarray(dates)
    x, target, factor, reward = map(np.concatenate, (x, target, factor, reward))
    oof, constant_oof = np.empty_like(target), np.empty_like(target)
    folds = []
    for day in sorted(by_day):
        held = dates == day
        coefficients, rank = fit_coefficients(x[~held], target[~held], factor[~held])
        mean = float(
            np.dot(factor[~held], target[~held]) / np.dot(factor[~held], factor[~held])
        )
        oof[held] = prediction(x[held], factor[held], coefficients)
        constant_oof[held] = factor[held] * mean
        folds.append(
            {
                "day": day,
                "rank": rank,
                "coefficients": coefficients.tolist(),
                "state_rmse": rmse(oof[held], target[held]),
                "constant_rmse": rmse(constant_oof[held], target[held]),
            }
        )
    coefficients, rank = fit_coefficients(x, target, factor)
    configuration = {
        "kind": "linear_causal",
        "weights": coefficients.tolist(),
        "alpha": 1.0,
    }
    report = {
        "source_registration_id": protocol["registration_id"],
        "fitter_sha256": file_hash(__file__),
        "input_hashes": input_hashes,
        "training_days": sorted(by_day),
        "unique_training_transitions": len(x),
        "duplicate_day_trajectories_verified_bitwise": duplicates,
        "features": ["constant", "native_obs_3", "native_obs_4", "native_obs_16"],
        "clock": "native history observation 23, rounded to remaining service ticks",
        "label": "recorded training-only discounted throughput return",
        "day_used_only_for_grouping_not_a_potential_input": True,
        "coefficient_rank": rank,
        "folds": folds,
        "discount": gamma,
        "configuration": configuration,
        "raw_throughput_mean": float(reward.mean()),
        "raw_throughput_sd": float(reward.std()),
        "throughput_return_mean": float(target.mean()),
        "throughput_return_sd": float(target.std()),
        "date_group_state_rmse": rmse(oof, target),
        "date_group_constant_rmse": rmse(constant_oof, target),
        "no_new_rollouts": True,
        "training_target_conditioning_not_new_thermal_experience": True,
    }
    write_json(output / "model.json", configuration)
    report["model"] = {
        "path": str(output / "model.json"),
        "sha256": file_hash(output / "model.json"),
    }
    write_json(output / "report.json", report)
    np.savez_compressed(
        output / "training_labels.npz",
        dates=dates,
        features=x,
        targets=target,
        factors=factor,
        throughput=reward,
        out_of_fold=oof,
        constant_out_of_fold=constant_oof,
    )
    print(
        {
            key: value
            for key, value in report.items()
            if key
            not in (
                "input_hashes",
                "folds",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "potential_v1")
    args = parser.parse_args()
    fit_value(args.output)
