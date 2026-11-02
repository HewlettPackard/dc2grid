"""Training-date-only forecasts of the next quarter-hour's signed RegD mean."""

from __future__ import annotations

import argparse
from pathlib import Path

# isort: off
from study import ROOT, file_hash, read_json, write_json, verify_registration
import numpy as np
# isort: on

FEATURES = (
    "current_regd",
    "previous_window_regd_mean",
    "current_lmp_over_200",
    "current_grid_load_normalized",
    "sin_clock",
    "cos_clock",
)


def causal_features(native_observations, previous_mean, elapsed_seconds):
    observation = np.asarray(native_observations, dtype=np.float64)
    clock = 2 * np.pi * np.asarray(elapsed_seconds) / 86400
    result = np.stack(
        (
            observation[..., 6],
            np.asarray(previous_mean),
            observation[..., 7],
            observation[..., 8],
            np.sin(clock),
            np.cos(clock),
        ),
        axis=-1,
    )
    if result.shape[-1] != len(FEATURES) or not np.isfinite(result).all():
        raise ValueError("Invalid causal forecast features")
    return result


def episode_examples(history):
    if history.shape != (17280, 24) or not np.isfinite(history).all():
        raise ValueError("Forecast corpus must contain a complete pre-action day")
    expected_clock = 1 - np.arange(17280) / 17280
    np.testing.assert_allclose(history[:, 23], expected_clock, atol=1e-7, rtol=0)
    windows = history.reshape(96, 180, 24)
    labels = windows[:, :, 6].mean(axis=1, dtype=np.float64)
    previous = np.concatenate(([0.0], labels[:-1]))
    features = causal_features(windows[:, 0, :], previous, np.arange(96) * 900)
    return features, labels


def fit_ridge(features, labels, penalty):
    if penalty <= 0 or not np.isfinite(penalty):
        raise ValueError("Ridge penalty must be strictly positive")
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.0
    normalized = (features - center) / scale
    label_mean = float(labels.mean())
    coefficients = np.linalg.solve(
        normalized.T @ normalized + penalty * np.eye(features.shape[1]),
        normalized.T @ (labels - label_mean),
    )
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": label_mean,
        "constant_columns": np.flatnonzero(constant).tolist(),
    }


def predict_ridge(model, features):
    features = np.asarray(features, dtype=np.float64)
    prediction = (features - np.array(model["center"])) / np.array(
        model["scale"]
    ) @ np.array(model["coefficients"]) + model["intercept"]
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite frozen forecast")
    return np.clip(prediction, -1, 1)


def metrics(predictions, labels):
    error = np.asarray(predictions) - np.asarray(labels)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(error.mean()),
    }


def fit_forecast(protocol, output):
    verify_registration(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = read_json(Path(protocol["corpus_path"]) / "summary.json")
    x, y, dates, biases, hashes = [], [], [], [], {}
    for row in summary["results"]:
        day = row["condition"]["start_day"]
        if not row["complete"] or day not in protocol["train_days"]:
            raise ValueError("Forecast data outside completed training support")
        path = Path(row["artifact"])
        if file_hash(path) != row["artifact_sha256"]:
            raise ValueError("Forecast corpus artifact changed")
        hashes[str(path)] = row["artifact_sha256"]
        with np.load(path) as data:
            features, labels = episode_examples(data["history"])
        x.append(features)
        y.append(labels)
        dates.extend([day] * len(labels))
        biases.extend([row["condition"]["regd_load_bias"]] * len(labels))
    x, y, dates, biases = map(
        np.asarray, (np.concatenate(x), np.concatenate(y), dates, biases)
    )
    if len(x) != 8640 or set(dates) != set(protocol["train_days"]):
        raise ValueError("Expected all 90 registered demonstrations and 8640 windows")
    penalty = protocol["high_level"]["ridge_penalty"]
    out_of_fold = np.empty_like(y)
    simple_predictions = {size: np.empty_like(y) for size in (1, 2)}
    folds = []
    for day in sorted(set(dates)):
        held = dates == day
        model = fit_ridge(x[~held], y[~held], penalty)
        out_of_fold[held] = predict_ridge(model, x[held])
        for size, prediction in simple_predictions.items():
            simple_model = fit_ridge(x[~held, :size], y[~held], penalty)
            prediction[held] = predict_ridge(simple_model, x[held, :size])
        folds.append(
            {
                "validation_day": int(day),
                "validation_rows": int(held.sum()),
                "training_days": sorted(int(value) for value in set(dates[~held])),
                "six_feature_coefficients": model["coefficients"],
            }
        )
    rho = protocol["high_level"]["decay_rho"]
    decay = (1 - rho**180) / (180 * (1 - rho))
    predictions = {
        "zero": np.zeros_like(y),
        "current_signal": x[:, 0],
        "previous_window": x[:, 1],
        "linear_decay_not_exact_biased_clipped_AR_forecast": decay * x[:, 0],
        "date_group_cross_validated_ridge": out_of_fold,
        "date_group_one_feature_ridge_diagnostic": simple_predictions[1],
        "date_group_two_feature_ridge_diagnostic": simple_predictions[2],
    }
    report = {
        "registration_id": protocol["registration_id"],
        "rows": len(x),
        "date_groups": len(set(dates)),
        "features": FEATURES,
        "folds": folds,
        "ridge_penalty": penalty,
        "causal_inputs_only": True,
        "future_truth_used_only_as_training_labels": True,
        "scope": (
            "Forecast accuracy is not a cash or headroom bound. Dates are groups, "
            "not assumed independent. The deployed cash screen assumes current-price "
            "persistence; actual variable-price cash requires complete paired rollouts."
        ),
        "overall": {key: metrics(value, y) for key, value in predictions.items()},
        "by_bias": {
            str(bias): {
                key: metrics(value[biases == bias], y[biases == bias])
                for key, value in predictions.items()
            }
            for bias in sorted(set(biases))
        },
        "by_date": {
            str(day): {
                key: metrics(value[dates == day], y[dates == day])
                for key, value in predictions.items()
            }
            for day in sorted(set(dates))
        },
        "input_hashes": hashes,
    }
    model = {
        "schema": "causal-six-feature-regd-mean-ridge-v1",
        "training_registration_id": protocol["registration_id"],
        "features": FEATURES,
        "target": "mean(history[t:t+180,6]) from pre-action service records",
        "ridge_penalty": penalty,
        **fit_ridge(x, y, penalty),
    }
    write_json(output / "model.json", model)
    report["model"] = {
        "path": str(output / "model.json"),
        "sha256": file_hash(output / "model.json"),
    }
    write_json(output / "report.json", report)
    np.savez_compressed(
        output / "cross_validation.npz",
        features=x,
        labels=y,
        dates=dates,
        biases=biases,
        **predictions,
    )
    verify_registration(protocol)
    print(report["overall"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "registration.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_forecast(read_json(args.protocol), args.output)
