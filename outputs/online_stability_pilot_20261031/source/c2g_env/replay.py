from __future__ import annotations

import numpy as np


def sample_previous(
    timestamps: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    *,
    max_age_seconds: float,
) -> np.ndarray:
    """Causal zero-order hold with explicit coverage and gap checks."""
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    targets = np.asarray(targets, dtype="datetime64[ns]")
    values = np.asarray(values, dtype=np.float64)
    if (
        times.ndim != 1
        or values.shape != times.shape
        or len(times) == 0
        or targets.ndim != 1
        or np.isnat(times).any()
        or np.isnat(targets).any()
        or not np.isfinite(values).all()
        or np.any(np.diff(times) <= np.timedelta64(0, "ns"))
    ):
        raise ValueError(
            "Replay requires finite values and strictly increasing source timestamps"
        )
    if not np.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be finite and positive")
    indices = np.searchsorted(times, targets, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Replay begins before the first source timestamp")
    ages = (targets - times[indices]) / np.timedelta64(1, "s")
    if np.any(ages > max_age_seconds):
        first = int(np.flatnonzero(ages > max_age_seconds)[0])
        raise ValueError(
            f"Missing replay coverage at {targets[first]}: "
            f"latest source sample is {ages[first]:g} seconds old"
        )
    return values[indices]
