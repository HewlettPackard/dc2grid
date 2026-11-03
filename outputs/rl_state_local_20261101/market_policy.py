"""Causal macro-policy interface shared by greedy and forecast-based controls."""

from __future__ import annotations

import math

from study import BASELINE, file_hash, read_json  # noqa: F401

# isort: off
import numpy as np
from baselines.rule_based_macro import RuleBasedMacroController
from controllers import FullResidualMacro
from forecast import FEATURES, causal_features, predict_ridge
# isort: on


class ExistingMacro:
    def __init__(self, kind):
        self.kind = kind
        self.policy = (
            RuleBasedMacroController() if kind == "rule" else FullResidualMacro()
        )
        self.last_diagnostics = {}

    def choose(self, macro_observation, hardware_observation):
        action = self.policy.predict(macro_observation, deterministic=True)[0]
        self.last_diagnostics = {"kind": self.kind}
        return action

    def observe(self, macro_info, signed_mean):
        pass


def make_macro(spec, protocol):
    kind = spec["macro"]
    if kind in ("rule", "full_residual"):
        return ExistingMacro(kind)
    if kind == "forecast_quantity":
        return BudgetedQuantity(spec, protocol)
    raise ValueError(f"Unknown registered macro policy: {kind}")


def maximum_native_quantity(residual_mw):
    if not math.isfinite(residual_mw) or residual_mw < 5:
        raise ValueError("Invalid posted residual for the quantity contract")
    maximum = min(30.0, float(residual_mw))
    native = np.float32(maximum / 30)
    if float(native) * 30 > maximum:
        native = np.nextafter(native, np.float32(0))
    return float(native) * 30


class BudgetedQuantity:
    def __init__(self, spec, protocol):
        self.config = protocol["high_level"]
        self.forecaster = spec["forecaster"]
        self.threshold = spec["margin_threshold_usd_mw_h"]
        if self.threshold not in self.config["margin_thresholds_usd_mw_h"]:
            raise ValueError("Unregistered marginal-cash threshold")
        if not 0.9 <= self.config["guard_ratio"] <= 1:
            raise ValueError("The causal service guard cannot relax the 90% floor")
        self.ridge = None
        if self.forecaster == "ridge":
            artifact = spec["forecast_checkpoint"]
            if file_hash(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Frozen forecast checkpoint changed")
            self.ridge = read_json(artifact["path"])
            if (
                tuple(self.ridge["features"]) != FEATURES
                or self.ridge["training_registration_id"] != protocol["registration_id"]
            ):
                raise ValueError("Wrong forecast feature or training contract")
        elif self.forecaster != "decay":
            raise ValueError("Unknown causal forecaster")
        self.window = 0
        self.previous_mean = 0.0
        self.delivered = 0.0
        self.maximum = 0.0
        self.last_diagnostics = {}

    def choose(self, macro_observation, hardware_observation):
        if (
            macro_observation.shape != (19,)
            or hardware_observation.shape != (18,)
            or not np.isfinite(macro_observation).all()
            or not np.isfinite(hardware_observation).all()
            or self.window >= 96
        ):
            raise ValueError("Invalid causal macro decision context")
        full_action = FullResidualMacro().predict(macro_observation)[0]
        maximum = float(full_action[0]) * 30
        features = causal_features(
            hardware_observation, self.previous_mean, self.window * 900
        )
        if self.ridge is not None:
            predicted = float(predict_ridge(self.ridge, features))
        else:
            rho = self.config["decay_rho"]
            predicted = (1 - rho**180) / (180 * (1 - rho)) * features[0]
        current_lmp = float(hardware_observation[7]) * 200
        rate = float(macro_observation[17]) * self.config["rmcp_normalizer"]
        margin = rate * self.config["planning_performance"] + current_lmp * predicted
        guard = self.config["guard_ratio"]
        credit = self.delivered - guard * self.maximum
        # Even zero delivery in this accepted interval cannot spend more than this.
        sufficient_credit = credit >= guard * maximum * 0.25 + 1e-3
        withhold = bool(sufficient_credit and margin < -self.threshold)
        action = full_action.copy()
        if withhold:
            lower = np.float32(5 / 30)
            if float(lower) * 30 < 5:
                lower = np.nextafter(lower, np.float32(1))
            action[0] = lower
        self.last_diagnostics = {
            "kind": "forecast_quantity",
            "forecaster": self.forecaster,
            "decision_window": self.window,
            "previous_window_mean": self.previous_mean,
            "predicted_signed_regd_mean": predicted,
            "current_lmp_usd_mwh": current_lmp,
            "posted_rmcp_usd_mw_h": rate,
            "estimated_marginal_cash_usd_mw_h": margin,
            "causal_capacity_credit_mwh": credit,
            "sufficient_worst_interval_credit": sufficient_credit,
            "quantity_reduced": withhold,
            "future_service_guarantee_claimed": False,
        }
        return action

    def observe(self, macro_info, signed_mean):
        maximum = maximum_native_quantity(macro_info["residual_mw"])
        reference = maximum if macro_info["bid_accepted"] else 5.0
        self.maximum += reference * macro_info["duration_hours"]
        self.delivered += (
            macro_info["committed_mw"]
            * macro_info["perf_score"]
            * macro_info["duration_hours"]
        )
        self.previous_mean = float(signed_mean)
        self.window += 1
