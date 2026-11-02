"""Causal quota borrowing against an explicitly assumed future performance floor."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parents[1] / "rl_state_local_20261101"
sys.path.insert(0, str(PRIOR))

# isort: off
from market_policy import BudgetedQuantity  # noqa: E402
from controllers import FullResidualMacro  # noqa: E402
from forecast import causal_features, predict_ridge  # noqa: E402
import numpy as np  # noqa: E402
# isort: on


class ExpectedSurplusQuantity(BudgetedQuantity):
    def choose(self, macro_observation, hardware_observation):
        if (
            macro_observation.shape != (19,)
            or hardware_observation.shape != (18,)
            or not np.isfinite(macro_observation).all()
            or not np.isfinite(hardware_observation).all()
            or self.window >= 96
        ):
            raise ValueError("Invalid causal expected-surplus context")
        if self.ridge is None:
            raise ValueError("This single follow-up uses the already frozen ridge")
        full_action = FullResidualMacro().predict(macro_observation)[0]
        maximum = float(full_action[0]) * 30
        features = causal_features(
            hardware_observation, self.previous_mean, self.window * 900
        )
        predicted = float(predict_ridge(self.ridge, features))
        current_lmp = float(hardware_observation[7]) * 200
        rate = float(macro_observation[17]) * self.config["rmcp_normalizer"]
        planned_performance = self.config["planning_performance"]
        guard = self.config["guard_ratio"]
        margin = rate * planned_performance + current_lmp * predicted
        measured_credit = self.delivered - guard * self.maximum
        remaining_hours = (95 - self.window) * 0.25
        expected_future_credit = (planned_performance - guard) * 5 * remaining_hours
        enough = (
            measured_credit + expected_future_credit >= guard * maximum * 0.25 + 1e-3
        )
        withhold = bool(enough and margin < -self.threshold)
        action = full_action.copy()
        if withhold:
            lower = np.float32(5 / 30)
            if float(lower) * 30 < 5:
                lower = np.nextafter(lower, np.float32(1))
            action[0] = lower
        self.last_diagnostics = {
            "kind": "expected_surplus_quantity",
            "decision_window": self.window,
            "predicted_signed_regd_mean": predicted,
            "estimated_marginal_cash_usd_mw_h": margin,
            "measured_capacity_credit_mw_hours": measured_credit,
            "expected_future_credit_mw_hours": expected_future_credit,
            "remaining_hours_after_interval": remaining_hours,
            "future_performance_assumption": planned_performance,
            "sufficient_expected_terminal_credit": enough,
            "quantity_reduced": withhold,
            "unconditional_future_service_guarantee_claimed": False,
        }
        return action
