from __future__ import annotations

import csv
import math
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class C2GMacroMetricsCallback(BaseCallback):
    """Log macro economics and fast-step tracking without schema fallbacks."""

    def __init__(
        self, csv_path: Path | None = None, print_freq: int = 20, verbose: int = 0
    ):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.print_freq = print_freq
        self._episodes: dict[int, dict[str, float]] = {}
        self._completed = 0

    def _on_step(self) -> bool:
        for index, (reward, info, done) in enumerate(
            zip(
                self.locals["rewards"],
                self.locals["infos"],
                self.locals["dones"],
            )
        ):
            episode = self._episodes.setdefault(
                index,
                {
                    "macro_steps": 0.0,
                    "hardware_steps": 0.0,
                    "total_reward": 0.0,
                    "tracking_error_sq_sum": 0.0,
                    "regulation_revenue": 0.0,
                    "electricity_cost": 0.0,
                    "temp_A_max": -math.inf,
                    "temp_B_max": -math.inf,
                },
            )
            episode["macro_steps"] += 1
            episode["hardware_steps"] += info["sub_steps_run"]
            episode["total_reward"] += float(reward)
            for key in (
                "tracking_error_sq_sum",
                "regulation_revenue",
                "electricity_cost",
            ):
                episode[key] += info[key]
            for key in ("temp_A_max", "temp_B_max"):
                episode[key] = max(episode[key], info[key])
            if done:
                metrics = {
                    **episode,
                    "tracking_rmse_kw": math.sqrt(
                        episode["tracking_error_sq_sum"] / episode["hardware_steps"]
                    ),
                    "mean_reward": episode["total_reward"] / episode["macro_steps"],
                    "soc_final": float(info["bess_soc_end"]),
                    "terminated": int(info["terminated"]),
                    "truncated": int(info["truncated"]),
                }
                for key, value in metrics.items():
                    self.logger.record(f"macro/{key}", value)
                self._completed += 1
                if self.csv_path is not None:
                    self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                    header = not self.csv_path.exists()
                    with self.csv_path.open("a", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(metrics))
                        if header:
                            writer.writeheader()
                        writer.writerow(metrics)
                if (
                    self.verbose
                    and self.print_freq > 0
                    and self._completed % self.print_freq == 0
                ):
                    print(f"Macro episode {self._completed}: {metrics}")
                del self._episodes[index]
        return True
