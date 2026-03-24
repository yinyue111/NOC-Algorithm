from __future__ import annotations

import statistics

from noccore.config.settings import PipelineSettings
from noccore.utils.stats import boxplot_bounds, huber_linear_forecast, mann_kendall_test, safe_mean, safe_std


class LocalDetectorSuite:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def detect(self, current_value: float, history_values: list[float]) -> dict[str, float | list[str]]:
        labels: list[str] = []

        if self._robust_regression(current_value, history_values):
            labels.append("robust_regression")
        if self._boxplot(current_value, history_values):
            labels.append("boxplot")
        if self._n_sigma(current_value, history_values):
            labels.append("n_sigma")
        if self._trend(current_value, history_values):
            labels.append("trend")

        return {
            "local_score": len(labels) / 4.0,
            "labels": labels,
        }

    def _robust_regression(self, current_value: float, history_values: list[float]) -> bool:
        window = history_values[-self.settings.local_regression_window :]
        if len(window) < 8:
            return False
        predict_value, residual_std, _ = huber_linear_forecast(window)
        residual = abs(current_value - predict_value)
        return residual > max(3.0 * residual_std, 0.01)

    def _boxplot(self, current_value: float, history_values: list[float]) -> bool:
        window = history_values[-self.settings.local_boxplot_window :]
        if len(window) < 8:
            return False
        lower_bound, upper_bound = boxplot_bounds(window)
        std = safe_std(window, 0.0)
        median = statistics.median(window)
        if abs(current_value - median) < max(2.0 * std, 0.005):
            return False
        return current_value < lower_bound or current_value > upper_bound

    def _n_sigma(self, current_value: float, history_values: list[float]) -> bool:
        window = history_values[-self.settings.local_nsigma_window :]
        if len(window) < 8:
            return False
        mean = safe_mean(window, current_value)
        std = safe_std(window, 0.0)
        if std <= 0:
            return False
        if abs(current_value - mean) < max(2.0 * std, 0.005):
            return False
        return current_value < mean - 3 * std or current_value > mean + 3 * std

    def _trend(self, current_value: float, history_values: list[float]) -> bool:
        window = history_values[-self.settings.local_trend_window :]
        if len(window) < 6:
            return False
        diffs = [b - a for a, b in zip(window[:-1], window[1:])]
        diffs.append(current_value - window[-1])
        significant, _ = mann_kendall_test(diffs)
        if not significant:
            return False
        recent_mean = safe_mean(window, current_value)
        recent_std = safe_std(window, 0.0)
        return abs(current_value - recent_mean) > max(2.0 * recent_std, 0.005)
