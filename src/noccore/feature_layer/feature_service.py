from __future__ import annotations

from typing import Iterable

from noccore.config.metric_registry import MetricMetadata
from noccore.feature_layer.calendar_provider import CalendarProvider
from noccore.schemas.metric import MetricPoint
from noccore.schemas.prediction import PredictionResult
from noccore.utils.stats import safe_mean, safe_percentile, safe_std
from noccore.utils.time import to_local_datetime


class FeatureService:
    def __init__(self, calendar_provider: CalendarProvider, timezone: str) -> None:
        self.calendar_provider = calendar_provider
        self.timezone = timezone

    def build(
        self,
        point: MetricPoint,
        history_points: Iterable[MetricPoint],
        prediction_history: Iterable[PredictionResult],
        metadata: MetricMetadata,
        calendar_context: dict[str, bool] | None = None,
    ) -> dict[str, float | int | bool | str]:
        history_points = list(history_points)
        prediction_history = list(prediction_history)
        calendar_context = calendar_context or self.calendar_provider.get_context(point.metric_name, point.timestamp)
        lookback_start = point.timestamp - 7 * 24 * 3600
        recent_values = [item.value for item in history_points if item.timestamp >= lookback_start]
        current_dt = to_local_datetime(point.timestamp, self.timezone)
        residual_mean = 0.0
        if prediction_history and history_points:
            aligned_length = min(len(history_points), len(prediction_history), 120)
            residuals = [
                history_points[-aligned_length + idx].value - prediction_history[-aligned_length + idx].predict_value
                for idx in range(aligned_length)
            ]
            residual_mean = safe_mean(residuals, 0.0)

        rolling_mean = safe_mean(recent_values, point.value)
        scale = max(abs(rolling_mean), abs(point.value), 1e-3)
        base_day_diff = max(min(residual_mean, 0.2 * scale), -0.2 * scale)

        return {
            "isHoliday": calendar_context["isHoliday"],
            "isRCA": calendar_context["isRCA"],
            "isHotSpot": calendar_context["isHotSpot"],
            "business_unit": metadata.business_line,
            "baseDayDiff": float(base_day_diff),
            "hour_of_day": current_dt.hour,
            "day_of_week": current_dt.weekday(),
            "rolling_mean_7d": rolling_mean,
            "rolling_std_7d": safe_std(recent_values, 0.0),
            "rolling_p70_7d": safe_percentile(recent_values, 70, rolling_mean),
            "rolling_p30_7d": safe_percentile(recent_values, 30, rolling_mean),
        }
