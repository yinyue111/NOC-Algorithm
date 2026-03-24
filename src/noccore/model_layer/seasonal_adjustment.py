from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from noccore.config.metric_registry import MetricMetadata
from noccore.history_layer.store import HistoryStore
from noccore.schemas.metric import MetricPoint
from noccore.schemas.prediction import PredictionResult
from noccore.utils.stats import EPS, safe_mean


logger = logging.getLogger(__name__)

BU_FACTORS: dict[str, float] = {"AA": 1.00, "BB": 1.02, "CC": 0.98}


@dataclass
class SeasonalState:
    correction_rate: float = 1.0
    residual_std: float = 0.01
    residual_history: deque[float] = field(default_factory=lambda: deque(maxlen=1200))
    fallback_logged: bool = False
    model_version: str = "sec_v1"


@dataclass
class SeasonalPredictionPlan:
    prediction: PredictionResult
    correction_rate: float
    residual_std: float
    residual: float
    fallback_logged: bool


class SeasonalAdjustmentModel:
    def __init__(self, timezone: str, history_store: HistoryStore) -> None:
        self.timezone = timezone
        self.history_store = history_store
        self._state: dict[str, SeasonalState] = {}

    def get_state(self, metric_name: str) -> SeasonalState:
        return self._state.setdefault(metric_name, SeasonalState())

    def predict(
        self,
        state: SeasonalState,
        point: MetricPoint,
        history_points: list[MetricPoint],
        prediction_history: list[PredictionResult],
        features: dict,
        metadata: MetricMetadata,
        raw_history_points: list[MetricPoint] | None = None,
    ) -> SeasonalPredictionPlan:
        is_holiday = bool(features["isHoliday"])
        aligned_values = self.history_store.get_aligned_history_values(
            point.metric_name,
            timestamp=point.timestamp,
            timezone=self.timezone,
            is_holiday=is_holiday,
            aligned_limit=28,
            fallback_limit=60,
        )

        if aligned_values:
            fallback_logged = False
            base_value = float(np.median(aligned_values))
        else:
            fallback_logged = state.fallback_logged
            if not fallback_logged:
                logger.warning("No aligned history for %s, falling back to recent mean", point.metric_name)
                fallback_logged = True
            recent_values = [item.value for item in (raw_history_points or [])[-120:]]
            if not recent_values:
                recent_values = [
                    item.value
                    for item in self.history_store.get_recent_raw_points(point.metric_name, limit=120)
                ]
            if not recent_values:
                recent_values = [item.value for item in history_points[-120:]]
            base_value = safe_mean(recent_values, point.value)

        preprocessed = max(base_value + float(features["baseDayDiff"]), base_value * 0.5, EPS)
        feature_factor = self._feature_factor(features, metadata)
        predict_value = max(preprocessed * state.correction_rate * feature_factor, EPS)
        residual = point.value - predict_value
        next_correction_rate, next_residual_std = self._next_state_values(
            state=state,
            actual_value=point.value,
            base_value=preprocessed * feature_factor,
            predict_value=predict_value,
        )

        return SeasonalPredictionPlan(
            prediction=PredictionResult(
                metric_name=point.metric_name,
                timestamp=point.timestamp,
                granularity=point.granularity,
                actual_value=point.value,
                predict_value=float(predict_value),
                lower_bound=max(float(predict_value - 2 * state.residual_std), 0.0),
                upper_bound=float(predict_value + 2 * state.residual_std),
                model_name="seasonal_adjustment",
                model_version=state.model_version,
                features=features,
            ),
            correction_rate=next_correction_rate,
            residual_std=next_residual_std,
            residual=float(residual),
            fallback_logged=fallback_logged,
        )

    def _feature_factor(self, features: dict, metadata: MetricMetadata) -> float:
        factor = 1.0
        if features.get("isHoliday"):
            factor *= 1.02
        if features.get("isHotSpot"):
            factor *= 1.05
        if features.get("isRCA"):
            factor *= 0.97
        if metadata.business_line:
            factor *= BU_FACTORS.get(metadata.business_line, 1.0)
        return factor

    def apply(self, metric_name: str, plan: SeasonalPredictionPlan) -> None:
        state = self.get_state(metric_name)
        state.correction_rate = plan.correction_rate
        state.residual_std = plan.residual_std
        state.fallback_logged = plan.fallback_logged
        state.residual_history.append(plan.residual)

    def _next_state_values(
        self,
        state: SeasonalState,
        actual_value: float,
        base_value: float,
        predict_value: float,
    ) -> tuple[float, float]:
        observed_rate = actual_value / max(base_value, EPS)
        residual = actual_value - predict_value
        residual_abs = abs(residual)
        return (
            float(np.clip(0.90 * state.correction_rate + 0.10 * observed_rate, 0.5, 1.5)),
            max(0.90 * state.residual_std + 0.10 * residual_abs, 0.001),
        )

    def drop_metric(self, metric_name: str) -> None:
        self._state.pop(metric_name, None)
