from __future__ import annotations

from dataclasses import dataclass

from noccore.schemas.metric import MetricPoint
from noccore.schemas.prediction import PredictionResult
from noccore.utils.stats import EPS, huber_linear_forecast, safe_mean, safe_std


@dataclass
class MinuteModelState:
    residual_std: float = 0.05
    model_version: str = "min_v1"


@dataclass
class MinutePredictionPlan:
    prediction: PredictionResult
    residual_std: float


class MinuteHuberModel:
    def __init__(self) -> None:
        self._state: dict[str, MinuteModelState] = {}

    def get_state(self, metric_name: str) -> MinuteModelState:
        return self._state.setdefault(metric_name, MinuteModelState())

    def predict(
        self,
        state: MinuteModelState,
        point: MetricPoint,
        history_points: list[MetricPoint],
        prediction_history: list[PredictionResult],
        abnormal_flags: list[bool],
    ) -> MinutePredictionPlan:
        cleaned_values = self._clean_history(history_points, prediction_history, abnormal_flags)
        if len(cleaned_values) < 5:
            predict_value = safe_mean(cleaned_values, point.value)
            residual_std = safe_std(cleaned_values, state.residual_std)
        else:
            window_values = cleaned_values[-60:]
            predict_value, residual_std, _ = huber_linear_forecast(
                window_values,
                epsilon=1.35,
                alpha=0.0001,
                max_iter=200,
                steps_ahead=1,
            )

        next_residual_std = max(0.8 * state.residual_std + 0.2 * residual_std, 0.001)
        return MinutePredictionPlan(
            prediction=PredictionResult(
                metric_name=point.metric_name,
                timestamp=point.timestamp,
                granularity=point.granularity,
                actual_value=point.value,
                predict_value=float(max(predict_value, EPS)),
                lower_bound=max(float(predict_value - 2 * next_residual_std), 0.0),
                upper_bound=float(predict_value + 2 * next_residual_std),
                model_name="minute_huber",
                model_version=state.model_version,
                features={},
            ),
            residual_std=next_residual_std,
        )

    def apply(self, metric_name: str, plan: MinutePredictionPlan) -> None:
        state = self.get_state(metric_name)
        state.residual_std = plan.residual_std

    def _clean_history(
        self,
        history_points: list[MetricPoint],
        prediction_history: list[PredictionResult],
        abnormal_flags: list[bool],
    ) -> list[float]:
        tail_len = min(60, len(history_points), len(prediction_history), len(abnormal_flags))
        cleaned: list[float] = []
        for index in range(tail_len):
            offset = tail_len - index
            history_point = history_points[-offset]
            if abnormal_flags[-offset]:
                cleaned.append(prediction_history[-offset].predict_value)
            else:
                cleaned.append(history_point.value)
        if not cleaned:
            return [point.value for point in history_points[-60:]]
        return cleaned

    def drop_metric(self, metric_name: str) -> None:
        self._state.pop(metric_name, None)
