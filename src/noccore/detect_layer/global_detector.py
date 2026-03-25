from __future__ import annotations

import logging
from dataclasses import dataclass, field

from noccore.config.settings import PipelineSettings
from noccore.schemas.prediction import PredictionResult
from noccore.utils.stats import EPS, gaussian_kde_bounds, normalized_severity


logger = logging.getLogger(__name__)
MAX_Z_HISTORY = 5000


@dataclass
class SegmentBoundsCache:
    cached_bounds: tuple[float, float] | None = None
    dirty_points: int = 0
    last_refresh_ts: int = 0


@dataclass
class GlobalDetectorState:
    z_history: dict[str, list[float]] = field(
        default_factory=lambda: {"peak": [], "valley": [], "mid": []}
    )
    bounds_cache: dict[str, SegmentBoundsCache] = field(
        default_factory=lambda: {
            "peak": SegmentBoundsCache(),
            "valley": SegmentBoundsCache(),
            "mid": SegmentBoundsCache(),
        }
    )
    up_streak: int = 0
    down_streak: int = 0


@dataclass
class GlobalDetectionPlan:
    result: dict[str, float | bool | str]
    segment: str
    up_streak: int
    down_streak: int
    new_z_score: float | None
    cache: SegmentBoundsCache


class GlobalAnomalyDetector:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self._state: dict[str, GlobalDetectorState] = {}

    def get_state(self, metric_name: str) -> GlobalDetectorState:
        return self._state.setdefault(metric_name, GlobalDetectorState())

    def detect(self, state: GlobalDetectorState, prediction: PredictionResult) -> GlobalDetectionPlan:
        segment = self._segment(prediction)
        history = state.z_history[segment]
        cache = self._copy_cache(state.bounds_cache[segment])

        lower_bound, upper_bound = self._get_bounds(history, cache, prediction.timestamp, prediction.metric_name, segment)
        lower_bound = min(lower_bound, -0.05)
        upper_bound = max(upper_bound, 0.05)
        z_score = (prediction.actual_value - prediction.predict_value) / max(abs(prediction.predict_value) ** 0.5, EPS)

        direction = "normal"
        up_streak = state.up_streak
        down_streak = state.down_streak
        if z_score < lower_bound:
            down_streak = state.down_streak + 1
            up_streak = 0
            direction = "down"
        elif z_score > upper_bound:
            up_streak = state.up_streak + 1
            down_streak = 0
            direction = "up"
        else:
            up_streak = 0
            down_streak = 0

        required = (
            self.settings.second_consecutive_points
            if prediction.granularity == "1s"
            else self.settings.minute_consecutive_points
        )
        global_abnormal = down_streak >= required or up_streak >= required
        severity = normalized_severity(z_score, lower_bound, upper_bound)

        new_z_score: float | None = None
        if not global_abnormal:
            new_z_score = float(z_score)
            cache.dirty_points += 1
            if cache.dirty_points >= self.settings.kde_refresh_every_points:
                refresh_history = history[-(MAX_Z_HISTORY - 1) :] + [new_z_score]
                self._refresh_bounds(refresh_history, cache, prediction.timestamp, prediction.metric_name, segment)

        return GlobalDetectionPlan(
            result={
                "global_abnormal": bool(global_abnormal),
                "z_score": float(z_score),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "severity": float(severity),
                "direction": direction,
            },
            segment=segment,
            up_streak=up_streak,
            down_streak=down_streak,
            new_z_score=new_z_score,
            cache=cache,
        )

    def _segment(self, prediction: PredictionResult) -> str:
        rolling_mean = float(prediction.features.get("rolling_mean_7d", prediction.predict_value))
        if "rolling_p70_7d" in prediction.features and "rolling_p30_7d" in prediction.features:
            rolling_p70 = float(prediction.features["rolling_p70_7d"])
            rolling_p30 = float(prediction.features["rolling_p30_7d"])
        else:
            rolling_std = float(prediction.features.get("rolling_std_7d", 0.0))
            rolling_p70 = rolling_mean + 0.25 * rolling_std
            rolling_p30 = rolling_mean - 0.25 * rolling_std
        if prediction.predict_value >= rolling_p70:
            return "peak"
        if prediction.predict_value <= rolling_p30:
            return "valley"
        return "mid"

    def _get_bounds(
        self,
        history: list[float],
        cache: SegmentBoundsCache,
        timestamp: int,
        metric_name: str,
        segment: str,
    ) -> tuple[float, float]:
        if (
            cache.cached_bounds is None
            or timestamp - cache.last_refresh_ts >= self.settings.kde_refresh_interval_sec
        ):
            self._refresh_bounds(history, cache, timestamp, metric_name, segment)
        return cache.cached_bounds or (-3.0, 3.0)

    def _refresh_bounds(
        self,
        history: list[float],
        cache: SegmentBoundsCache,
        timestamp: int,
        metric_name: str,
        segment: str,
    ) -> None:
        cache.cached_bounds = gaussian_kde_bounds(
            history,
            self.settings.global_lower_quantile,
            self.settings.global_upper_quantile,
        )
        cache.dirty_points = 0
        cache.last_refresh_ts = timestamp
        logger.debug(
            "KDE bounds refreshed %s/%s: [%.3f, %.3f]",
            metric_name,
            segment,
            cache.cached_bounds[0],
            cache.cached_bounds[1],
        )

    def apply(self, metric_name: str, plan: GlobalDetectionPlan) -> None:
        state = self.get_state(metric_name)
        state.up_streak = plan.up_streak
        state.down_streak = plan.down_streak
        if plan.new_z_score is not None:
            history = state.z_history[plan.segment]
            history.append(plan.new_z_score)
            if len(history) > MAX_Z_HISTORY:
                del history[:-MAX_Z_HISTORY]
        state.bounds_cache[plan.segment] = self._copy_cache(plan.cache)

    def _copy_cache(self, cache: SegmentBoundsCache) -> SegmentBoundsCache:
        return SegmentBoundsCache(
            cached_bounds=cache.cached_bounds,
            dirty_points=cache.dirty_points,
            last_refresh_ts=cache.last_refresh_ts,
        )

    def drop_metric(self, metric_name: str) -> None:
        self._state.pop(metric_name, None)
