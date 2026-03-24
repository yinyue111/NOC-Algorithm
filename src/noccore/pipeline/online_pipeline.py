from __future__ import annotations

import logging
import threading
import zlib
from itertools import islice
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from noccore.alert_layer.policy import AlertPolicy
from noccore.alert_layer.suppressor import AlertSuppressor
from noccore.config.metric_registry import MetricRegistry
from noccore.config.settings import DEFAULT_SETTINGS, PipelineSettings
from noccore.data_layer.downsample import Downsampler
from noccore.data_layer.eligibility import AIEligibilityAssessor
from noccore.data_layer.repair import MetricRepairer
from noccore.data_layer.router import MetricRouter, RouteDecision
from noccore.data_layer.validator import MetricValidator
from noccore.detect_layer.confidence import ConfidenceScorer
from noccore.detect_layer.global_detector import GlobalAnomalyDetector
from noccore.detect_layer.local_detectors import LocalDetectorSuite
from noccore.detect_layer.periodic_filter import PeriodicityFilter
from noccore.feature_layer.calendar_provider import CalendarProvider
from noccore.feature_layer.feature_service import FeatureService
from noccore.history_layer.store import HistoryStore, InMemoryHistoryStore
from noccore.model_layer.minute_huber import MinuteHuberModel
from noccore.model_layer.seasonal_adjustment import SeasonalAdjustmentModel
from noccore.schemas.alert import AlertEvent
from noccore.schemas.anomaly import AnomalyEvent
from noccore.schemas.metric import MetricPoint
from noccore.schemas.prediction import PredictionResult


logger = logging.getLogger(__name__)


@dataclass
class MetricRuntimeState:
    last_raw_point: MetricPoint | None = None
    last_seen_timestamp: int = 0
    eligibility_last_eval_ts: int | None = None
    recent_raw_points: deque[MetricPoint] = field(default_factory=deque)
    recent_series_points: dict[str, deque[MetricPoint]] = field(default_factory=dict)
    recent_predictions: dict[str, deque[PredictionResult]] = field(default_factory=dict)
    recent_anomalies: dict[str, deque[AnomalyEvent]] = field(default_factory=dict)
    raw_history_primed: bool = False
    series_history_primed: set[str] = field(default_factory=set)
    prediction_history_primed: set[str] = field(default_factory=set)
    anomaly_history_primed: set[str] = field(default_factory=set)


class OnlinePipeline:
    def __init__(
        self,
        settings: PipelineSettings | None = None,
        registry: MetricRegistry | None = None,
        calendar_provider: CalendarProvider | None = None,
        history_store: HistoryStore | None = None,
    ) -> None:
        self.settings = settings or DEFAULT_SETTINGS
        self.registry = registry or MetricRegistry()
        self.validator = MetricValidator()
        self.repairer = MetricRepairer(self.settings)
        self.eligibility = AIEligibilityAssessor(self.settings)
        self.router = MetricRouter(self.settings)
        self.downsampler = Downsampler()
        self.calendar_provider = calendar_provider or CalendarProvider(timezone=self.settings.timezone)
        self.history_store = history_store or InMemoryHistoryStore(self.settings)
        self.feature_service = FeatureService(self.calendar_provider, timezone=self.settings.timezone)
        self.second_model = SeasonalAdjustmentModel(
            timezone=self.settings.timezone,
            history_store=self.history_store,
        )
        self.minute_model = MinuteHuberModel()
        self.global_detector = GlobalAnomalyDetector(self.settings)
        self.local_detectors = LocalDetectorSuite(self.settings)
        self.periodicity_filter = PeriodicityFilter(self.settings)
        self.confidence = ConfidenceScorer()
        self.alert_policy = AlertPolicy(
            settings=self.settings,
            suppressor=AlertSuppressor(self.settings.alert_merge_window_sec),
        )
        self._state: dict[str, MetricRuntimeState] = {}
        self._state_lock = threading.RLock()
        self._maintenance_lock = threading.RLock()
        self._cleanup_running = False
        stripe_count = max(int(self.settings.metric_lock_stripes), 1)
        self._metric_locks = [threading.RLock() for _ in range(stripe_count)]
        self._processed_points = 0

    def reload_registry(self, registry: MetricRegistry) -> None:
        self.registry.reload_from(registry)

    def process_points(
        self,
        raw_points: list[MetricPoint | dict[str, Any]],
        flush: bool = False,
    ) -> list[AlertEvent]:
        validated = [self.validator.validate(raw_point) for raw_point in raw_points]
        deduped = self._sort_and_deduplicate(validated)
        alerts: list[AlertEvent] = []
        index = 0
        while index < len(deduped):
            metric_name = deduped[index].metric_name
            group_end = index + 1
            while group_end < len(deduped) and deduped[group_end].metric_name == metric_name:
                group_end += 1
            alerts.extend(self._process_validated_points(deduped[index:group_end], flush_writes=False))
            index = group_end
        if flush:
            alerts.extend(self.flush())
        self.history_store.flush_writes()
        return alerts

    def process_point(self, point: MetricPoint | dict[str, Any]) -> list[AlertEvent]:
        point = self.validator.validate(point)
        return self._process_validated_points([point], flush_writes=True)

    def _process_validated_points(self, points: list[MetricPoint], flush_writes: bool) -> list[AlertEvent]:
        if not points:
            return []
        metric_name = points[0].metric_name
        alerts: list[AlertEvent] = []
        processed_points = 0
        last_timestamp: int | None = None
        with self._metric_lock(metric_name):
            metadata = self.registry.get_or_create(points[0])
            runtime = self._get_or_create_runtime_state(metric_name)
            for point in points:
                prepared_points, current_point = self.repairer.repair_continuity(runtime.last_raw_point, point)
                prepared_points.append(current_point)

                for prepared_point in prepared_points:
                    raw_history = self._get_recent_raw_history(
                        runtime,
                        prepared_point.metric_name,
                        limit=self.settings.jump_clip_history_window,
                    )
                    repaired_point = self.repairer.clip_jump(prepared_point, (item.value for item in raw_history))
                    eligibility = self._get_or_refresh_eligibility(runtime, metadata, repaired_point)
                    route = self.router.route(metadata, eligibility)
                    series_calendar_context = None
                    raw_is_holiday: bool
                    if route.granularity_sec == 1 and route.mode != "zero_rule":
                        series_calendar_context = self.calendar_provider.get_context(
                            repaired_point.metric_name,
                            repaired_point.timestamp,
                        )
                        raw_is_holiday = bool(series_calendar_context["isHoliday"])
                    else:
                        raw_is_holiday = self.calendar_provider.is_holiday(repaired_point.timestamp)
                    alerts.extend(
                        self._route_point(
                            runtime,
                            metadata,
                            repaired_point,
                            route,
                            calendar_context=series_calendar_context,
                        )
                    )
                    self._append_raw_point(runtime, repaired_point, raw_is_holiday)
                    runtime.last_raw_point = repaired_point
                    runtime.last_seen_timestamp = repaired_point.timestamp
                    processed_points += 1
                    last_timestamp = repaired_point.timestamp
        if processed_points > 0 and last_timestamp is not None:
            self._record_processed_points(processed_points, last_timestamp)
        if flush_writes:
            self.history_store.flush_writes()
        return alerts

    def flush(self) -> list[AlertEvent]:
        alerts: list[AlertEvent] = []
        for point in self.downsampler.flush():
            with self._metric_lock(point.metric_name):
                metadata = self.registry.get_or_create(point)
                runtime = self._get_or_create_runtime_state(point.metric_name)
                eligibility = self.registry.get_eligibility(point.metric_name)
                route = self.router.route(metadata, eligibility)
                series_calendar_context = (
                    None
                    if route.mode == "zero_rule"
                    else self.calendar_provider.get_context(point.metric_name, point.timestamp)
                )
                alerts.extend(self._process_series_point(runtime, metadata, point, route, series_calendar_context))
        self.history_store.flush_writes()
        return alerts

    def _route_point(
        self,
        runtime: MetricRuntimeState,
        metadata: Any,
        point: MetricPoint,
        route: RouteDecision,
        calendar_context: dict[str, bool] | None = None,
    ) -> list[AlertEvent]:
        if route.granularity_sec == 1:
            return self._process_series_point(runtime, metadata, point, route, calendar_context)

        aggregated = self.downsampler.push(point, route.granularity_sec)
        if aggregated is None:
            return []
        series_calendar_context = (
            None
            if route.mode == "zero_rule"
            else self.calendar_provider.get_context(aggregated.metric_name, aggregated.timestamp)
        )
        return self._process_series_point(runtime, metadata, aggregated, route, series_calendar_context)

    def _process_series_point(
        self,
        runtime: MetricRuntimeState,
        metadata: Any,
        point: MetricPoint,
        route: RouteDecision,
        calendar_context: dict[str, bool] | None = None,
    ) -> list[AlertEvent]:
        history_limit = self.settings.second_history_window if route.label == "1s" else self.settings.minute_history_window
        anomaly_limit = self.settings.anomaly_window if route.label == "1s" else self.settings.minute_history_window
        history_points = self._get_recent_series_history(
            runtime,
            point.metric_name,
            route.label,
            limit=history_limit,
        )
        prediction_history = self._get_recent_prediction_history(
            runtime,
            point.metric_name,
            route.label,
            limit=self.settings.prediction_window,
        )
        anomaly_history = self._get_recent_anomaly_history(
            runtime,
            point.metric_name,
            route.label,
            limit=anomaly_limit,
        )

        if route.mode == "zero_rule":
            prediction = self._zero_rule_prediction(point, history_points)
            anomaly = self._zero_rule_anomaly(point, history_points, prediction, route.label)
            prediction_plan = None
            global_plan = None
        else:
            features = self.feature_service.build(
                point,
                history_points,
                prediction_history,
                metadata,
                calendar_context=calendar_context,
            )
            if route.mode == "ai":
                raw_history_points = self._get_recent_raw_history(runtime, point.metric_name, limit=120)
                prediction_plan = self.second_model.predict(
                    self.second_model.get_state(point.metric_name),
                    point,
                    history_points,
                    prediction_history,
                    features,
                    metadata,
                    raw_history_points=raw_history_points,
                )
            else:
                abnormal_flags = [item.global_abnormal or item.local_score > 0 for item in anomaly_history]
                prediction_plan = self.minute_model.predict(
                    self.minute_model.get_state(point.metric_name),
                    point,
                    history_points,
                    prediction_history,
                    abnormal_flags,
                )
            prediction = prediction_plan.prediction.model_copy(update={"features": features})
            anomaly, global_plan = self._detect_anomaly(
                point,
                prediction,
                history_points,
                prediction_history,
                anomaly_history,
                route.label,
            )
            if route.mode == "ai":
                self.second_model.apply(point.metric_name, prediction_plan)
            else:
                self.minute_model.apply(point.metric_name, prediction_plan)

        alert_plan = self.alert_policy.decide(
            metadata,
            prediction,
            anomaly,
            self.alert_policy.get_state(point.metric_name),
        )
        if global_plan is not None:
            self.global_detector.apply(point.metric_name, global_plan)
        self.alert_policy.apply(point.metric_name, alert_plan)
        alerts = alert_plan.alerts
        self._append_series(runtime, point, prediction, anomaly)
        return alerts

    def _detect_anomaly(
        self,
        point: MetricPoint,
        prediction: PredictionResult,
        history_points: list[MetricPoint],
        prediction_history: list[PredictionResult],
        anomaly_history: list[AnomalyEvent],
        granularity: str,
    ) -> tuple[AnomalyEvent, Any]:
        global_plan = self.global_detector.detect(
            self.global_detector.get_state(point.metric_name),
            prediction,
        )
        global_result = global_plan.result
        local_result = self.local_detectors.detect(point.value, [item.value for item in history_points])

        labels = list(local_result["labels"])
        if global_result["global_abnormal"]:
            labels.insert(0, "global")

        provisional_flag = bool(global_result["global_abnormal"] or local_result["local_score"] > 0)
        anomaly_flags = [item.global_abnormal or item.local_score > 0 for item in anomaly_history] + [provisional_flag]
        values = [item.value for item in history_points] + [point.value]
        is_periodic = self.periodicity_filter.is_periodic(values, anomaly_flags, granularity) if provisional_flag else False
        severity = max(float(global_result["severity"]), float(local_result["local_score"]))
        confidence_score, components = self.confidence.score(
            actual_history=[item.value for item in history_points] + [point.value],
            predicted_history=[item.predict_value for item in prediction_history] + [prediction.predict_value],
            severity=severity,
            is_periodic=is_periodic,
        )

        return AnomalyEvent(
            metric_name=point.metric_name,
            timestamp=point.timestamp,
            granularity=granularity,
            global_abnormal=bool(global_result["global_abnormal"]),
            local_score=float(local_result["local_score"]),
            is_periodic=is_periodic,
            z_score=float(global_result["z_score"]),
            lower_bound=float(global_result["lower_bound"]),
            upper_bound=float(global_result["upper_bound"]),
            severity=float(severity),
            direction=str(global_result["direction"]),
            abnormal_labels=labels,
            confidence_score=float(confidence_score),
            confidence_components=components,
        ), global_plan

    def _zero_rule_prediction(self, point: MetricPoint, history_points: list[MetricPoint]) -> PredictionResult:
        baseline_values = [item.value for item in history_points[-6:]]
        predict_value = sum(baseline_values) / len(baseline_values) if baseline_values else point.value
        return PredictionResult(
            metric_name=point.metric_name,
            timestamp=point.timestamp,
            granularity=point.granularity,
            actual_value=point.value,
            predict_value=float(max(predict_value, 0.0)),
            lower_bound=0.0,
            upper_bound=float(max(predict_value, 0.0)),
            model_name="zero_rule_baseline",
            model_version="rule_v1",
            features={},
        )

    def _zero_rule_anomaly(
        self,
        point: MetricPoint,
        history_points: list[MetricPoint],
        prediction: PredictionResult,
        granularity: str,
    ) -> AnomalyEvent:
        history_values = [item.value for item in history_points[-6:]]
        baseline = prediction.predict_value
        drop_ratio = point.value / baseline if baseline > 0 else 0.0
        abnormal = point.value <= self.settings.zero_rule_floor or (
            baseline > 0 and drop_ratio <= self.settings.zero_rule_drop_ratio
        )
        severity = 1.0 if abnormal else 0.0
        confidence_score = 0.75 if abnormal else 0.0
        return AnomalyEvent(
            metric_name=point.metric_name,
            timestamp=point.timestamp,
            granularity=granularity,
            global_abnormal=abnormal,
            local_score=1.0 if abnormal else 0.0,
            is_periodic=False,
            z_score=point.value - baseline,
            lower_bound=0.0,
            upper_bound=baseline,
            severity=severity,
            direction="down" if abnormal else "normal",
            abnormal_labels=["zero_rule"] if abnormal else [],
            confidence_score=confidence_score,
            confidence_components={
                "metric_reliability": 1.0 if history_values else 0.5,
                "anomaly_severity": severity,
                "volatility_level": 1.0,
                "is_periodic": 1.0,
            },
        )

    def _append_series(
        self,
        runtime: MetricRuntimeState,
        point: MetricPoint,
        prediction: PredictionResult,
        anomaly: AnomalyEvent,
    ) -> None:
        self.history_store.append_series_point(point)
        self.history_store.append_prediction(prediction)
        self.history_store.append_anomaly(anomaly)
        self._get_or_create_series_cache(runtime, point.granularity).append(point)
        self._get_or_create_prediction_cache(runtime, prediction.granularity).append(prediction)
        self._get_or_create_anomaly_cache(runtime, anomaly.granularity).append(anomaly)

    def _sort_and_deduplicate(self, points: list[MetricPoint]) -> list[MetricPoint]:
        ordered = sorted(points, key=lambda item: (item.metric_name, item.timestamp))
        deduped: list[MetricPoint] = []
        seen: dict[tuple[str, int], int] = {}
        for point in ordered:
            key = (point.metric_name, point.timestamp)
            if key in seen:
                deduped[seen[key]] = point
            else:
                seen[key] = len(deduped)
                deduped.append(point)
        return deduped

    def _append_raw_point(self, runtime: MetricRuntimeState, point: MetricPoint, is_holiday: bool) -> None:
        self.history_store.append_raw_point(
            point,
            timezone=self.settings.timezone,
            is_holiday=is_holiday,
        )
        runtime.recent_raw_points.append(point)

    def _get_or_refresh_eligibility(
        self,
        runtime: MetricRuntimeState,
        metadata: Any,
        point: MetricPoint,
    ) -> Any:
        decision = self.registry.get_eligibility(point.metric_name)
        if metadata.force_ai:
            if decision is None:
                decision = self.eligibility.assess([point], metadata)
                self.registry.set_eligibility(point.metric_name, decision)
                runtime.eligibility_last_eval_ts = point.timestamp
            return decision

        if decision is None or self._should_reassess_eligibility(runtime, decision, point.timestamp):
            required_duration = self.settings.history_weeks_required * 7 * 24 * 3600
            since_timestamp = point.timestamp - required_duration if required_duration > 0 else None
            history = self.history_store.get_recent_raw_points(
                point.metric_name,
                since_timestamp=since_timestamp,
            ) + [point]
            decision = self.eligibility.assess(history, metadata)
            self.registry.set_eligibility(point.metric_name, decision)
            runtime.eligibility_last_eval_ts = point.timestamp
        return decision

    def _should_reassess_eligibility(self, runtime: MetricRuntimeState, decision: Any, timestamp: int) -> bool:
        if runtime.eligibility_last_eval_ts is None:
            return True
        interval = (
            self.settings.eligibility_refresh_interval_sec
            if decision.eligible
            else self.settings.eligibility_retry_interval_sec
        )
        return timestamp - runtime.eligibility_last_eval_ts >= interval

    def _cleanup_stale_metrics(self, current_timestamp: int) -> None:
        with self._state_lock:
            stale_metrics = [
                metric_name
                for metric_name, runtime in self._state.items()
                if runtime.last_seen_timestamp > 0
                and current_timestamp - runtime.last_seen_timestamp >= self.settings.state_ttl_sec
            ]
        for metric_name in stale_metrics:
            with self._metric_lock(metric_name):
                runtime = self._get_runtime_state(metric_name)
                if runtime is None:
                    continue
                if current_timestamp - runtime.last_seen_timestamp < self.settings.state_ttl_sec:
                    continue
                logger.info(
                    "Dropped stale metric state: %s (idle %.1fh)",
                    metric_name,
                    (current_timestamp - runtime.last_seen_timestamp) / 3600.0,
                )
                self._drop_metric_state(metric_name)
        if self.settings.data_retention_sec > 0:
            retention_cutoff = current_timestamp - self.settings.data_retention_sec
            self.history_store.prune_old_data(retention_cutoff)

    def _drop_metric_state(self, metric_name: str) -> None:
        with self._state_lock:
            self._state.pop(metric_name, None)
        self.registry.clear_eligibility(metric_name)
        self.downsampler.drop_metric(metric_name)
        self.second_model.drop_metric(metric_name)
        self.minute_model.drop_metric(metric_name)
        self.global_detector.drop_metric(metric_name)
        self.alert_policy.drop_metric(metric_name)
        if self.history_store.should_drop_metric_on_ttl():
            self.history_store.drop_metric(metric_name)

    def _get_or_create_runtime_state(self, metric_name: str) -> MetricRuntimeState:
        with self._state_lock:
            runtime = self._state.get(metric_name)
            if runtime is None:
                runtime = MetricRuntimeState(
                    recent_raw_points=deque(maxlen=self.settings.jump_clip_history_window),
                )
                self._state[metric_name] = runtime
            return runtime

    def _get_runtime_state(self, metric_name: str) -> MetricRuntimeState | None:
        with self._state_lock:
            return self._state.get(metric_name)

    def _record_processed_points(self, count: int, current_timestamp: int) -> None:
        should_cleanup = False
        with self._maintenance_lock:
            previous = self._processed_points
            self._processed_points += count
            crossed_boundary = (
                self.settings.state_cleanup_every_points > 0
                and previous // self.settings.state_cleanup_every_points
                != self._processed_points // self.settings.state_cleanup_every_points
            )
            if crossed_boundary and not self._cleanup_running:
                self._cleanup_running = True
                should_cleanup = True
        if not should_cleanup:
            return
        try:
            self._cleanup_stale_metrics(current_timestamp)
        finally:
            with self._maintenance_lock:
                self._cleanup_running = False

    def _metric_lock(self, metric_name: str) -> threading.RLock:
        index = zlib.adler32(metric_name.encode("utf-8")) % len(self._metric_locks)
        return self._metric_locks[index]

    def _get_recent_raw_history(
        self,
        runtime: MetricRuntimeState,
        metric_name: str,
        limit: int,
    ) -> list[MetricPoint]:
        if limit <= 0:
            return []
        self._validate_cache_limit(
            metric_name=metric_name,
            granularity="raw",
            limit=limit,
            cache_maxlen=self.settings.jump_clip_history_window,
        )
        if not runtime.raw_history_primed:
            runtime.recent_raw_points = deque(
                self.history_store.get_recent_raw_points(metric_name, limit=limit),
                maxlen=self.settings.jump_clip_history_window,
            )
            runtime.raw_history_primed = True
        return self._tail_items(runtime.recent_raw_points, limit)

    def _get_recent_series_history(
        self,
        runtime: MetricRuntimeState,
        metric_name: str,
        granularity: str,
        limit: int,
    ) -> list[MetricPoint]:
        cache = self._get_or_create_series_cache(runtime, granularity)
        self._validate_cache_limit(metric_name, granularity, limit, cache.maxlen)
        if granularity not in runtime.series_history_primed:
            cache.extend(self.history_store.get_recent_series_points(metric_name, granularity, limit=limit))
            runtime.series_history_primed.add(granularity)
        return self._tail_items(cache, limit)

    def _get_recent_prediction_history(
        self,
        runtime: MetricRuntimeState,
        metric_name: str,
        granularity: str,
        limit: int,
    ) -> list[PredictionResult]:
        cache = self._get_or_create_prediction_cache(runtime, granularity)
        self._validate_cache_limit(metric_name, granularity, limit, cache.maxlen, cache_kind="prediction")
        if granularity not in runtime.prediction_history_primed:
            cache.extend(self.history_store.get_recent_predictions(metric_name, granularity, limit=limit))
            runtime.prediction_history_primed.add(granularity)
        return self._tail_items(cache, limit)

    def _get_recent_anomaly_history(
        self,
        runtime: MetricRuntimeState,
        metric_name: str,
        granularity: str,
        limit: int,
    ) -> list[AnomalyEvent]:
        cache = self._get_or_create_anomaly_cache(runtime, granularity)
        self._validate_cache_limit(metric_name, granularity, limit, cache.maxlen, cache_kind="anomaly")
        if granularity not in runtime.anomaly_history_primed:
            cache.extend(self.history_store.get_recent_anomalies(metric_name, granularity, limit=limit))
            runtime.anomaly_history_primed.add(granularity)
        return self._tail_items(cache, limit)

    def _get_or_create_series_cache(
        self,
        runtime: MetricRuntimeState,
        granularity: str,
    ) -> deque[MetricPoint]:
        cache = runtime.recent_series_points.get(granularity)
        if cache is None:
            cache = deque(maxlen=self._series_cache_limit(granularity))
            runtime.recent_series_points[granularity] = cache
        return cache

    def _get_or_create_prediction_cache(
        self,
        runtime: MetricRuntimeState,
        granularity: str,
    ) -> deque[PredictionResult]:
        cache = runtime.recent_predictions.get(granularity)
        if cache is None:
            cache = deque(maxlen=self.settings.prediction_window)
            runtime.recent_predictions[granularity] = cache
        return cache

    def _get_or_create_anomaly_cache(
        self,
        runtime: MetricRuntimeState,
        granularity: str,
    ) -> deque[AnomalyEvent]:
        cache = runtime.recent_anomalies.get(granularity)
        if cache is None:
            cache = deque(maxlen=self._anomaly_cache_limit(granularity))
            runtime.recent_anomalies[granularity] = cache
        return cache

    def _series_cache_limit(self, granularity: str) -> int:
        if granularity == "1s":
            return self.settings.second_history_window
        return self.settings.minute_history_window

    def _anomaly_cache_limit(self, granularity: str) -> int:
        if granularity == "1s":
            return self.settings.anomaly_window
        return self.settings.minute_history_window

    def _tail_items(self, items: deque[Any], limit: int) -> list[Any]:
        if limit <= 0:
            return []
        return list(islice(reversed(items), limit))[::-1]

    def _validate_cache_limit(
        self,
        metric_name: str,
        granularity: str,
        limit: int,
        cache_maxlen: int | None,
        cache_kind: str = "series",
    ) -> None:
        if cache_maxlen is None or limit <= cache_maxlen:
            return
        raise ValueError(
            f"Requested {cache_kind} history limit {limit} exceeds cache maxlen {cache_maxlen} "
            f"for {metric_name}/{granularity}"
        )
