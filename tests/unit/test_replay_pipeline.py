from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import noccore.api as api_module
from fastapi.testclient import TestClient

from noccore.api import app
from noccore.alert_layer.policy import AlertPolicy
from noccore.alert_layer.suppressor import AlertSuppressor
from noccore.config.metric_registry import MetricMetadata, MetricRegistry
from noccore.config.settings import PipelineSettings
from noccore.data_layer.repair import MetricRepairer
from noccore.detect_layer.global_detector import GlobalAnomalyDetector
from noccore.detect_layer.periodic_filter import PeriodicityFilter
from noccore.feature_layer.calendar_provider import CalendarProvider
from noccore.feature_layer.feature_service import FeatureService
from noccore.history_layer.store import InMemoryHistoryStore, SQLiteHistoryStore
from noccore.model_layer.minute_huber import MinuteHuberModel
from noccore.model_layer.seasonal_adjustment import SeasonalAdjustmentModel
from noccore.pipeline.online_pipeline import OnlinePipeline
from noccore.pipeline.replay_pipeline import build_demo_registry, generate_demo_points, run_replay
from noccore.schemas.anomaly import AnomalyEvent
from noccore.schemas.metric import MetricPoint
from noccore.schemas.prediction import PredictionResult


class ReplayPipelineTests(unittest.TestCase):
    def test_demo_replay_emits_payment_and_zero_rule_alerts(self) -> None:
        alerts = run_replay(generate_demo_points())

        self.assertTrue(
            any(
                alert.metric_name == "payment.success_rate"
                and alert.status == "firing"
                and alert.alert_level == "P0"
                for alert in alerts
            )
        )
        self.assertTrue(
            any(
                alert.metric_name == "payment.success_rate"
                and alert.status == "recovered"
                for alert in alerts
            )
        )
        self.assertTrue(
            any(
                alert.metric_name == "payment.long_tail_count"
                and alert.alert_type == "zero_rule"
                for alert in alerts
            )
        )

    def test_demo_primary_alert_starts_in_anomaly_window(self) -> None:
        start_ts = 1711180800
        alerts = run_replay(generate_demo_points(start_ts=start_ts))
        primary_alerts = [
            alert
            for alert in alerts
            if alert.metric_name == "payment.success_rate" and alert.status == "firing"
        ]

        self.assertTrue(primary_alerts)
        self.assertGreaterEqual(primary_alerts[0].timestamp, start_ts + 720)
        self.assertLessEqual(primary_alerts[0].timestamp, start_ts + 780)

    def test_process_points_keeps_partial_minute_bucket_until_flush(self) -> None:
        registry = MetricRegistry()
        registry.register(
            MetricMetadata(
                metric_name="payment.partial_bucket_count",
                business_line="AA",
                priority="P2",
                metric_type="count",
                core_link=False,
            )
        )
        pipeline = OnlinePipeline(registry=registry)
        points: list[MetricPoint] = []
        start_ts = 1711180800
        for offset in range(1200):
            value = 10.0 if offset < 600 else 0.0
            points.append(
                MetricPoint(
                    metric_name="payment.partial_bucket_count",
                    timestamp=start_ts + offset,
                    value=value,
                    business_line="AA",
                    priority="P2",
                    metric_type="count",
                    tags={},
                )
            )

        alerts_before_flush = pipeline.process_points(points, flush=False)
        alerts_after_flush = pipeline.flush()

        self.assertEqual(alerts_before_flush, [])
        self.assertTrue(
            any(
                alert.metric_name == "payment.partial_bucket_count"
                and alert.alert_type == "zero_rule"
                for alert in alerts_after_flush
            )
        )

    def test_feature_service_uses_true_time_window_for_rolling_stats(self) -> None:
        service = FeatureService(CalendarProvider(timezone="Asia/Shanghai"), timezone="Asia/Shanghai")
        point = MetricPoint(
            metric_name="payment.success_rate",
            timestamp=1711180800,
            value=10.0,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )
        history = [
            MetricPoint(
                metric_name="payment.success_rate",
                timestamp=point.timestamp - 8 * 24 * 3600,
                value=1000.0,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True},
            ),
            MetricPoint(
                metric_name="payment.success_rate",
                timestamp=point.timestamp - 60,
                value=10.0,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True},
            ),
        ]
        metadata = MetricRegistry().get_or_create(point)
        features = service.build(point, history, [], metadata)

        self.assertLess(features["rolling_mean_7d"], 100.0)

    def test_stable_stream_does_not_emit_false_alerts(self) -> None:
        pipeline = OnlinePipeline(registry=build_demo_registry())
        start_ts = 1711180800
        points = [
            MetricPoint(
                metric_name="payment.success_rate",
                timestamp=start_ts + offset,
                value=0.962,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True, "channel": "alipay"},
            )
            for offset in range(900)
        ]

        alerts = pipeline.process_points(points, flush=True)
        self.assertEqual(alerts, [])

    def test_large_gap_marks_point_untrusted(self) -> None:
        pipeline = OnlinePipeline(registry=build_demo_registry())
        first = MetricPoint(
            metric_name="payment.success_rate",
            timestamp=1711180800,
            value=0.962,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )
        second = MetricPoint(
            metric_name="payment.success_rate",
            timestamp=1711180810,
            value=0.961,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )

        pipeline.process_point(first)
        pipeline.process_point(second)
        runtime_point = pipeline._state["payment.success_rate"].last_raw_point

        self.assertIsNotNone(runtime_point)
        self.assertFalse(runtime_point.trusted)
        self.assertEqual(runtime_point.tags["missing_gap"], 9)

    def test_eligibility_assessment_is_throttled(self) -> None:
        registry = MetricRegistry()
        registry.register(
            MetricMetadata(
                metric_name="payment.eligibility_probe",
                business_line="AA",
                priority="P1",
                metric_type="rate",
                core_link=False,
                force_ai=False,
            )
        )
        pipeline = OnlinePipeline(registry=registry)
        points = [
            MetricPoint(
                metric_name="payment.eligibility_probe",
                timestamp=1711180800 + offset,
                value=0.9,
                business_line="AA",
                priority="P1",
                metric_type="rate",
                tags={},
            )
            for offset in range(30)
        ]

        with patch.object(pipeline.eligibility, "assess", wraps=pipeline.eligibility.assess) as mocked_assess:
            pipeline.process_points(points, flush=True)

        self.assertEqual(mocked_assess.call_count, 1)

    def test_pipeline_eligibility_uses_external_history_store(self) -> None:
        settings = PipelineSettings(history_weeks_required=0)
        history_store = InMemoryHistoryStore(settings)
        registry = MetricRegistry()
        registry.register(
            MetricMetadata(
                metric_name="payment.external_eligibility_probe",
                business_line="AA",
                priority="P1",
                metric_type="rate",
                core_link=False,
                force_ai=False,
            )
        )
        history_store.append_raw_point(
            MetricPoint(
                metric_name="payment.external_eligibility_probe",
                timestamp=1711180700,
                value=0.91,
                business_line="AA",
                priority="P1",
                metric_type="rate",
                tags={},
            ),
            timezone="Asia/Shanghai",
            is_holiday=False,
        )
        history_store.append_raw_point(
            MetricPoint(
                metric_name="payment.external_eligibility_probe",
                timestamp=1711180750,
                value=0.92,
                business_line="AA",
                priority="P1",
                metric_type="rate",
                tags={},
            ),
            timezone="Asia/Shanghai",
            is_holiday=False,
        )
        pipeline = OnlinePipeline(settings=settings, registry=registry, history_store=history_store)

        with patch.object(pipeline.eligibility, "assess", wraps=pipeline.eligibility.assess) as mocked_assess:
            pipeline.process_point(
                MetricPoint(
                    metric_name="payment.external_eligibility_probe",
                    timestamp=1711180800,
                    value=0.93,
                    business_line="AA",
                    priority="P1",
                    metric_type="rate",
                    tags={},
                )
            )

        assessed_points = mocked_assess.call_args.args[0]
        self.assertEqual(len(list(assessed_points)), 3)

    def test_seasonal_alignment_reads_external_history_store(self) -> None:
        history_store = InMemoryHistoryStore(PipelineSettings())
        model = SeasonalAdjustmentModel(timezone="Asia/Shanghai", history_store=history_store)
        metadata = MetricMetadata(
            metric_name="payment.aligned_history_probe",
            business_line="AA",
            priority="P0",
            metric_type="rate",
            core_link=True,
            force_ai=True,
        )
        ts_non_holiday = 1710724800  # 2024-03-18 12:00:00 +08:00, Monday
        ts_holiday = ts_non_holiday + 7 * 24 * 3600
        non_holiday_point = MetricPoint(
            metric_name="payment.aligned_history_probe",
            timestamp=ts_non_holiday,
            value=10.0,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )
        holiday_point = MetricPoint(
            metric_name="payment.aligned_history_probe",
            timestamp=ts_holiday,
            value=20.0,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )

        first_plan = model.predict(
            model.get_state(non_holiday_point.metric_name),
            non_holiday_point,
            history_points=[],
            prediction_history=[],
            features={
                "isHoliday": False,
                "isRCA": False,
                "isHotSpot": False,
                "baseDayDiff": 0.0,
            },
            metadata=metadata,
        )
        model.apply(non_holiday_point.metric_name, first_plan)
        history_store.append_raw_point(non_holiday_point, timezone="Asia/Shanghai", is_holiday=False)
        history_store.append_raw_point(holiday_point, timezone="Asia/Shanghai", is_holiday=True)

        second_plan = model.predict(
            model.get_state(holiday_point.metric_name),
            holiday_point.model_copy(update={"timestamp": ts_holiday + 7 * 24 * 3600, "value": 21.0}),
            history_points=[],
            prediction_history=[],
            features={
                "isHoliday": True,
                "isRCA": False,
                "isHotSpot": False,
                "baseDayDiff": 0.0,
            },
            metadata=metadata,
        )
        prediction = second_plan.prediction

        self.assertGreater(prediction.predict_value, 19.0)
        self.assertLess(prediction.predict_value, 21.5)


class ApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        client = TestClient(app)
        previous_active_requests = api_module._active_requests
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(api_module._active_requests, previous_active_requests)

    def test_flush_endpoint_returns_alerts_for_completed_stream(self) -> None:
        client = TestClient(app)
        registry_payload = [
            {
                "metric_name": "api.partial_bucket_count",
                "business_line": "AA",
                "priority": "P2",
                "metric_type": "count",
                "core_link": False,
            }
        ]
        reload_response = client.post("/v1/reload-registry", json=registry_payload)
        self.assertEqual(reload_response.status_code, 200)

        start_ts = 1711180800
        points = []
        for offset in range(1200):
            value = 10.0 if offset < 600 else 0.0
            points.append(
                {
                    "metric_name": "api.partial_bucket_count",
                    "timestamp": start_ts + offset,
                    "value": value,
                    "business_line": "AA",
                    "priority": "P2",
                    "metric_type": "count",
                    "tags": {},
                }
            )

        ingest_response = client.post("/v1/ingest", json=points)
        self.assertEqual(ingest_response.status_code, 200)
        self.assertEqual(ingest_response.json(), [])

        flush_response = client.post("/v1/flush")
        self.assertEqual(flush_response.status_code, 200)
        self.assertTrue(
            any(item["metric_name"] == "api.partial_bucket_count" for item in flush_response.json())
        )

    def test_reload_registry_preserves_pipeline_warm_state(self) -> None:
        original_pipeline = api_module.pipeline
        try:
            warm_pipeline = OnlinePipeline(registry=build_demo_registry())
            warm_pipeline.process_point(
                MetricPoint(
                    metric_name="payment.success_rate",
                    timestamp=1711180800,
                    value=0.96,
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    tags={"force_ai": True, "channel": "alipay"},
                )
            )
            api_module.pipeline = warm_pipeline
            client = TestClient(app)

            before_state = warm_pipeline.second_model._state["payment.success_rate"]
            response = client.post(
                "/v1/reload-registry",
                json=[
                    {
                        "metric_name": "payment.success_rate",
                        "business_line": "AA",
                        "priority": "P1",
                        "metric_type": "rate",
                        "core_link": True,
                        "force_ai": True,
                    }
                ],
            )

            self.assertEqual(response.status_code, 200)
            self.assertIs(api_module.pipeline, warm_pipeline)
            self.assertIs(api_module.pipeline.second_model._state["payment.success_rate"], before_state)
            self.assertEqual(api_module.pipeline.registry.get("payment.success_rate").priority, "P1")
        finally:
            api_module.pipeline = original_pipeline

    def test_lifespan_flushes_and_closes_history_store_on_shutdown(self) -> None:
        class TrackingHistoryStore(InMemoryHistoryStore):
            def __init__(self) -> None:
                super().__init__(PipelineSettings())
                self.flushed = False
                self.closed = False

            def flush_writes(self) -> None:
                self.flushed = True

            def close(self) -> None:
                self.closed = True

        original_pipeline = api_module.pipeline
        history_store = TrackingHistoryStore()
        try:
            api_module.pipeline = OnlinePipeline(registry=build_demo_registry(), history_store=history_store)
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)

            self.assertTrue(history_store.flushed)
            self.assertTrue(history_store.closed)
        finally:
            api_module.pipeline = original_pipeline

    def test_ingest_only_holds_pipeline_lock_for_reference_read(self) -> None:
        class BlockingPipeline:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def process_points(self, points):
                self.started.set()
                self.release.wait(timeout=5)
                return []

        original_pipeline = api_module.pipeline
        blocking_pipeline = BlockingPipeline()
        point = MetricPoint(
            metric_name="payment.lock_probe",
            timestamp=1711180800,
            value=1.0,
            business_line="AA",
            priority="P1",
            metric_type="rate",
            tags={},
        )
        try:
            api_module.pipeline = blocking_pipeline
            worker = threading.Thread(target=lambda: api_module.ingest([point]), daemon=True)
            worker.start()

            self.assertTrue(blocking_pipeline.started.wait(timeout=1))
            acquired = api_module._pipeline_lock.acquire(timeout=1)
            if acquired:
                api_module._pipeline_lock.release()
            blocking_pipeline.release.set()
            worker.join(timeout=1)

            self.assertTrue(acquired)
            self.assertFalse(worker.is_alive())
        finally:
            api_module.pipeline = original_pipeline

    def test_request_tracker_waits_for_inflight_requests(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def hold_request() -> None:
            with api_module._track_request():
                started.set()
                release.wait(timeout=5)

        worker = threading.Thread(target=hold_request, daemon=True)
        worker.start()

        self.assertTrue(started.wait(timeout=1))
        self.assertFalse(api_module._wait_for_inflight_requests(0.05))
        release.set()
        worker.join(timeout=1)
        self.assertTrue(api_module._wait_for_inflight_requests(0.5))


class PerformanceGuardTests(unittest.TestCase):
    def test_pipeline_runtime_history_caches_avoid_repeated_store_reads(self) -> None:
        settings = PipelineSettings(
            jump_clip_history_window=5,
            second_history_window=5,
            minute_history_window=5,
            prediction_window=5,
            anomaly_window=5,
        )

        class CountingHistoryStore(InMemoryHistoryStore):
            def __init__(self, local_settings: PipelineSettings) -> None:
                super().__init__(local_settings)
                self.raw_reads = 0
                self.series_reads = 0
                self.prediction_reads = 0
                self.anomaly_reads = 0

            def get_recent_raw_points(self, metric_name: str, limit: int | None = None, since_timestamp: int | None = None) -> list[MetricPoint]:
                self.raw_reads += 1
                return super().get_recent_raw_points(metric_name, limit=limit, since_timestamp=since_timestamp)

            def get_recent_series_points(
                self,
                metric_name: str,
                granularity: str,
                limit: int | None = None,
                since_timestamp: int | None = None,
            ) -> list[MetricPoint]:
                self.series_reads += 1
                return super().get_recent_series_points(
                    metric_name,
                    granularity,
                    limit=limit,
                    since_timestamp=since_timestamp,
                )

            def get_recent_predictions(
                self,
                metric_name: str,
                granularity: str,
                limit: int | None = None,
                since_timestamp: int | None = None,
            ) -> list[PredictionResult]:
                self.prediction_reads += 1
                return super().get_recent_predictions(
                    metric_name,
                    granularity,
                    limit=limit,
                    since_timestamp=since_timestamp,
                )

            def get_recent_anomalies(
                self,
                metric_name: str,
                granularity: str,
                limit: int | None = None,
                since_timestamp: int | None = None,
            ) -> list[AnomalyEvent]:
                self.anomaly_reads += 1
                return super().get_recent_anomalies(
                    metric_name,
                    granularity,
                    limit=limit,
                    since_timestamp=since_timestamp,
                )

        history_store = CountingHistoryStore(settings)
        pipeline = OnlinePipeline(settings=settings, registry=build_demo_registry(), history_store=history_store)
        metric_name = "payment.cache_probe"
        point = MetricPoint(
            metric_name=metric_name,
            timestamp=1711180800,
            value=0.95,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )
        prediction = PredictionResult(
            metric_name=metric_name,
            timestamp=1711180800,
            granularity="1s",
            actual_value=0.95,
            predict_value=0.95,
            lower_bound=0.90,
            upper_bound=1.00,
            model_name="seasonal_adjustment",
            model_version="sec_v1",
            features={},
        )
        anomaly = AnomalyEvent(
            metric_name=metric_name,
            timestamp=1711180800,
            granularity="1s",
            global_abnormal=False,
            local_score=0.0,
            is_periodic=False,
            z_score=0.0,
            lower_bound=-0.1,
            upper_bound=0.1,
            severity=0.0,
            direction="normal",
            abnormal_labels=[],
            confidence_score=0.0,
            confidence_components={},
        )
        history_store.append_raw_point(point, timezone="Asia/Shanghai", is_holiday=False)
        history_store.append_series_point(point)
        history_store.append_prediction(prediction)
        history_store.append_anomaly(anomaly)
        runtime = pipeline._get_or_create_runtime_state(metric_name)

        self.assertEqual(len(pipeline._get_recent_raw_history(runtime, metric_name, 5)), 1)
        self.assertEqual(len(pipeline._get_recent_raw_history(runtime, metric_name, 5)), 1)
        self.assertEqual(len(pipeline._get_recent_series_history(runtime, metric_name, "1s", 5)), 1)
        self.assertEqual(len(pipeline._get_recent_series_history(runtime, metric_name, "1s", 5)), 1)
        self.assertEqual(len(pipeline._get_recent_prediction_history(runtime, metric_name, "1s", 5)), 1)
        self.assertEqual(len(pipeline._get_recent_prediction_history(runtime, metric_name, "1s", 5)), 1)
        self.assertEqual(len(pipeline._get_recent_anomaly_history(runtime, metric_name, "1s", 5)), 1)
        self.assertEqual(len(pipeline._get_recent_anomaly_history(runtime, metric_name, "1s", 5)), 1)

        self.assertEqual(history_store.raw_reads, 1)
        self.assertEqual(history_store.series_reads, 1)
        self.assertEqual(history_store.prediction_reads, 1)
        self.assertEqual(history_store.anomaly_reads, 1)

    def test_pipeline_reuses_single_calendar_context_for_second_ai_point(self) -> None:
        class CountingCalendarProvider(CalendarProvider):
            def __init__(self, timezone: str) -> None:
                super().__init__(timezone=timezone)
                self.context_calls = 0
                self.holiday_calls = 0

            def get_context(self, metric_name: str, timestamp: int) -> dict[str, bool]:
                self.context_calls += 1
                return super().get_context(metric_name, timestamp)

            def is_holiday(self, timestamp: int) -> bool:
                self.holiday_calls += 1
                return super().is_holiday(timestamp)

        provider = CountingCalendarProvider("Asia/Shanghai")
        pipeline = OnlinePipeline(registry=build_demo_registry(), calendar_provider=provider)
        pipeline.process_point(
            MetricPoint(
                metric_name="payment.success_rate",
                timestamp=1711180800,
                value=0.95,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True},
            )
        )

        self.assertEqual(provider.context_calls, 1)
        self.assertEqual(provider.holiday_calls, 0)

    def test_pipeline_raises_when_requested_history_limit_exceeds_cache_capacity(self) -> None:
        settings = PipelineSettings(second_history_window=4, minute_history_window=4, prediction_window=4, anomaly_window=4)
        pipeline = OnlinePipeline(settings=settings, registry=build_demo_registry())
        runtime = pipeline._get_or_create_runtime_state("payment.limit_probe")

        with self.assertRaises(ValueError):
            pipeline._get_recent_series_history(runtime, "payment.limit_probe", "1s", 5)

    def test_seasonal_predict_does_not_mutate_state_before_apply(self) -> None:
        history_store = InMemoryHistoryStore(PipelineSettings())
        model = SeasonalAdjustmentModel(timezone="Asia/Shanghai", history_store=history_store)
        metric_name = "payment.seasonal_split_probe"
        history_store.append_raw_point(
            MetricPoint(
                metric_name=metric_name,
                timestamp=1711180740,
                value=10.0,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True},
            ),
            timezone="Asia/Shanghai",
            is_holiday=False,
        )
        state = model.get_state(metric_name)
        initial_rate = state.correction_rate
        initial_residual_std = state.residual_std
        initial_residual_history_len = len(state.residual_history)
        initial_fallback_logged = state.fallback_logged
        point = MetricPoint(
            metric_name=metric_name,
            timestamp=1711180800,
            value=12.0,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )
        metadata = MetricRegistry().get_or_create(point)

        plan = model.predict(
            state,
            point,
            history_points=[],
            prediction_history=[],
            features={
                "isHoliday": False,
                "isRCA": False,
                "isHotSpot": False,
                "baseDayDiff": 0.0,
            },
            metadata=metadata,
        )

        self.assertEqual(state.correction_rate, initial_rate)
        self.assertEqual(state.residual_std, initial_residual_std)
        self.assertEqual(len(state.residual_history), initial_residual_history_len)
        self.assertEqual(state.fallback_logged, initial_fallback_logged)

        model.apply(metric_name, plan)

        self.assertEqual(state.correction_rate, plan.correction_rate)
        self.assertEqual(state.residual_std, plan.residual_std)
        self.assertEqual(state.fallback_logged, plan.fallback_logged)
        self.assertGreater(len(state.residual_history), initial_residual_history_len)

    def test_seasonal_predict_uses_runtime_raw_history_before_store_fallback(self) -> None:
        class CountingHistoryStore(InMemoryHistoryStore):
            def __init__(self, settings: PipelineSettings) -> None:
                super().__init__(settings)
                self.raw_reads = 0

            def get_recent_raw_points(
                self,
                metric_name: str,
                limit: int | None = None,
                since_timestamp: int | None = None,
            ) -> list[MetricPoint]:
                self.raw_reads += 1
                return super().get_recent_raw_points(metric_name, limit=limit, since_timestamp=since_timestamp)

        history_store = CountingHistoryStore(PipelineSettings())
        model = SeasonalAdjustmentModel(timezone="Asia/Shanghai", history_store=history_store)
        point = MetricPoint(
            metric_name="payment.raw_history_probe",
            timestamp=1711180800,
            value=12.0,
            business_line="AA",
            priority="P0",
            metric_type="rate",
            tags={"force_ai": True},
        )

        plan = model.predict(
            model.get_state(point.metric_name),
            point,
            history_points=[],
            prediction_history=[],
            features={
                "isHoliday": False,
                "isRCA": False,
                "isHotSpot": False,
                "baseDayDiff": 0.0,
            },
            metadata=MetricRegistry().get_or_create(point),
            raw_history_points=[
                point.model_copy(update={"timestamp": point.timestamp - 60, "value": 10.0}),
                point.model_copy(update={"timestamp": point.timestamp - 30, "value": 11.0}),
            ],
        )

        self.assertGreater(plan.prediction.predict_value, 0.0)
        self.assertEqual(history_store.raw_reads, 0)

    def test_alert_policy_decide_does_not_mutate_state_before_apply(self) -> None:
        policy = AlertPolicy(PipelineSettings(), AlertSuppressor(300))
        metadata = MetricMetadata(
            metric_name="payment.alert_split_probe",
            business_line="AA",
            priority="P0",
            metric_type="rate",
            core_link=True,
            force_ai=True,
        )
        prediction = PredictionResult(
            metric_name="payment.alert_split_probe",
            timestamp=1711180800,
            granularity="1s",
            actual_value=0.8,
            predict_value=0.95,
            lower_bound=0.9,
            upper_bound=1.0,
            model_name="seasonal_adjustment",
            model_version="sec_v1",
            features={},
        )
        anomaly = AnomalyEvent(
            metric_name="payment.alert_split_probe",
            timestamp=1711180800,
            granularity="1s",
            global_abnormal=True,
            local_score=0.9,
            is_periodic=False,
            z_score=-4.0,
            lower_bound=-0.1,
            upper_bound=0.1,
            severity=1.0,
            direction="down",
            abnormal_labels=["global"],
            confidence_score=0.95,
            confidence_components={},
        )
        state = policy.get_state("payment.alert_split_probe")

        plan = policy.decide(metadata, prediction, anomaly, state)

        self.assertFalse(policy.get_state("payment.alert_split_probe").incident.is_open)
        self.assertEqual(policy.get_state("payment.alert_split_probe").suppressor_last_emit_timestamp, None)

        policy.apply("payment.alert_split_probe", plan)

        applied = policy.get_state("payment.alert_split_probe")
        self.assertTrue(applied.incident.is_open)
        self.assertEqual(applied.suppressor_last_emit_timestamp, 1711180800)

    def test_sqlite_history_store_rolls_back_failed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_store = SQLiteHistoryStore(Path(tmp_dir) / "history.db", batch_size=1000)
            first_point = MetricPoint(
                metric_name="payment.rollback_probe",
                timestamp=1711180800,
                value=1.0,
                business_line="AA",
                priority="P1",
                metric_type="rate",
                tags={},
            )
            second_point = first_point.model_copy(update={"timestamp": 1711180810, "value": 2.0})

            history_store.append_raw_point(first_point, timezone="Asia/Shanghai", is_holiday=False)

            with self.assertRaises(sqlite3.Error):
                history_store._execute_write(
                    "INSERT INTO missing_table(value) VALUES (?)",
                    (1,),
                    "forced_failure",
                )

            self.assertEqual(history_store._pending_writes, 0)
            self.assertEqual(history_store.get_recent_raw_points("payment.rollback_probe", limit=5), [])

            history_store.append_raw_point(second_point, timezone="Asia/Shanghai", is_holiday=False)
            history_store.flush_writes()

            self.assertEqual(
                [item.timestamp for item in history_store.get_recent_raw_points("payment.rollback_probe", limit=5)],
                [1711180810],
            )

    def test_global_detector_reuses_cached_kde_bounds(self) -> None:
        detector = GlobalAnomalyDetector(PipelineSettings())
        prediction = PredictionResult(
            metric_name="payment.kde_probe",
            timestamp=1711180800,
            granularity="1s",
            actual_value=0.95,
            predict_value=0.95,
            lower_bound=0.0,
            upper_bound=1.0,
            model_name="seasonal_adjustment",
            model_version="sec_v1",
            features={"rolling_mean_7d": 0.95, "rolling_std_7d": 0.01},
        )

        with patch("noccore.detect_layer.global_detector.gaussian_kde_bounds") as mocked_kde:
            mocked_kde.return_value = (-0.2, 0.2)
            for offset in range(10):
                plan = detector.detect(
                    detector.get_state(prediction.metric_name),
                    prediction.model_copy(update={"timestamp": prediction.timestamp + offset}),
                )
                detector.apply(prediction.metric_name, plan)

        self.assertEqual(mocked_kde.call_count, 1)

    def test_global_detector_apply_appends_history_in_place(self) -> None:
        detector = GlobalAnomalyDetector(PipelineSettings())
        metric_name = "payment.kde_incremental_probe"
        state = detector.get_state(metric_name)
        history_ref = state.z_history["mid"]
        history_ref.extend([0.01, -0.02])
        prediction = PredictionResult(
            metric_name=metric_name,
            timestamp=1711180800,
            granularity="1s",
            actual_value=1.00,
            predict_value=1.00,
            lower_bound=0.0,
            upper_bound=1.0,
            model_name="seasonal_adjustment",
            model_version="sec_v1",
            features={"rolling_mean_7d": 1.00, "rolling_std_7d": 0.01},
        )

        plan = detector.detect(state, prediction)
        detector.apply(metric_name, plan)

        self.assertIs(state.z_history["mid"], history_ref)
        self.assertEqual(len(history_ref), 3)
        self.assertAlmostEqual(history_ref[-1], plan.result["z_score"])

    def test_pipeline_cleans_stale_metric_state(self) -> None:
        settings = PipelineSettings(state_ttl_sec=50, state_cleanup_every_points=1)
        registry = MetricRegistry()
        registry.register(
            MetricMetadata(
                metric_name="payment.old_metric",
                business_line="AA",
                priority="P0",
                metric_type="rate",
                core_link=True,
                force_ai=True,
            )
        )
        registry.register(
            MetricMetadata(
                metric_name="payment.new_metric",
                business_line="AA",
                priority="P0",
                metric_type="rate",
                core_link=True,
                force_ai=True,
            )
        )
        pipeline = OnlinePipeline(settings=settings, registry=registry)

        pipeline.process_point(
            MetricPoint(
                metric_name="payment.old_metric",
                timestamp=1711180800,
                value=0.96,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True},
            )
        )
        pipeline.process_point(
            MetricPoint(
                metric_name="payment.new_metric",
                timestamp=1711180900,
                value=0.97,
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True},
            )
        )

        self.assertNotIn("payment.old_metric", pipeline._state)
        self.assertIsNone(pipeline.registry.get_eligibility("payment.old_metric"))
        self.assertEqual(pipeline.history_store.get_recent_raw_points("payment.old_metric"), [])

    def test_sqlite_history_store_supports_cross_pipeline_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_store = SQLiteHistoryStore(Path(tmp_dir) / "history.db")
            registry = build_demo_registry()
            pipeline_a = OnlinePipeline(registry=registry, history_store=history_store)
            pipeline_a.process_point(
                MetricPoint(
                    metric_name="payment.success_rate",
                    timestamp=1710724800,
                    value=0.95,
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    tags={"force_ai": True, "channel": "alipay"},
                )
            )

            pipeline_b = OnlinePipeline(registry=registry, history_store=history_store)
            seasonal_model = pipeline_b.second_model
            plan = seasonal_model.predict(
                seasonal_model.get_state("payment.success_rate"),
                MetricPoint(
                    metric_name="payment.success_rate",
                    timestamp=1711329600,
                    value=0.96,
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    tags={"force_ai": True, "channel": "alipay"},
                ),
                history_points=[],
                prediction_history=[],
                features={
                    "isHoliday": False,
                    "isRCA": False,
                    "isHotSpot": False,
                    "baseDayDiff": 0.0,
                },
                metadata=registry.get("payment.success_rate"),
            )
            prediction = plan.prediction

            self.assertGreater(prediction.predict_value, 0.9)
            self.assertLess(prediction.predict_value, 1.0)

    def test_pipeline_cleans_sqlite_runtime_but_keeps_history_until_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = PipelineSettings(
                state_ttl_sec=50,
                state_cleanup_every_points=1,
                data_retention_sec=3600,
            )
            registry = MetricRegistry()
            registry.register(
                MetricMetadata(
                    metric_name="payment.sqlite_old_metric",
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    core_link=True,
                    force_ai=True,
                )
            )
            registry.register(
                MetricMetadata(
                    metric_name="payment.sqlite_new_metric",
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    core_link=True,
                    force_ai=True,
                )
            )
            history_store = SQLiteHistoryStore(Path(tmp_dir) / "history.db")
            pipeline = OnlinePipeline(settings=settings, registry=registry, history_store=history_store)

            pipeline.process_point(
                MetricPoint(
                    metric_name="payment.sqlite_old_metric",
                    timestamp=1711180800,
                    value=0.96,
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    tags={"force_ai": True},
                )
            )
            pipeline.process_point(
                MetricPoint(
                    metric_name="payment.sqlite_new_metric",
                    timestamp=1711180900,
                    value=0.97,
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    tags={"force_ai": True},
                )
            )

            self.assertNotIn("payment.sqlite_old_metric", pipeline._state)
            self.assertIsNone(pipeline.registry.get_eligibility("payment.sqlite_old_metric"))
            self.assertEqual(
                len(history_store.get_recent_raw_points("payment.sqlite_old_metric", limit=5)),
                1,
            )

    def test_sqlite_history_store_persists_predictions_and_anomalies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_store = SQLiteHistoryStore(Path(tmp_dir) / "history.db")
            prediction = PredictionResult(
                metric_name="payment.persistence_probe",
                timestamp=1711180800,
                granularity="1m",
                actual_value=0.8,
                predict_value=0.95,
                lower_bound=0.9,
                upper_bound=1.0,
                model_name="minute_huber",
                model_version="min_v1",
                features={"source": "test"},
            )
            anomaly = AnomalyEvent(
                metric_name="payment.persistence_probe",
                timestamp=1711180800,
                granularity="1m",
                global_abnormal=True,
                local_score=0.75,
                is_periodic=False,
                z_score=-0.15,
                lower_bound=-0.05,
                upper_bound=0.05,
                severity=1.0,
                direction="down",
                abnormal_labels=["global", "trend"],
                confidence_score=0.86,
                confidence_components={"metric_reliability": 0.9},
            )

            history_store.append_prediction(prediction)
            history_store.append_anomaly(anomaly)

            predictions = history_store.get_recent_predictions("payment.persistence_probe", "1m", limit=5)
            anomalies = history_store.get_recent_anomalies("payment.persistence_probe", "1m", limit=5)

            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0].predict_value, 0.95)
            self.assertEqual(len(anomalies), 1)
            self.assertEqual(anomalies[0].abnormal_labels, ["global", "trend"])

    def test_pipeline_flushes_sqlite_batches_for_cross_connection_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "history.db"
            history_store = SQLiteHistoryStore(db_path, batch_size=1000)
            pipeline = OnlinePipeline(registry=build_demo_registry(), history_store=history_store)

            pipeline.process_points(
                [
                    MetricPoint(
                        metric_name="payment.success_rate",
                        timestamp=1711180800,
                        value=0.95,
                        business_line="AA",
                        priority="P0",
                        metric_type="rate",
                        tags={"force_ai": True, "channel": "alipay"},
                    )
                ],
                flush=False,
            )

            another_store = SQLiteHistoryStore(db_path, batch_size=1000)
            points = another_store.get_recent_raw_points("payment.success_rate", limit=5)

            self.assertEqual(len(points), 1)

    def test_process_point_flushes_single_sqlite_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "history.db"
            history_store = SQLiteHistoryStore(db_path, batch_size=1000)
            pipeline = OnlinePipeline(registry=build_demo_registry(), history_store=history_store)

            pipeline.process_point(
                MetricPoint(
                    metric_name="payment.success_rate",
                    timestamp=1711180800,
                    value=0.95,
                    business_line="AA",
                    priority="P0",
                    metric_type="rate",
                    tags={"force_ai": True, "channel": "alipay"},
                )
            )

            another_store = SQLiteHistoryStore(db_path, batch_size=1000)
            points = another_store.get_recent_raw_points("payment.success_rate", limit=5)

            self.assertEqual(len(points), 1)

    def test_sqlite_history_store_prunes_old_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_store = SQLiteHistoryStore(Path(tmp_dir) / "history.db", batch_size=1000)
            old_ts = 1711180800
            new_ts = old_ts + 500
            point_old = MetricPoint(
                metric_name="payment.retention_probe",
                timestamp=old_ts,
                value=1.0,
                business_line="AA",
                priority="P1",
                metric_type="rate",
                tags={},
                granularity="1s",
            )
            point_new = point_old.model_copy(update={"timestamp": new_ts, "value": 2.0})
            prediction_old = PredictionResult(
                metric_name="payment.retention_probe",
                timestamp=old_ts,
                granularity="1s",
                actual_value=1.0,
                predict_value=1.0,
                lower_bound=0.9,
                upper_bound=1.1,
                model_name="seasonal_adjustment",
                model_version="sec_v1",
                features={},
            )
            anomaly_old = AnomalyEvent(
                metric_name="payment.retention_probe",
                timestamp=old_ts,
                granularity="1s",
                global_abnormal=False,
                local_score=0.0,
                is_periodic=False,
                z_score=0.0,
                lower_bound=-0.1,
                upper_bound=0.1,
                severity=0.0,
                direction="normal",
                abnormal_labels=[],
                confidence_score=0.0,
                confidence_components={},
            )

            history_store.append_raw_point(point_old, timezone="Asia/Shanghai", is_holiday=False)
            history_store.append_raw_point(point_new, timezone="Asia/Shanghai", is_holiday=False)
            history_store.append_series_point(point_old)
            history_store.append_prediction(prediction_old)
            history_store.append_anomaly(anomaly_old)
            history_store.flush_writes()

            deleted = history_store.prune_old_data(old_ts + 1)

            self.assertGreaterEqual(deleted, 4)
            self.assertEqual(
                [item.timestamp for item in history_store.get_recent_raw_points("payment.retention_probe", limit=5)],
                [new_ts],
            )
            self.assertEqual(
                history_store.get_recent_predictions("payment.retention_probe", "1s", limit=5),
                [],
            )

    def test_sqlite_history_store_serializes_concurrent_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_store = SQLiteHistoryStore(Path(tmp_dir) / "history.db", batch_size=1000)

            def append_point(index: int) -> None:
                history_store.append_raw_point(
                    MetricPoint(
                        metric_name="payment.concurrent_probe",
                        timestamp=1711180800 + index,
                        value=float(index),
                        business_line="AA",
                        priority="P1",
                        metric_type="rate",
                        tags={},
                    ),
                    timezone="Asia/Shanghai",
                    is_holiday=False,
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(append_point, range(100)))
            history_store.flush_writes()

            self.assertEqual(
                len(history_store.get_recent_raw_points("payment.concurrent_probe", limit=200)),
                100,
            )

    def test_periodicity_filter_skips_impossible_periods_for_short_history(self) -> None:
        periodicity_filter = PeriodicityFilter(PipelineSettings(second_history_window=4000))

        periods = periodicity_filter._candidate_periods("1s", 4001)

        self.assertEqual(periods, [60, 300])

    def test_repairer_uses_zero_fill_for_count_gaps(self) -> None:
        repairer = MetricRepairer(PipelineSettings(max_missing_fill=3))
        previous = MetricPoint(
            metric_name="payment.count_gap_probe",
            timestamp=1711180800,
            value=8.0,
            business_line="AA",
            priority="P2",
            metric_type="count",
            tags={},
        )
        current = previous.model_copy(update={"timestamp": 1711180803, "value": 14.0})

        filled, _ = repairer.repair_continuity(previous, current)

        self.assertEqual([item.value for item in filled], [0.0, 0.0])

    def test_minute_huber_clean_history_aligns_tail_windows(self) -> None:
        model = MinuteHuberModel()
        history_points = [
            MetricPoint(
                metric_name="payment.minute_clean_probe",
                timestamp=1711180800 + index,
                value=float(index + 1),
                business_line="AA",
                priority="P1",
                metric_type="rate",
                tags={},
                granularity="1m",
            )
            for index in range(5)
        ]
        prediction_history = [
            PredictionResult(
                metric_name="payment.minute_clean_probe",
                timestamp=1711180803 + index,
                granularity="1m",
                actual_value=0.0,
                predict_value=float(30 + 10 * index),
                lower_bound=0.0,
                upper_bound=100.0,
                model_name="minute_huber",
                model_version="min_v1",
                features={},
            )
            for index in range(2)
        ]

        cleaned = model._clean_history(history_points, prediction_history, [False, True])

        self.assertEqual(cleaned, [4.0, 40.0])


if __name__ == "__main__":
    unittest.main()
