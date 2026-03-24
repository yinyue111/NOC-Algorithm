from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI

from noccore.config.metric_registry import MetricMetadata, MetricRegistry
from noccore.config.settings import DEFAULT_SETTINGS
from noccore.history_layer.store import InMemoryHistoryStore, SQLiteHistoryStore
from noccore.pipeline.online_pipeline import OnlinePipeline
from noccore.pipeline.replay_pipeline import build_demo_registry
from noccore.schemas.alert import AlertEvent
from noccore.schemas.metric import MetricPoint


logger = logging.getLogger(__name__)


def _build_history_store() -> InMemoryHistoryStore | SQLiteHistoryStore:
    history_db = os.getenv("NOCCORE_HISTORY_DB")
    if history_db:
        return SQLiteHistoryStore(history_db, batch_size=DEFAULT_SETTINGS.history_commit_batch_size)
    return InMemoryHistoryStore(DEFAULT_SETTINGS)


def _build_registry(payload: list[dict]) -> MetricRegistry:
    registry = MetricRegistry()
    for item in payload:
        registry.register(
            MetricMetadata(
                metric_name=item["metric_name"],
                business_line=item.get("business_line", "default"),
                priority=item.get("priority", "P1"),
                metric_type=item.get("metric_type", "gauge"),
                core_link=item.get("core_link", True),
                force_ai=item.get("force_ai", False),
                tags=item.get("tags", {}),
            )
        )
    return registry


pipeline = OnlinePipeline(registry=build_demo_registry(), history_store=_build_history_store())
_pipeline_lock = threading.RLock()
_requests_condition = threading.Condition()
_active_requests = 0


@contextmanager
def _track_request():
    global _active_requests
    with _requests_condition:
        _active_requests += 1
    try:
        yield
    finally:
        with _requests_condition:
            _active_requests -= 1
            if _active_requests <= 0:
                _requests_condition.notify_all()


def _wait_for_inflight_requests(timeout_sec: float) -> bool:
    with _requests_condition:
        if _active_requests <= 0:
            return True
        return _requests_condition.wait_for(lambda: _active_requests <= 0, timeout=timeout_sec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    drained = _wait_for_inflight_requests(DEFAULT_SETTINGS.shutdown_drain_timeout_sec)
    if not drained:
        logger.warning(
            "Timed out waiting for in-flight requests to drain within %.1fs",
            DEFAULT_SETTINGS.shutdown_drain_timeout_sec,
        )
    with _pipeline_lock:
        current_pipeline = pipeline
    try:
        current_pipeline.history_store.flush_writes()
        current_pipeline.history_store.close()
        logger.info("History store closed cleanly on shutdown.")
    except Exception:
        logger.exception("Failed to close history store cleanly on shutdown.")
        raise


app = FastAPI(title="NOCcore", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ingest", response_model=list[AlertEvent])
def ingest(points: list[MetricPoint]) -> list[AlertEvent]:
    with _track_request():
        with _pipeline_lock:
            current_pipeline = pipeline
        return current_pipeline.process_points(points)


@app.post("/v1/flush", response_model=list[AlertEvent])
def flush() -> list[AlertEvent]:
    with _track_request():
        with _pipeline_lock:
            current_pipeline = pipeline
        return current_pipeline.flush()


@app.post("/v1/reload-registry")
def reload_registry(payload: list[dict]) -> dict[str, int]:
    with _track_request():
        registry = _build_registry(payload)
        with _pipeline_lock:
            current_pipeline = pipeline
        current_pipeline.reload_registry(registry)
        return {"metrics": len(payload)}
