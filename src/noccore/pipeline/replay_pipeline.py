from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

from noccore.config.metric_registry import MetricMetadata, MetricRegistry
from noccore.config.settings import DEFAULT_SETTINGS
from noccore.history_layer.store import InMemoryHistoryStore, SQLiteHistoryStore
from noccore.pipeline.online_pipeline import OnlinePipeline
from noccore.schemas.alert import AlertEvent
from noccore.schemas.metric import MetricPoint


def build_demo_registry() -> MetricRegistry:
    registry = MetricRegistry()
    registry.register(
        MetricMetadata(
            metric_name="payment.success_rate",
            business_line="AA",
            priority="P0",
            metric_type="rate",
            core_link=True,
            force_ai=True,
            tags={"channel": "alipay"},
        )
    )
    registry.register(
        MetricMetadata(
            metric_name="payment.long_tail_count",
            business_line="AA",
            priority="P2",
            metric_type="count",
            core_link=False,
            force_ai=False,
            tags={"channel": "legacy"},
        )
    )
    return registry


def generate_demo_points(duration_sec: int = 1800, start_ts: int = 1711180800) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for offset in range(duration_sec):
        base_rate = 0.962 + 0.008 * math.sin(offset / 45.0)
        noise = 0.002 * math.sin(offset / 7.0)
        rate_value = base_rate + noise
        if 720 <= offset < 780:
            rate_value -= 0.14
        points.append(
            MetricPoint(
                metric_name="payment.success_rate",
                timestamp=start_ts + offset,
                value=max(rate_value, 0.0),
                business_line="AA",
                priority="P0",
                metric_type="rate",
                tags={"force_ai": True, "channel": "alipay"},
            )
        )

        count_value = 40.0 + 4.0 * math.sin(offset / 30.0)
        if offset >= 1200:
            count_value = 0.0
        points.append(
            MetricPoint(
                metric_name="payment.long_tail_count",
                timestamp=start_ts + offset,
                value=max(count_value, 0.0),
                business_line="AA",
                priority="P2",
                metric_type="count",
                tags={"channel": "legacy"},
            )
        )
    return points


def load_jsonl(path: str | Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        points.append(MetricPoint.model_validate(json.loads(line)))
    return points


def run_replay(points: Iterable[MetricPoint], pipeline: OnlinePipeline | None = None) -> list[AlertEvent]:
    pipeline = pipeline or OnlinePipeline(registry=build_demo_registry())
    return pipeline.process_points(list(points), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay payment monitoring points through NOCcore.")
    parser.add_argument("--input", help="Path to a JSONL file. If omitted, a demo stream is generated.")
    parser.add_argument("--registry", help="Optional registry JSON file.")
    parser.add_argument("--history-db", help="Optional SQLite history database path.")
    parser.add_argument("--json", action="store_true", help="Print alert payloads as JSON.")
    args = parser.parse_args()

    registry = MetricRegistry.load_json(args.registry) if args.registry else build_demo_registry()
    history_store = (
        SQLiteHistoryStore(args.history_db, batch_size=DEFAULT_SETTINGS.history_commit_batch_size)
        if args.history_db
        else InMemoryHistoryStore(DEFAULT_SETTINGS)
    )
    pipeline = OnlinePipeline(registry=registry, history_store=history_store)
    points = load_jsonl(args.input) if args.input else generate_demo_points()
    alerts = run_replay(points, pipeline=pipeline)

    for alert in alerts:
        if args.json:
            print(json.dumps(alert.model_dump(), ensure_ascii=False))
        else:
            print(
                f"[{alert.alert_level}] {alert.metric_name} {alert.status} "
                f"confidence={alert.confidence_score:.3f} "
                f"current={alert.current_value:.4f} predict={alert.predict_value:.4f}"
            )

    print(f"alerts={len(alerts)}")


if __name__ == "__main__":
    main()
