from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from noccore.schemas.metric import MetricPoint
from noccore.utils.time import floor_timestamp, format_granularity


@dataclass
class BucketState:
    bucket_start: int
    points: list[MetricPoint] = field(default_factory=list)


class Downsampler:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, int], BucketState] = {}
        self._lock = threading.RLock()

    def push(self, point: MetricPoint, granularity_sec: int) -> MetricPoint | None:
        with self._lock:
            key = (point.metric_name, granularity_sec)
            bucket_start = floor_timestamp(point.timestamp, granularity_sec)
            state = self._buckets.get(key)

            if state is None:
                self._buckets[key] = BucketState(bucket_start=bucket_start, points=[point])
                return None

            if state.bucket_start == bucket_start:
                state.points.append(point)
                return None

            aggregated = self._aggregate(state.points, granularity_sec)
            self._buckets[key] = BucketState(bucket_start=bucket_start, points=[point])
            return aggregated

    def flush(self) -> list[MetricPoint]:
        with self._lock:
            items = [
                (granularity, state.points[:])
                for (_, granularity), state in self._buckets.items()
                if state.points
            ]
            self._buckets.clear()
        flushed = [self._aggregate(points, granularity) for granularity, points in items]
        return flushed

    def drop_metric(self, metric_name: str) -> None:
        with self._lock:
            keys_to_remove = [key for key in self._buckets if key[0] == metric_name]
            for key in keys_to_remove:
                self._buckets.pop(key, None)

    def _aggregate(self, points: list[MetricPoint], granularity_sec: int) -> MetricPoint:
        first = points[0]
        values = np.asarray([point.value for point in points], dtype=float)

        if first.metric_type in {"count", "qps"}:
            aggregated_value = float(np.sum(values))
        elif first.metric_type in {"rate", "ratio"}:
            aggregated_value = float(np.mean(values))
        elif first.metric_type == "latency":
            aggregated_value = float(np.quantile(values, 0.95))
        else:
            aggregated_value = float(values[-1])

        return first.model_copy(
            update={
                "timestamp": points[-1].timestamp,
                "value": aggregated_value,
                "granularity": format_granularity(granularity_sec),
                "tags": {
                    **first.tags,
                    "bucket_size": len(points),
                    "downsampled": True,
                },
            }
        )
