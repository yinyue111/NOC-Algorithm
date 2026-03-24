from __future__ import annotations

from typing import Iterable

import numpy as np

from noccore.config.settings import PipelineSettings
from noccore.schemas.metric import MetricPoint
from noccore.utils.stats import EPS


class MetricRepairer:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def repair_continuity(self, previous: MetricPoint | None, current: MetricPoint) -> tuple[list[MetricPoint], MetricPoint]:
        if previous is None:
            return [], current

        gap = current.timestamp - previous.timestamp - 1
        if gap <= 0 or gap > self.settings.max_missing_fill:
            if gap > self.settings.max_missing_fill:
                tags = dict(current.tags)
                tags["missing_gap"] = gap
                current = current.model_copy(update={"trusted": False, "tags": tags})
            return [], current

        filled: list[MetricPoint] = []
        step = (current.value - previous.value) / float(gap + 1)
        for index in range(1, gap + 1):
            fill_value = (
                0.0
                if current.metric_type in {"count", "qps"}
                else previous.value + step * index
            )
            filled.append(
                previous.model_copy(
                    update={
                        "timestamp": previous.timestamp + index,
                        "value": fill_value,
                        "trusted": True,
                        "is_interpolated": True,
                    }
                )
            )
        return filled, current

    def clip_jump(self, point: MetricPoint, history_values: Iterable[float]) -> MetricPoint:
        values = np.asarray(list(history_values), dtype=float)
        if values.size < 20:
            return point

        upper_ref = max(float(np.quantile(np.abs(values), 0.99)), EPS)
        upper_limit = upper_ref * self.settings.jump_clip_multiplier
        clipped_value = float(np.clip(point.value, 0.0, upper_limit))
        if abs(clipped_value - point.value) <= EPS:
            return point

        tags = dict(point.tags)
        tags["repaired_jump"] = True
        tags["raw_value"] = point.value
        return point.model_copy(update={"value": clipped_value, "tags": tags})
