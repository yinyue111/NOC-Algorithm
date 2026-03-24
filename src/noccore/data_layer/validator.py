from __future__ import annotations

from typing import Any

from noccore.schemas.metric import MetricPoint


class MetricValidator:
    def validate(self, raw_point: MetricPoint | dict[str, Any]) -> MetricPoint:
        if isinstance(raw_point, MetricPoint):
            return raw_point
        return MetricPoint.model_validate(raw_point)
