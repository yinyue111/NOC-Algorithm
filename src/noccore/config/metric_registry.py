from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EligibilityDecision:
    eligible: bool
    score: float
    forced: bool = False
    reasons: tuple[str, ...] = ()
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class MetricMetadata:
    metric_name: str
    business_line: str
    priority: str = "P1"
    metric_type: str = "gauge"
    core_link: bool = True
    force_ai: bool = False
    tags: dict[str, Any] = field(default_factory=dict)


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, MetricMetadata] = {}
        self._eligibility: dict[str, EligibilityDecision] = {}
        self._lock = threading.RLock()

    def register(self, metadata: MetricMetadata) -> None:
        with self._lock:
            self._metrics[metadata.metric_name] = metadata

    def get(self, metric_name: str) -> MetricMetadata | None:
        with self._lock:
            return self._metrics.get(metric_name)

    def get_or_create(self, point: Any) -> MetricMetadata:
        with self._lock:
            metadata = self._metrics.get(point.metric_name)
            if metadata is not None:
                return metadata

            tags = dict(point.tags)
            force_ai = bool(tags.get("force_ai", False))
            core_link = bool(tags.get("core_link", True))
            metadata = MetricMetadata(
                metric_name=point.metric_name,
                business_line=point.business_line,
                priority=point.priority,
                metric_type=point.metric_type,
                core_link=core_link,
                force_ai=force_ai,
                tags=tags,
            )
            self._metrics[metadata.metric_name] = metadata
            return metadata

    def set_eligibility(self, metric_name: str, decision: EligibilityDecision) -> None:
        with self._lock:
            self._eligibility[metric_name] = decision

    def get_eligibility(self, metric_name: str) -> EligibilityDecision | None:
        with self._lock:
            return self._eligibility.get(metric_name)

    def clear_eligibility(self, metric_name: str) -> None:
        with self._lock:
            self._eligibility.pop(metric_name, None)

    def metric_names(self) -> set[str]:
        with self._lock:
            return set(self._metrics)

    def reload_from(self, other: "MetricRegistry") -> None:
        with other._lock:
            new_metrics = dict(other._metrics)
        with self._lock:
            self._metrics = new_metrics
            self._eligibility = {
                metric_name: decision
                for metric_name, decision in self._eligibility.items()
                if metric_name in new_metrics
            }

    @classmethod
    def load_json(cls, path: str | Path) -> "MetricRegistry":
        registry = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
