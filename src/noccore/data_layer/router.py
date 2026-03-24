from __future__ import annotations

from dataclasses import dataclass

from noccore.config.metric_registry import EligibilityDecision, MetricMetadata
from noccore.config.settings import PipelineSettings
from noccore.utils.time import format_granularity


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    granularity_sec: int
    label: str


class MetricRouter:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def route(self, metadata: MetricMetadata, eligibility: EligibilityDecision | None) -> RouteDecision:
        if metadata.force_ai or (eligibility is not None and eligibility.eligible):
            return RouteDecision(mode="ai", granularity_sec=1, label="1s")

        granularity = self.settings.priority_route_seconds.get(metadata.priority, 300)
        mode = "zero_rule" if granularity >= 600 else "minute_ai"
        return RouteDecision(mode=mode, granularity_sec=granularity, label=format_granularity(granularity))
