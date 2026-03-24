from __future__ import annotations

import logging
from typing import Iterable

from noccore.config.metric_registry import EligibilityDecision, MetricMetadata
from noccore.config.settings import PipelineSettings
from noccore.schemas.metric import MetricPoint
from noccore.utils.stats import coefficient_of_variation, periodicity_score


logger = logging.getLogger(__name__)


class AIEligibilityAssessor:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def assess(self, points: Iterable[MetricPoint], metadata: MetricMetadata) -> EligibilityDecision:
        points = list(points)
        if metadata.force_ai:
            decision = EligibilityDecision(
                eligible=True,
                score=1.0,
                forced=True,
                reasons=("manual_override",),
                breakdown={"manual_override": 1.0},
            )
            self._log_decision(metadata.metric_name, decision)
            return decision

        if len(points) < 2:
            decision = EligibilityDecision(eligible=False, score=0.0, reasons=("insufficient_history",))
            self._log_decision(metadata.metric_name, decision)
            return decision

        history_duration = points[-1].timestamp - points[0].timestamp
        required_duration = self.settings.history_weeks_required * 7 * 24 * 3600
        if history_duration < required_duration:
            decision = EligibilityDecision(
                eligible=False,
                score=0.0,
                reasons=("history_lt_4_weeks",),
                breakdown={"history_duration_sec": float(history_duration)},
            )
            self._log_decision(metadata.metric_name, decision)
            return decision

        values = [point.value for point in points]
        timestamps = [point.timestamp for point in points]
        diffs = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
        median_gap = int(sorted(diffs)[len(diffs) // 2]) if diffs else 1

        if median_gap <= 1:
            candidate_lags = [86400, 604800]
        else:
            candidate_lags = [1440, 10080]

        periodicity = periodicity_score(values, candidate_lags)
        non_zero_ratio = sum(1 for value in values if value > 0) / max(len(values), 1)
        priority_score = {"P0": 1.0, "P1": 0.8, "P2": 0.5, "P3": 0.3}.get(metadata.priority, 0.5)
        cv = coefficient_of_variation(values)
        regularity = max(0.0, 1.0 - cv / 0.8)

        score = (
            0.30 * periodicity
            + 0.25 * min(non_zero_ratio / 0.7, 1.0)
            + 0.20 * priority_score
            + 0.25 * regularity
        )
        decision = EligibilityDecision(
            eligible=score >= 0.65,
            score=float(score),
            reasons=(),
            breakdown={
                "periodicity": float(periodicity),
                "non_zero_ratio": float(non_zero_ratio),
                "priority_score": float(priority_score),
                "regularity": float(regularity),
            },
        )
        self._log_decision(metadata.metric_name, decision)
        return decision

    def _log_decision(self, metric_name: str, decision: EligibilityDecision) -> None:
        logger.info(
            "Eligibility %s: eligible=%s score=%.2f reasons=%s",
            metric_name,
            decision.eligible,
            decision.score,
            decision.reasons,
        )
