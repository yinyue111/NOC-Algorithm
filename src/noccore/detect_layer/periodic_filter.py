from __future__ import annotations

from noccore.config.settings import PipelineSettings
from noccore.utils.stats import mann_kendall_test, safe_mean, safe_std


class PeriodicityFilter:
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def is_periodic(
        self,
        values: list[float],
        anomaly_flags: list[bool],
        granularity: str,
    ) -> bool:
        if len(values) < 10:
            return False

        current_index = len(values) - 1
        candidate_periods = self._candidate_periods(granularity, len(values))
        for period in candidate_periods:
            periodic_hits: list[float] = []
            normalized_diffs: list[float] = []
            tolerance = max(1, min(10, period // 10))

            for repeat_index in range(1, self.settings.periodic_week_lookback + 1):
                center = current_index - repeat_index * period
                if center <= 0:
                    continue
                start = max(0, center - tolerance)
                end = min(current_index - 1, center + tolerance)
                window_flags = anomaly_flags[start : end + 1]
                if not window_flags:
                    continue
                periodic_hits.append(1.0 if any(window_flags) else 0.0)
                window_values = values[start : end + 1]
                baseline = safe_mean(window_values, values[current_index])
                scale = max(safe_std(window_values, 1.0), 1.0)
                normalized_diffs.append((values[current_index] - baseline) / scale)

            if len(periodic_hits) < self.settings.periodic_min_matches:
                continue

            if safe_mean(periodic_hits, 0.0) >= self.settings.periodic_ratio_threshold:
                return True

            significant, _ = mann_kendall_test(normalized_diffs)
            if significant:
                return False

        return False

    def _candidate_periods(self, granularity: str, history_len: int) -> list[int]:
        if granularity == "1s":
            periods = [60, 300, 3600, 86400]
        elif granularity == "1m":
            periods = [60, 1440]
        elif granularity == "5m":
            periods = [12, 288]
        elif granularity == "10m":
            periods = [6, 144]
        elif granularity == "30m":
            periods = [48]
        else:
            periods = [60]
        max_period = history_len // max(self.settings.periodic_week_lookback + 1, 1)
        return [period for period in periods if period <= max_period]
