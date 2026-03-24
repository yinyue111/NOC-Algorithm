from __future__ import annotations

from noccore.utils.stats import coefficient_of_variation, rolling_mae, safe_mean


class ConfidenceScorer:
    def score(
        self,
        actual_history: list[float],
        predicted_history: list[float],
        severity: float,
        is_periodic: bool,
    ) -> tuple[float, dict[str, float]]:
        recent_actual = actual_history[-300:]
        recent_pred = predicted_history[-300:]
        mean_abs_actual = max(safe_mean([abs(value) for value in recent_actual], 1.0), 1.0)
        mae = rolling_mae(recent_actual, recent_pred)
        metric_reliability = max(0.0, min(1.0, 1.0 - mae / mean_abs_actual))
        anomaly_severity = max(0.0, min(1.0, severity))
        cv = coefficient_of_variation(recent_actual)
        volatility_level = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))
        periodic_flag = 0.0 if is_periodic else 1.0

        confidence = (
            0.25 * metric_reliability
            + 0.35 * anomaly_severity
            + 0.20 * volatility_level
            + 0.20 * periodic_flag
        )
        components = {
            "metric_reliability": float(metric_reliability),
            "anomaly_severity": float(anomaly_severity),
            "volatility_level": float(volatility_level),
            "is_periodic": float(periodic_flag),
        }
        return float(confidence), components
