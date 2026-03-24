from __future__ import annotations

from dataclasses import dataclass

from noccore.alert_layer.suppressor import AlertSuppressor
from noccore.config.metric_registry import MetricMetadata
from noccore.config.settings import PipelineSettings
from noccore.schemas.alert import AlertEvent
from noccore.schemas.anomaly import AnomalyEvent
from noccore.schemas.prediction import PredictionResult
from noccore.utils.time import humanize_duration


@dataclass
class IncidentState:
    is_open: bool = False
    open_since: int = 0
    current_level: str = "P3"
    normal_streak: int = 0


@dataclass
class AlertPolicyState:
    incident: IncidentState
    suppressor_last_emit_timestamp: int | None = None


@dataclass
class AlertDecisionPlan:
    alerts: list[AlertEvent]
    incident: IncidentState
    suppressor_last_emit_timestamp: int | None


class AlertPolicy:
    def __init__(self, settings: PipelineSettings, suppressor: AlertSuppressor) -> None:
        self.settings = settings
        self.suppressor = suppressor
        self._incidents: dict[str, IncidentState] = {}

    def get_state(self, metric_name: str) -> AlertPolicyState:
        incident = self._incidents.get(metric_name, IncidentState())
        return AlertPolicyState(
            incident=IncidentState(
                is_open=incident.is_open,
                open_since=incident.open_since,
                current_level=incident.current_level,
                normal_streak=incident.normal_streak,
            ),
            suppressor_last_emit_timestamp=self.suppressor.get_last_emit_timestamp(metric_name, "anomaly"),
        )

    def decide(
        self,
        metadata: MetricMetadata,
        prediction: PredictionResult,
        anomaly: AnomalyEvent,
        state: AlertPolicyState,
    ) -> AlertDecisionPlan:
        incident = IncidentState(
            is_open=state.incident.is_open,
            open_since=state.incident.open_since,
            current_level=state.incident.current_level,
            normal_streak=state.incident.normal_streak,
        )
        suppressor_last_emit_timestamp = state.suppressor_last_emit_timestamp
        alerts: list[AlertEvent] = []
        level = self._determine_level(metadata, anomaly)

        if level is None:
            if incident.is_open:
                incident.normal_streak += 1
                if incident.normal_streak >= self.settings.recovery_normal_points:
                    alert_type = "recovery"
                    alerts.append(
                        AlertEvent(
                            metric_name=prediction.metric_name,
                            timestamp=prediction.timestamp,
                            granularity=prediction.granularity,
                            alert_level=incident.current_level,
                            alert_type=alert_type,
                            confidence_score=anomaly.confidence_score,
                            duration_sec=max(prediction.timestamp - incident.open_since, 0),
                            message=f"{prediction.metric_name} recovered after {humanize_duration(prediction.timestamp - incident.open_since)}",
                            status="recovered",
                            current_value=prediction.actual_value,
                            predict_value=prediction.predict_value,
                            z_score=anomaly.z_score,
                        )
                    )
                    incident = IncidentState()
                    suppressor_last_emit_timestamp = None
            return AlertDecisionPlan(
                alerts=alerts,
                incident=incident,
                suppressor_last_emit_timestamp=suppressor_last_emit_timestamp,
            )

        incident.normal_streak = 0
        status = "firing"
        if not incident.is_open:
            incident.is_open = True
            incident.open_since = prediction.timestamp
            incident.current_level = level
        else:
            if self._compare_level(level, incident.current_level) < 0:
                status = "escalated"
                incident.current_level = level
            elif (
                incident.current_level == "P1"
                and metadata.core_link
                and prediction.timestamp - incident.open_since >= self.settings.alert_upgrade_window_sec
            ):
                status = "escalated"
                incident.current_level = "P0"

        duration = max(prediction.timestamp - incident.open_since, 0)
        if status == "escalated" or self._should_emit(state.suppressor_last_emit_timestamp, prediction.timestamp):
            suppressor_last_emit_timestamp = prediction.timestamp
            alerts.append(
                AlertEvent(
                    metric_name=prediction.metric_name,
                    timestamp=prediction.timestamp,
                    granularity=prediction.granularity,
                    alert_level=incident.current_level,
                    alert_type="+".join(anomaly.abnormal_labels) if anomaly.abnormal_labels else "global",
                    confidence_score=anomaly.confidence_score,
                    duration_sec=duration,
                    message=self._format_message(metadata, prediction, anomaly, duration),
                    status=status,
                    current_value=prediction.actual_value,
                    predict_value=prediction.predict_value,
                    z_score=anomaly.z_score,
                )
            )
        return AlertDecisionPlan(
            alerts=alerts,
            incident=incident,
            suppressor_last_emit_timestamp=suppressor_last_emit_timestamp,
        )

    def _determine_level(self, metadata: MetricMetadata, anomaly: AnomalyEvent) -> str | None:
        has_anomaly_signal = (
            anomaly.global_abnormal
            or anomaly.local_score >= 0.5
            or "zero_rule" in anomaly.abnormal_labels
        )
        if not has_anomaly_signal:
            return None
        if "zero_rule" in anomaly.abnormal_labels:
            return "P2"
        if anomaly.confidence_score >= self.settings.confidence_p0 and metadata.core_link:
            return "P0"
        if anomaly.confidence_score >= self.settings.confidence_p1 and metadata.core_link:
            return "P1"
        if anomaly.confidence_score >= self.settings.confidence_p0:
            return "P1"
        if anomaly.confidence_score >= self.settings.confidence_p1:
            return "P2"
        return None

    def _format_message(
        self,
        metadata: MetricMetadata,
        prediction: PredictionResult,
        anomaly: AnomalyEvent,
        duration: int,
    ) -> str:
        label = prediction.metric_name
        if metadata.business_line:
            label = f"{metadata.business_line} {label}"
        return (
            f"{label} abnormal: current={prediction.actual_value:.4f}, "
            f"predict={prediction.predict_value:.4f}, z={anomaly.z_score:.3f}, "
            f"confidence={anomaly.confidence_score:.3f}, duration={humanize_duration(duration)}"
        )

    def _compare_level(self, left: str, right: str) -> int:
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return order.get(left, 9) - order.get(right, 9)

    def _should_emit(self, last_emit_timestamp: int | None, timestamp: int) -> bool:
        if last_emit_timestamp is None:
            return True
        return timestamp - last_emit_timestamp >= self.settings.alert_merge_window_sec

    def apply(self, metric_name: str, plan: AlertDecisionPlan) -> None:
        self._incidents[metric_name] = IncidentState(
            is_open=plan.incident.is_open,
            open_since=plan.incident.open_since,
            current_level=plan.incident.current_level,
            normal_streak=plan.incident.normal_streak,
        )
        self.suppressor.set_last_emit_timestamp(metric_name, "anomaly", plan.suppressor_last_emit_timestamp)

    def drop_metric(self, metric_name: str) -> None:
        self._incidents.pop(metric_name, None)
        self.suppressor.drop_metric(metric_name)
