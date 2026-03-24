from __future__ import annotations

from typing import Protocol

from noccore.schemas.alert import AlertEvent


class Notifier(Protocol):
    def send(self, alert: AlertEvent) -> None:
        ...


class ConsoleNotifier:
    def format(self, alert: AlertEvent) -> str:
        return (
            f"[{alert.alert_level}] {alert.metric_name} {alert.status} "
            f"value={alert.current_value:.4f} predict={alert.predict_value:.4f} "
            f"confidence={alert.confidence_score:.3f} z={alert.z_score:.3f}"
        )

    def send(self, alert: AlertEvent) -> None:
        print(self.format(alert))
