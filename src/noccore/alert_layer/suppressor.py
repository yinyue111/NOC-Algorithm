from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SuppressionState:
    last_emit_timestamp: int | None = None


class AlertSuppressor:
    def __init__(self, merge_window_sec: int) -> None:
        self.merge_window_sec = merge_window_sec
        self._state: dict[tuple[str, str], SuppressionState] = {}

    def get_last_emit_timestamp(self, metric_name: str, alert_type: str) -> int | None:
        state = self._state.get((metric_name, alert_type))
        return None if state is None else state.last_emit_timestamp

    def set_last_emit_timestamp(self, metric_name: str, alert_type: str, timestamp: int | None) -> None:
        key = (metric_name, alert_type)
        if timestamp is None:
            self._state.pop(key, None)
            return
        state = self._state.setdefault(key, SuppressionState())
        state.last_emit_timestamp = timestamp

    def should_emit(self, metric_name: str, alert_type: str, timestamp: int) -> bool:
        key = (metric_name, alert_type)
        state = self._state.setdefault(key, SuppressionState())
        if state.last_emit_timestamp is None:
            state.last_emit_timestamp = timestamp
            return True
        if timestamp - state.last_emit_timestamp >= self.merge_window_sec:
            state.last_emit_timestamp = timestamp
            return True
        return False

    def reset(self, metric_name: str, alert_type: str) -> None:
        key = (metric_name, alert_type)
        self._state.pop(key, None)

    def drop_metric(self, metric_name: str) -> None:
        keys_to_remove = [key for key in self._state if key[0] == metric_name]
        for key in keys_to_remove:
            self._state.pop(key, None)
