from __future__ import annotations

from dataclasses import dataclass, field

from noccore.utils.time import to_local_datetime


@dataclass
class CalendarProvider:
    timezone: str
    holiday_dates: set[str] = field(default_factory=set)
    hotspot_dates: set[str] = field(default_factory=set)
    rca_windows: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def get_context(self, metric_name: str, timestamp: int) -> dict[str, bool]:
        date_key = to_local_datetime(timestamp, self.timezone).date().isoformat()
        return {
            "isHoliday": date_key in self.holiday_dates,
            "isHotSpot": date_key in self.hotspot_dates,
            "isRCA": any(start_ts <= timestamp <= end_ts for start_ts, end_ts in self.rca_windows.get(metric_name, [])),
        }

    def is_holiday(self, timestamp: int) -> bool:
        return self.get_context("", timestamp)["isHoliday"]

    def is_hotspot(self, metric_name: str, timestamp: int) -> bool:
        return self.get_context(metric_name, timestamp)["isHotSpot"]

    def is_rca_window(self, metric_name: str, timestamp: int) -> bool:
        return self.get_context(metric_name, timestamp)["isRCA"]
