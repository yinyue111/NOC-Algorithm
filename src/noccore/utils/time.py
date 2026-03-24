from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def get_tzinfo(timezone_name: str) -> ZoneInfo:
    return ZoneInfo(timezone_name)


def to_local_datetime(timestamp: int, timezone_name: str) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=get_tzinfo(timezone_name))


def floor_timestamp(timestamp: int, granularity_sec: int) -> int:
    return timestamp - (timestamp % granularity_sec)


def format_granularity(granularity_sec: int) -> str:
    if granularity_sec < 60:
        return f"{granularity_sec}s"
    minutes = granularity_sec // 60
    return f"{minutes}m"


def humanize_duration(duration_sec: int) -> str:
    minutes, seconds = divmod(max(duration_sec, 0), 60)
    if minutes == 0:
        return f"{seconds}s"
    return f"{minutes}m{seconds}s"
