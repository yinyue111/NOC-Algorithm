from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_name: str
    timestamp: int
    value: float
    business_line: str
    tags: dict[str, Any] = Field(default_factory=dict)
    priority: str = "P1"
    metric_type: str = "gauge"
    granularity: str = "1s"
    trusted: bool = True
    is_interpolated: bool = False

    @field_validator("metric_name", "business_line", "priority", "metric_type", "granularity")
    @classmethod
    def _strip_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be empty")
        return value

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("timestamp must be a positive unix second")
        return value

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return float(value)
