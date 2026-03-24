from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AlertEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_name: str
    timestamp: int
    granularity: str
    alert_level: str
    alert_type: str
    confidence_score: float
    duration_sec: int
    message: str
    status: str
    current_value: float
    predict_value: float
    z_score: float
