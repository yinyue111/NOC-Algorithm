from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_name: str
    timestamp: int
    granularity: str
    actual_value: float
    predict_value: float
    lower_bound: float
    upper_bound: float
    model_name: str
    model_version: str
    features: dict[str, Any] = Field(default_factory=dict)
