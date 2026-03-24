from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnomalyEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_name: str
    timestamp: int
    granularity: str
    global_abnormal: bool
    local_score: float
    is_periodic: bool
    z_score: float
    lower_bound: float
    upper_bound: float
    severity: float
    direction: str = "normal"
    abnormal_labels: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_components: dict[str, float] = Field(default_factory=dict)
