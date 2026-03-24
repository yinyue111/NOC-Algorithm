from __future__ import annotations

from dataclasses import dataclass
from time import time

from noccore.model_layer.model_store import InMemoryModelStore


@dataclass
class TrainingResult:
    metric_name: str
    version: str
    parameters: dict


class TrainingManager:
    def __init__(self, model_store: InMemoryModelStore) -> None:
        self.model_store = model_store

    def publish(self, metric_name: str, parameters: dict, prefix: str) -> TrainingResult:
        version = f"{prefix}_{int(time())}"
        self.model_store.put(metric_name, version, parameters)
        return TrainingResult(metric_name=metric_name, version=version, parameters=parameters)
