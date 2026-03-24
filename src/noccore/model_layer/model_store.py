from __future__ import annotations


class InMemoryModelStore:
    def __init__(self) -> None:
        self._active_version: dict[str, str] = {}
        self._parameters: dict[tuple[str, str], dict] = {}

    def put(self, metric_name: str, version: str, params: dict) -> None:
        self._parameters[(metric_name, version)] = params
        self._active_version[metric_name] = version

    def get_active(self, metric_name: str) -> tuple[str | None, dict]:
        version = self._active_version.get(metric_name)
        if version is None:
            return None, {}
        return version, self._parameters.get((metric_name, version), {})

    def rollback(self, metric_name: str, version: str) -> bool:
        if (metric_name, version) not in self._parameters:
            return False
        self._active_version[metric_name] = version
        return True
