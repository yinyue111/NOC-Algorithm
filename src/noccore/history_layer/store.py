from __future__ import annotations

import itertools
import json
import logging
import sqlite3
import threading
from contextlib import suppress
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from pathlib import Path

from noccore.config.settings import PipelineSettings
from noccore.schemas.anomaly import AnomalyEvent
from noccore.schemas.metric import MetricPoint
from noccore.schemas.prediction import PredictionResult
from noccore.utils.time import to_local_datetime


logger = logging.getLogger(__name__)

_PRUNE_STATEMENTS = (
    "DELETE FROM raw_points WHERE timestamp < ?",
    "DELETE FROM series_points WHERE timestamp < ?",
    "DELETE FROM predictions WHERE timestamp < ?",
    "DELETE FROM anomalies WHERE timestamp < ?",
)


class HistoryStore(ABC):
    @abstractmethod
    def append_raw_point(self, point: MetricPoint, timezone: str, is_holiday: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_series_point(self, point: MetricPoint) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_prediction(self, prediction: PredictionResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_anomaly(self, anomaly: AnomalyEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_recent_raw_points(
        self,
        metric_name: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[MetricPoint]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_series_points(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[MetricPoint]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_predictions(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[PredictionResult]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_anomalies(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[AnomalyEvent]:
        raise NotImplementedError

    @abstractmethod
    def get_aligned_history_values(
        self,
        metric_name: str,
        timestamp: int,
        timezone: str,
        is_holiday: bool,
        aligned_limit: int = 28,
        fallback_limit: int = 60,
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def drop_metric(self, metric_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush_writes(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prune_old_data(self, before_timestamp: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def should_drop_metric_on_ttl(self) -> bool:
        return True


class InMemoryHistoryStore(HistoryStore):
    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._raw_points: dict[str, deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=self.settings.max_history_points)
        )
        self._series_points: dict[tuple[str, str], deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=self.settings.max_history_points)
        )
        self._predictions: dict[tuple[str, str], deque[PredictionResult]] = defaultdict(
            lambda: deque(maxlen=self.settings.max_history_points)
        )
        self._anomalies: dict[tuple[str, str], deque[AnomalyEvent]] = defaultdict(
            lambda: deque(maxlen=self.settings.max_history_points)
        )
        self._aligned_index: dict[str, dict[tuple[int, int, int, int, bool], deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=28))
        )
        self._fallback_index: dict[str, dict[int, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=60))
        )

    def append_raw_point(self, point: MetricPoint, timezone: str, is_holiday: bool) -> None:
        with self._lock:
            self._raw_points[point.metric_name].append(point)
            point_dt = to_local_datetime(point.timestamp, timezone)
            aligned_key = (
                point_dt.weekday(),
                point_dt.hour,
                point_dt.minute,
                point_dt.second,
                is_holiday,
            )
            self._aligned_index[point.metric_name][aligned_key].append(point.value)
            self._fallback_index[point.metric_name][point_dt.second].append(point.value)

    def append_series_point(self, point: MetricPoint) -> None:
        with self._lock:
            self._series_points[(point.metric_name, point.granularity)].append(point)

    def append_prediction(self, prediction: PredictionResult) -> None:
        with self._lock:
            self._predictions[(prediction.metric_name, prediction.granularity)].append(prediction)

    def append_anomaly(self, anomaly: AnomalyEvent) -> None:
        with self._lock:
            self._anomalies[(anomaly.metric_name, anomaly.granularity)].append(anomaly)

    def get_recent_raw_points(
        self,
        metric_name: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[MetricPoint]:
        with self._lock:
            raw = self._raw_points.get(metric_name)
            if raw is None:
                return []
            if since_timestamp is None:
                return self._tail_items(raw, limit)
            points = [point for point in raw if point.timestamp >= since_timestamp]
            if limit is not None and len(points) > limit:
                return points[-limit:]
            return points

    def get_recent_series_points(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[MetricPoint]:
        with self._lock:
            points_deque = self._series_points.get((metric_name, granularity))
            if points_deque is None:
                return []
            if since_timestamp is None:
                return self._tail_items(points_deque, limit)
            points = [point for point in points_deque if point.timestamp >= since_timestamp]
            if limit is not None and len(points) > limit:
                return points[-limit:]
            return points

    def get_recent_predictions(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[PredictionResult]:
        with self._lock:
            predictions_deque = self._predictions.get((metric_name, granularity))
            if predictions_deque is None:
                return []
            if since_timestamp is None:
                return self._tail_items(predictions_deque, limit)
            predictions = [item for item in predictions_deque if item.timestamp >= since_timestamp]
            if limit is not None and len(predictions) > limit:
                return predictions[-limit:]
            return predictions

    def get_recent_anomalies(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[AnomalyEvent]:
        with self._lock:
            anomalies_deque = self._anomalies.get((metric_name, granularity))
            if anomalies_deque is None:
                return []
            if since_timestamp is None:
                return self._tail_items(anomalies_deque, limit)
            anomalies = [item for item in anomalies_deque if item.timestamp >= since_timestamp]
            if limit is not None and len(anomalies) > limit:
                return anomalies[-limit:]
            return anomalies

    def get_aligned_history_values(
        self,
        metric_name: str,
        timestamp: int,
        timezone: str,
        is_holiday: bool,
        aligned_limit: int = 28,
        fallback_limit: int = 60,
    ) -> list[float]:
        with self._lock:
            point_dt = to_local_datetime(timestamp, timezone)
            aligned_key = (
                point_dt.weekday(),
                point_dt.hour,
                point_dt.minute,
                point_dt.second,
                is_holiday,
            )
            aligned_values = list(self._aligned_index.get(metric_name, {}).get(aligned_key, ()))
            if aligned_values:
                return aligned_values[-aligned_limit:]
            fallback_values = list(self._fallback_index.get(metric_name, {}).get(point_dt.second, ()))
            return fallback_values[-fallback_limit:]

    def drop_metric(self, metric_name: str) -> None:
        with self._lock:
            self._raw_points.pop(metric_name, None)
            self._aligned_index.pop(metric_name, None)
            self._fallback_index.pop(metric_name, None)
            keys_to_remove = [key for key in self._series_points if key[0] == metric_name]
            for key in keys_to_remove:
                self._series_points.pop(key, None)
            keys_to_remove = [key for key in self._predictions if key[0] == metric_name]
            for key in keys_to_remove:
                self._predictions.pop(key, None)
            keys_to_remove = [key for key in self._anomalies if key[0] == metric_name]
            for key in keys_to_remove:
                self._anomalies.pop(key, None)

    def flush_writes(self) -> None:
        return None

    def prune_old_data(self, before_timestamp: int) -> int:
        return 0

    def close(self) -> None:
        return None

    def _tail_items(self, items: deque, limit: int | None) -> list:
        if limit is None:
            return list(items)
        return list(itertools.islice(reversed(items), limit))[::-1]


class SQLiteHistoryStore(HistoryStore):
    def __init__(self, db_path: str | Path, batch_size: int = 100) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = max(int(batch_size), 1)
        self._pending_writes = 0
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_points (
                    metric_name TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    value REAL NOT NULL,
                    business_line TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    trusted INTEGER NOT NULL,
                    is_interpolated INTEGER NOT NULL,
                    weekday INTEGER NOT NULL,
                    hour INTEGER NOT NULL,
                    minute INTEGER NOT NULL,
                    second INTEGER NOT NULL,
                    is_holiday INTEGER NOT NULL,
                    PRIMARY KEY (metric_name, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_raw_metric_ts
                    ON raw_points(metric_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_raw_aligned
                    ON raw_points(metric_name, weekday, hour, minute, second, is_holiday, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_raw_timestamp
                    ON raw_points(timestamp);

                CREATE TABLE IF NOT EXISTS series_points (
                    metric_name TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    value REAL NOT NULL,
                    business_line TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    trusted INTEGER NOT NULL,
                    is_interpolated INTEGER NOT NULL,
                    PRIMARY KEY (metric_name, granularity, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_series_metric_ts
                    ON series_points(metric_name, granularity, timestamp);
                CREATE INDEX IF NOT EXISTS idx_series_timestamp
                    ON series_points(timestamp);

                CREATE TABLE IF NOT EXISTS predictions (
                    metric_name TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (metric_name, granularity, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_metric_ts
                    ON predictions(metric_name, granularity, timestamp);
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                    ON predictions(timestamp);

                CREATE TABLE IF NOT EXISTS anomalies (
                    metric_name TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (metric_name, granularity, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_anomalies_metric_ts
                    ON anomalies(metric_name, granularity, timestamp);
                CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp
                    ON anomalies(timestamp);
                """
            )
            self._commit()

    def append_raw_point(self, point: MetricPoint, timezone: str, is_holiday: bool) -> None:
        point_dt = to_local_datetime(point.timestamp, timezone)
        self._execute_write(
            """
            INSERT OR REPLACE INTO raw_points (
                metric_name, timestamp, value, business_line, tags_json, priority,
                metric_type, granularity, trusted, is_interpolated,
                weekday, hour, minute, second, is_holiday
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                point.metric_name,
                point.timestamp,
                point.value,
                point.business_line,
                json.dumps(point.tags, ensure_ascii=False, sort_keys=True),
                point.priority,
                point.metric_type,
                point.granularity,
                int(point.trusted),
                int(point.is_interpolated),
                point_dt.weekday(),
                point_dt.hour,
                point_dt.minute,
                point_dt.second,
                int(is_holiday),
            ),
            context=f"raw_point:{point.metric_name}",
        )

    def append_series_point(self, point: MetricPoint) -> None:
        self._execute_write(
            """
            INSERT OR REPLACE INTO series_points (
                metric_name, granularity, timestamp, value, business_line,
                tags_json, priority, metric_type, trusted, is_interpolated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                point.metric_name,
                point.granularity,
                point.timestamp,
                point.value,
                point.business_line,
                json.dumps(point.tags, ensure_ascii=False, sort_keys=True),
                point.priority,
                point.metric_type,
                int(point.trusted),
                int(point.is_interpolated),
            ),
            context=f"series_point:{point.metric_name}/{point.granularity}",
        )

    def append_prediction(self, prediction: PredictionResult) -> None:
        self._execute_write(
            """
            INSERT OR REPLACE INTO predictions (
                metric_name, granularity, timestamp, payload_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                prediction.metric_name,
                prediction.granularity,
                prediction.timestamp,
                json.dumps(prediction.model_dump(), ensure_ascii=False, sort_keys=True),
            ),
            context=f"prediction:{prediction.metric_name}/{prediction.granularity}",
        )

    def append_anomaly(self, anomaly: AnomalyEvent) -> None:
        self._execute_write(
            """
            INSERT OR REPLACE INTO anomalies (
                metric_name, granularity, timestamp, payload_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                anomaly.metric_name,
                anomaly.granularity,
                anomaly.timestamp,
                json.dumps(anomaly.model_dump(), ensure_ascii=False, sort_keys=True),
            ),
            context=f"anomaly:{anomaly.metric_name}/{anomaly.granularity}",
        )

    def get_recent_raw_points(
        self,
        metric_name: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[MetricPoint]:
        sql = """
            SELECT metric_name, timestamp, value, business_line, tags_json, priority,
                   metric_type, granularity, trusted, is_interpolated
            FROM raw_points
            WHERE metric_name = ?
        """
        params: list[object] = [metric_name]
        if since_timestamp is not None:
            sql += " AND timestamp >= ?"
            params.append(since_timestamp)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._row_to_metric_point(row) for row in reversed(rows)]

    def get_recent_series_points(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[MetricPoint]:
        sql = """
            SELECT metric_name, timestamp, value, business_line, tags_json, priority,
                   metric_type, granularity, trusted, is_interpolated
            FROM series_points
            WHERE metric_name = ? AND granularity = ?
        """
        params: list[object] = [metric_name, granularity]
        if since_timestamp is not None:
            sql += " AND timestamp >= ?"
            params.append(since_timestamp)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._row_to_metric_point(row) for row in reversed(rows)]

    def get_recent_predictions(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[PredictionResult]:
        sql = """
            SELECT payload_json
            FROM predictions
            WHERE metric_name = ? AND granularity = ?
        """
        params: list[object] = [metric_name, granularity]
        if since_timestamp is not None:
            sql += " AND timestamp >= ?"
            params.append(since_timestamp)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [PredictionResult.model_validate_json(row["payload_json"]) for row in reversed(rows)]

    def get_recent_anomalies(
        self,
        metric_name: str,
        granularity: str,
        limit: int | None = None,
        since_timestamp: int | None = None,
    ) -> list[AnomalyEvent]:
        sql = """
            SELECT payload_json
            FROM anomalies
            WHERE metric_name = ? AND granularity = ?
        """
        params: list[object] = [metric_name, granularity]
        if since_timestamp is not None:
            sql += " AND timestamp >= ?"
            params.append(since_timestamp)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [AnomalyEvent.model_validate_json(row["payload_json"]) for row in reversed(rows)]

    def get_aligned_history_values(
        self,
        metric_name: str,
        timestamp: int,
        timezone: str,
        is_holiday: bool,
        aligned_limit: int = 28,
        fallback_limit: int = 60,
    ) -> list[float]:
        point_dt = to_local_datetime(timestamp, timezone)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT value
                FROM raw_points
                WHERE metric_name = ?
                  AND timestamp < ?
                  AND weekday = ?
                  AND hour = ?
                  AND minute = ?
                  AND second = ?
                  AND is_holiday = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (
                    metric_name,
                    timestamp,
                    point_dt.weekday(),
                    point_dt.hour,
                    point_dt.minute,
                    point_dt.second,
                    int(is_holiday),
                    aligned_limit,
                ),
            ).fetchall()
        if rows:
            return [float(row["value"]) for row in reversed(rows)]

        with self._lock:
            fallback_rows = self._connection.execute(
                """
                SELECT value
                FROM raw_points
                WHERE metric_name = ?
                  AND timestamp < ?
                  AND second = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (
                    metric_name,
                    timestamp,
                    point_dt.second,
                    fallback_limit,
                ),
            ).fetchall()
        return [float(row["value"]) for row in reversed(fallback_rows)]

    def drop_metric(self, metric_name: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM raw_points WHERE metric_name = ?", (metric_name,))
            self._connection.execute("DELETE FROM series_points WHERE metric_name = ?", (metric_name,))
            self._connection.execute("DELETE FROM predictions WHERE metric_name = ?", (metric_name,))
            self._connection.execute("DELETE FROM anomalies WHERE metric_name = ?", (metric_name,))
            self._commit()

    def flush_writes(self) -> None:
        with self._lock:
            if self._pending_writes > 0:
                self._commit()

    def prune_old_data(self, before_timestamp: int) -> int:
        total = 0
        with self._lock:
            if self._pending_writes > 0:
                self._commit()
            for sql in _PRUNE_STATEMENTS:
                cursor = self._connection.execute(
                    sql,
                    (before_timestamp,),
                )
                total += max(cursor.rowcount, 0)
            if total > 0:
                self._commit()
                logger.debug("SQLite pruned %d rows before ts=%d", total, before_timestamp)
        return total

    def should_drop_metric_on_ttl(self) -> bool:
        return False

    def _row_to_metric_point(self, row: sqlite3.Row) -> MetricPoint:
        return MetricPoint(
            metric_name=str(row["metric_name"]),
            timestamp=int(row["timestamp"]),
            value=float(row["value"]),
            business_line=str(row["business_line"]),
            tags=json.loads(row["tags_json"]),
            priority=str(row["priority"]),
            metric_type=str(row["metric_type"]),
            granularity=str(row["granularity"]),
            trusted=bool(row["trusted"]),
            is_interpolated=bool(row["is_interpolated"]),
        )

    def close(self) -> None:
        with self._lock:
            if self._pending_writes > 0:
                self._commit()
            with suppress(sqlite3.Error):
                self._connection.close()

    def _execute_write(self, sql: str, params: tuple[object, ...], context: str) -> None:
        with self._lock:
            try:
                self._connection.execute(sql, params)
                self._maybe_commit()
            except sqlite3.Error:
                try:
                    self._connection.rollback()
                except sqlite3.Error:
                    pass
                self._pending_writes = 0
                logger.exception("SQLite history write failed: %s", context)
                raise

    def _maybe_commit(self) -> None:
        self._pending_writes += 1
        if self._pending_writes >= self._batch_size:
            self._commit()

    def _commit(self) -> None:
        self._connection.commit()
        self._pending_writes = 0
