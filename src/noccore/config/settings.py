from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineSettings:
    timezone: str = "Asia/Shanghai"
    max_history_points: int = 12000
    history_commit_batch_size: int = 100
    metric_lock_stripes: int = 256
    history_weeks_required: int = 4
    max_missing_fill: int = 3
    jump_clip_multiplier: float = 10.0
    jump_clip_history_window: int = 600
    second_consecutive_points: int = 3
    minute_consecutive_points: int = 2
    global_lower_quantile: float = 0.025
    global_upper_quantile: float = 0.975
    kde_refresh_every_points: int = 300
    kde_refresh_interval_sec: int = 3600
    local_boxplot_window: int = 60
    local_regression_window: int = 30
    local_nsigma_window: int = 60
    local_trend_window: int = 15
    periodic_ratio_threshold: float = 0.60
    periodic_min_matches: int = 3
    periodic_week_lookback: int = 4
    confidence_p0: float = 0.60
    confidence_p1: float = 0.40
    alert_merge_window_sec: int = 300
    alert_upgrade_window_sec: int = 600
    recovery_normal_points: int = 3
    zero_rule_floor: float = 1e-6
    zero_rule_drop_ratio: float = 0.10
    eligibility_retry_interval_sec: int = 86400
    eligibility_refresh_interval_sec: int = 7 * 24 * 3600
    state_ttl_sec: int = 7 * 24 * 3600
    data_retention_sec: int = 30 * 24 * 3600
    state_cleanup_every_points: int = 1000
    shutdown_drain_timeout_sec: float = 2.0
    second_history_window: int = 4000
    minute_history_window: int = 600
    prediction_window: int = 600
    anomaly_window: int = 4000
    priority_route_seconds: dict[str, int] = field(
        default_factory=lambda: {
            "P0": 60,
            "P1": 300,
            "P2": 600,
            "P3": 1800,
        }
    )


DEFAULT_SETTINGS = PipelineSettings()
