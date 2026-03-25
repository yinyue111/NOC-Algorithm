from __future__ import annotations

import math
import warnings
from statistics import NormalDist
from typing import Iterable

import numpy as np

try:
    from scipy.stats import gaussian_kde as scipy_gaussian_kde
except Exception:  # pragma: no cover - optional dependency
    scipy_gaussian_kde = None

try:
    from sklearn.linear_model import HuberRegressor
    from sklearn.exceptions import ConvergenceWarning
except Exception:  # pragma: no cover - optional dependency
    HuberRegressor = None
    ConvergenceWarning = None


EPS = 1e-6


def to_float_array(values: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return np.asarray([], dtype=float)
    return data[np.isfinite(data)]


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    arr = to_float_array(values)
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def safe_std(values: Iterable[float], default: float = 0.0) -> float:
    arr = to_float_array(values)
    if arr.size <= 1:
        return default
    return float(np.std(arr))


def safe_percentile(values: Iterable[float], percentile: float, default: float = 0.0) -> float:
    arr = to_float_array(values)
    if arr.size == 0:
        return default
    return float(np.percentile(arr, percentile))


def rolling_mae(actual_values: Iterable[float], predicted_values: Iterable[float]) -> float:
    actual = to_float_array(actual_values)
    predicted = to_float_array(predicted_values)
    if actual.size == 0 or predicted.size == 0:
        return 0.0
    length = min(actual.size, predicted.size)
    return float(np.mean(np.abs(actual[-length:] - predicted[-length:])))


def coefficient_of_variation(values: Iterable[float]) -> float:
    arr = to_float_array(values)
    if arr.size <= 1:
        return 0.0
    mean = float(np.mean(np.abs(arr)))
    if mean <= EPS:
        return 0.0
    return float(np.std(arr) / mean)


def autocorrelation_at_lag(values: Iterable[float], lag: int) -> float:
    arr = to_float_array(values)
    if arr.size <= lag or lag <= 0:
        return 0.0
    left = arr[:-lag]
    right = arr[lag:]
    if np.std(left) <= EPS or np.std(right) <= EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def periodicity_score(values: Iterable[float], candidate_lags: Iterable[int]) -> float:
    scores = [max(autocorrelation_at_lag(values, lag), 0.0) for lag in candidate_lags if lag > 0]
    if not scores:
        return 0.0
    return float(max(scores))


def gaussian_kde_bounds(
    samples: Iterable[float],
    lower_quantile: float,
    upper_quantile: float,
    bandwidth: float | None = None,
) -> tuple[float, float]:
    data = to_float_array(samples)
    if data.size < 20:
        if data.size == 0:
            return -3.0, 3.0
        return (
            float(np.quantile(data, lower_quantile)),
            float(np.quantile(data, upper_quantile)),
        )

    if scipy_gaussian_kde is not None:
        try:
            kde = scipy_gaussian_kde(data, bw_method=bandwidth)
            lower = float(np.min(data)) - 3.0 * float(np.std(data))
            upper = float(np.max(data)) + 3.0 * float(np.std(data))
            grid = np.linspace(lower, upper, 1024)
            density = kde(grid)
            cdf = np.cumsum(density)
            cdf /= max(float(cdf[-1]), EPS)
            return (
                float(np.interp(lower_quantile, cdf, grid)),
                float(np.interp(upper_quantile, cdf, grid)),
            )
        except Exception:
            pass

    std = float(np.std(data))
    if bandwidth is None:
        bandwidth = max(1.06 * std * (data.size ** (-0.2)), 0.1)

    lower = float(np.min(data) - 3 * bandwidth)
    upper = float(np.max(data) + 3 * bandwidth)
    grid = np.linspace(lower, upper, 512)

    diff = (grid[np.newaxis, :] - data[:, np.newaxis]) / bandwidth
    density = np.exp(-0.5 * diff ** 2).sum(axis=0)
    density /= max(data.size * bandwidth * math.sqrt(2 * math.pi), EPS)

    cdf = np.cumsum(density)
    cdf /= max(float(cdf[-1]), EPS)

    return (
        float(np.interp(lower_quantile, cdf, grid)),
        float(np.interp(upper_quantile, cdf, grid)),
    )


def normalized_severity(z_score: float, lower_bound: float, upper_bound: float) -> float:
    if lower_bound >= upper_bound:
        return min(abs(z_score) / 3.0, 1.0)

    if z_score < lower_bound:
        denom = abs(lower_bound) + EPS
        return float(min(abs(z_score - lower_bound) / denom, 1.0))
    if z_score > upper_bound:
        denom = abs(upper_bound) + EPS
        return float(min(abs(z_score - upper_bound) / denom, 1.0))
    width = max((upper_bound - lower_bound) / 2.0, EPS)
    center = (upper_bound + lower_bound) / 2.0
    return float(min(abs(z_score - center) / width, 1.0))


def boxplot_bounds(values: Iterable[float]) -> tuple[float, float]:
    arr = to_float_array(values)
    if arr.size < 4:
        mean = safe_mean(arr, 0.0)
        std = safe_std(arr, 0.0)
        return mean - 3 * std, mean + 3 * std
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def mann_kendall_test(values: Iterable[float]) -> tuple[bool, float]:
    arr = to_float_array(values)
    n = arr.size
    if n < 5:
        return False, 1.0

    i_idx, j_idx = np.triu_indices(n, k=1)
    s = int(np.sign(arr[j_idx] - arr[i_idx]).sum())

    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= EPS:
        return False, 1.0

    if s > 0:
        z_score = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z_score = (s + 1) / math.sqrt(var_s)
    else:
        z_score = 0.0

    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_score)))
    return bool(p_value < 0.05), float(p_value)


def _weighted_ridge_regression(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones_like(x), x])
    ridge = alpha * np.eye(design.shape[1])
    lhs = (design.T * weights) @ design + ridge
    rhs = design.T @ (weights * y)
    return np.linalg.solve(lhs, rhs)


def huber_linear_forecast(
    values: Iterable[float],
    epsilon: float = 1.35,
    alpha: float = 0.0001,
    max_iter: int = 50,
    steps_ahead: int = 1,
) -> tuple[float, float, tuple[float, float]]:
    y = to_float_array(values)
    if y.size == 0:
        return 0.0, 0.0, (0.0, 0.0)
    if y.size == 1:
        return float(y[-1]), 0.0, (float(y[-1]), 0.0)

    if HuberRegressor is not None:
        try:
            x_raw = np.arange(1, y.size + 1, dtype=float)
            x_mean = float(np.mean(x_raw))
            x_std = max(float(np.std(x_raw)), 1.0)
            x = ((x_raw - x_mean) / x_std).reshape(-1, 1)
            prediction_x = np.asarray([[(float(y.size + steps_ahead) - x_mean) / x_std]])
            model = HuberRegressor(
                epsilon=epsilon,
                alpha=alpha,
                max_iter=max_iter,
                fit_intercept=True,
            )
            with warnings.catch_warnings():
                if ConvergenceWarning is not None:
                    warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x, y)
            prediction = float(model.predict(prediction_x)[0])
            residuals = y - model.predict(x)
            residual_std = max(safe_std(residuals, 0.0), EPS)
            return prediction, residual_std, (float(model.intercept_), float(model.coef_[0]))
        except Exception:
            pass

    x = np.arange(1, y.size + 1, dtype=float)
    weights = np.ones_like(y)
    beta = _weighted_ridge_regression(x, y, weights, alpha)

    for _ in range(max_iter):
        design = np.column_stack([np.ones_like(x), x])
        residuals = y - design @ beta
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = max(1.4826 * mad, safe_std(residuals, 1.0), EPS)
        cutoff = epsilon * scale
        new_weights = np.ones_like(residuals)
        mask = np.abs(residuals) > cutoff
        new_weights[mask] = cutoff / np.abs(residuals[mask])
        new_beta = _weighted_ridge_regression(x, y, new_weights, alpha)
        if np.allclose(new_beta, beta, atol=1e-6, rtol=1e-5):
            beta = new_beta
            break
        beta = new_beta
        weights = new_weights

    prediction_x = float(y.size + steps_ahead)
    prediction = beta[0] + beta[1] * prediction_x
    design = np.column_stack([np.ones_like(x), x])
    residuals = y - design @ beta
    residual_std = max(safe_std(residuals, 0.0), EPS)
    return float(prediction), float(residual_std), (float(beta[0]), float(beta[1]))
