"""Private model primitives.

Each function takes a 1-D numpy array `y` (history only — anti-leakage:
callers must pass `y[:t]` during walk-forward) and returns a single
float: the one-step-ahead point forecast for the next index.

All non-naive primitives are wrapped to fall back to `_naive` on any
exception (numerical issues, optimization failures, statsmodels API
quirks for very short windows). This guarantees the ensemble never
crashes on real-world inputs.
"""

from __future__ import annotations

import warnings

import numpy as np

# statsmodels emits convergence / future warnings that are noise here.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _naive(y: np.ndarray) -> float:
    """Random-walk forecast: y_hat_{t+1} = y_t."""
    return float(y[-1])


def _theta(y: np.ndarray) -> float:
    """Hyndman-Billah Theta method (M3 baseline). Falls back to naive."""
    if y.shape[0] < 4:
        return _naive(y)
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        model = ThetaModel(y, period=1, deseasonalize=False)
        fit = model.fit()
        fc = fit.forecast(steps=1)
        val = float(np.asarray(fc).ravel()[0])
        if not np.isfinite(val):
            return _naive(y)
        return val
    except Exception:  # noqa: BLE001
        return _naive(y)


def _damped_holt(y: np.ndarray) -> float:
    """Holt's linear trend with damping. Falls back to naive."""
    if y.shape[0] < 8:
        return _naive(y)
    try:
        from statsmodels.tsa.holtwinters import Holt

        model = Holt(y, damped_trend=True, initialization_method="estimated")
        fit = model.fit(optimized=True, use_brute=False)
        fc = fit.forecast(steps=1)
        val = float(np.asarray(fc).ravel()[0])
        if not np.isfinite(val):
            return _naive(y)
        return val
    except Exception:  # noqa: BLE001
        return _naive(y)


def ensemble_one_step(y: np.ndarray) -> float:
    """Mean of (naive, theta, damped_holt) — the canonical forecast.

    Public on purpose: both `forecaster.forecast_next` and
    `backtest.run_backtest` route through here, so they share a single
    source of truth.
    """
    preds = (_naive(y), _theta(y), _damped_holt(y))
    return float(np.mean(preds))
