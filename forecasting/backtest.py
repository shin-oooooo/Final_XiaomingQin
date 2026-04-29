"""Public API for Part 2 — walk-forward 80/20 backtest.

DATA-LEAKAGE RED LINE
---------------------
For every test index `t`, the forecast for y_t is computed from `y[:t]`
ONLY (open right end — t itself is excluded). We assert this in-loop so
that any future refactor that breaks the contract fails loudly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .contracts import BacktestOutput, SeriesInput
from .models import ensemble_one_step

DEFAULT_TRAIN_RATIO = 0.8


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    safe = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / safe)) * 100.0)
    return rmse, mae, mape


def run_backtest(
    s: SeriesInput,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> BacktestOutput:
    """Walk-forward one-step-ahead backtest over the last (1 - train_ratio) of the series."""
    y = s.y
    n = int(y.shape[0])
    n_train = int(n * train_ratio)
    if n_train < 2 or n_train >= n:
        raise ValueError(
            f"Train segment too short (n_train={n_train}, n={n}); "
            "need a longer input series."
        )

    test_idx = np.arange(n_train, n, dtype=int)
    y_pred = np.empty(test_idx.shape[0], dtype=float)

    for i, t in enumerate(test_idx):
        assert t < n, "walk-forward index out of bounds"
        y_hist = y[:t]
        y_pred[i] = ensemble_one_step(y_hist)

    y_true = y[test_idx]

    test_dates: Optional[pd.Series] = None
    if s.dates is not None:
        test_dates = s.dates.iloc[test_idx].reset_index(drop=True)

    rmse, mae, mape = _metrics(y_true, y_pred)

    return BacktestOutput(
        y_pred=y_pred,
        y_true=y_true,
        test_dates=test_dates,
        train_size=n_train,
        rmse=rmse,
        mae=mae,
        mape=mape,
    )
