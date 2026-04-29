"""Public dataclasses shared across the forecasting package.

These are the only structures the UI (`app.py`) ever sees from the
`forecasting` package, so their shape is a contract:
  - SeriesInput   -> validated user-uploaded series
  - ForecastOutput-> result of one-step-ahead forecast (Part 1)
  - BacktestOutput-> result of walk-forward backtest    (Part 2)

Adding fields is allowed; renaming/removing existing ones is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesInput:
    """Validated univariate series ready for modeling.

    `y` is always a 1-D float64 ndarray with no NaNs.
    `dates` is either None (input had no `date` column) or a pandas
    Series of datetime64[ns] aligned 1-1 with `y`.
    """

    y: np.ndarray
    dates: Optional[pd.Series]

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def has_dates(self) -> bool:
        return self.dates is not None


@dataclass(frozen=True)
class ForecastOutput:
    """Single one-step-ahead forecast for y_{N+1}."""

    y_hat_next: float
    next_date: Optional[pd.Timestamp]


@dataclass(frozen=True)
class BacktestOutput:
    """Walk-forward backtest result on the 20% test segment.

    `y_pred` and `y_true` are 1-D float64 arrays of identical length
    n_test = N - int(N * train_ratio).
    `test_dates` is None iff the input had no `date` column.
    """

    y_pred: np.ndarray
    y_true: np.ndarray
    test_dates: Optional[pd.Series]
    train_size: int
    rmse: float
    mae: float
    mape: float
