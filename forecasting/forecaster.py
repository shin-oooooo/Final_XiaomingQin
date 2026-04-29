"""Public API for Part 1 — one-step-ahead forecast for y_{N+1}."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .contracts import ForecastOutput, SeriesInput
from .models import ensemble_one_step


def _next_date(dates: Optional[pd.Series]) -> Optional[pd.Timestamp]:
    """Best-effort next-period stamp purely for display (no modeling use)."""
    if dates is None or len(dates) < 2:
        return None
    last = dates.iloc[-1]
    if len(dates) >= 2:
        delta = dates.iloc[-1] - dates.iloc[-2]
        if pd.notna(delta) and delta > pd.Timedelta(0):
            return last + delta
    return last + pd.tseries.offsets.BDay(1)


def forecast_next(s: SeriesInput) -> ForecastOutput:
    """Compute the single one-step-ahead forecast required by Part 1.

    Uses the full history `y` (i.e. `y[:N]`); never sees the future.
    """
    y_hat = ensemble_one_step(s.y)
    return ForecastOutput(y_hat_next=y_hat, next_date=_next_date(s.dates))
