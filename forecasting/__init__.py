"""Public API of the `forecasting` package.

`app.py` should import from here; nothing in the UI layer should
reach into private model code.
"""

from .backtest import run_backtest
from .contracts import BacktestOutput, ForecastOutput, SeriesInput
from .forecaster import forecast_next
from .io_utils import (
    build_backtest_dataframe,
    read_excel_input,
    to_excel_bytes,
    validate_series,
)

__all__ = [
    "BacktestOutput",
    "ForecastOutput",
    "SeriesInput",
    "build_backtest_dataframe",
    "forecast_next",
    "read_excel_input",
    "run_backtest",
    "to_excel_bytes",
    "validate_series",
]
