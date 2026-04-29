"""Sanity-check the simulated datasets against the real webapp pipeline.

For each generated .xlsx we exercise the exact same code path the
Streamlit app uses on upload:
  1) read_excel_input + validate_series  (input contract)
  2) forecast_next                       (Part 1)
  3) run_backtest + build_backtest_dataframe + to_excel_bytes (Part 2)

If any file violates the contract we want to know now, before
shipping these files to a user as test fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecasting import (
    build_backtest_dataframe,
    forecast_next,
    read_excel_input,
    run_backtest,
    to_excel_bytes,
    validate_series,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "simulated_datasets"

FILES = [
    "sim_similar.xlsx",
    "sim_diff_trend_seasonal.xlsx",
    "sim_diff_mean_reverting.xlsx",
    "sim_diff_volatile_jumps.xlsx",
]


def main() -> None:
    header = (
        f"{'file':40s} {'N':>4s} {'date?':>5s} {'y_{N+1}':>12s} "
        f"{'RMSE':>10s} {'MAE':>10s} {'MAPE%':>8s} {'rows':>5s}"
    )
    print(header)
    print("-" * len(header))

    for fname in FILES:
        path = DATA_DIR / fname

        df_raw = read_excel_input(path)
        series = validate_series(df_raw)

        fcst = forecast_next(series)
        bt = run_backtest(series, train_ratio=0.8)

        out_df = build_backtest_dataframe(bt.y_pred, bt.test_dates)
        out_bytes = to_excel_bytes(out_df)

        n = series.n
        has_date = series.has_dates
        print(
            f"{fname:40s} {n:>4d} {str(has_date):>5s} "
            f"{fcst.y_hat_next:>12.4f} "
            f"{bt.rmse:>10.4f} {bt.mae:>10.4f} "
            f"{bt.mape:>7.3f}% "
            f"{len(out_df):>5d}"
        )

        assert "y" in out_df.columns, f"{fname}: output must have 'y' column"
        if has_date:
            assert "date" in out_df.columns, (
                f"{fname}: input had date, output must too"
            )
        else:
            assert "date" not in out_df.columns, (
                f"{fname}: input lacked date, output must NOT include date"
            )
        assert len(out_df) == n - int(n * 0.8), (
            f"{fname}: output rows must equal N - int(N*0.8)"
        )
        assert len(out_bytes) > 0

    print(
        "\nAll 4 simulated datasets pass the full upload -> forecast -> "
        "backtest -> excel-export pipeline."
    )


if __name__ == "__main__":
    main()
