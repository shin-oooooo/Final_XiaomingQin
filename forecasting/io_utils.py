"""Excel input/validation/output helpers.

Per Exam_question.txt:
  - Input is .xlsx, first sheet, with required column 'y' and optional
    'date'. y must be numeric, ordered oldest -> newest, and contain no
    missing values.
  - Output for backtest is .xlsx with column 'y' (and 'date' iff input
    had 'date'), aligned with the test-segment row order.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd

from .contracts import SeriesInput

MIN_SAMPLES = 20


def read_excel_input(file) -> pd.DataFrame:
    """Read the first sheet of an uploaded .xlsx into a DataFrame.

    `file` may be a path, a file-like object, or a Streamlit
    UploadedFile — pandas handles all three through openpyxl.
    """
    df = pd.read_excel(file, sheet_name=0, engine="openpyxl")
    return df


def validate_series(df: pd.DataFrame) -> SeriesInput:
    """Validate the uploaded DataFrame and return a SeriesInput.

    Raises ValueError with a user-friendly message on any violation of
    the exam input contract.
    """
    if df is None or df.empty:
        raise ValueError("Uploaded file is empty.")

    if "y" not in df.columns:
        raise ValueError("Missing required column 'y'.")

    y_raw = df["y"]
    try:
        y = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Column 'y' must be numeric: {exc}") from exc

    if np.isnan(y).any():
        raise ValueError(
            "Column 'y' contains missing values; please clean the file before upload."
        )

    if y.shape[0] < MIN_SAMPLES:
        raise ValueError(
            f"Series too short (N={y.shape[0]}); need at least {MIN_SAMPLES} samples."
        )

    dates: Optional[pd.Series] = None
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"], errors="raise").reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Column 'date' is not a valid date format: {exc}") from exc
        if len(dates) != y.shape[0]:
            raise ValueError("Columns 'date' and 'y' have different lengths.")

    return SeriesInput(y=y, dates=dates)


def build_backtest_dataframe(
    y_pred: np.ndarray,
    test_dates: Optional[pd.Series],
) -> pd.DataFrame:
    """Build the export DataFrame: column 'y' (and 'date' if available).

    `date` is normalized to midnight (date-only) so that downstream Excel
    cells render as `yyyy-mm-dd` instead of a 19-char datetime string.
    """
    if test_dates is not None:
        d = pd.to_datetime(test_dates).dt.normalize().reset_index(drop=True)
        return pd.DataFrame({"date": d, "y": y_pred})
    return pd.DataFrame({"y": y_pred})


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to .xlsx bytes (no index column).

    Sets explicit column widths and a `yyyy-mm-dd` number format on the
    `date` column so Excel does not render dates as `#####` (the symptom
    of a too-narrow column with the default `yyyy-mm-dd hh:mm:ss` format).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="forecast", index=False)
        ws = writer.sheets["forecast"]
        for col_idx, col_name in enumerate(df.columns, start=1):
            letter = ws.cell(row=1, column=col_idx).column_letter
            if col_name == "date":
                ws.column_dimensions[letter].width = 14
                for cell in ws[letter][1:]:
                    cell.number_format = "yyyy-mm-dd"
            else:
                ws.column_dimensions[letter].width = 16
    return buf.getvalue()
