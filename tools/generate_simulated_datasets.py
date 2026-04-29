"""Generate simulated test datasets for the forecasting webapp.

All outputs strictly satisfy the input contract from `Exam_question.txt`:
  - .xlsx file, first sheet read by pandas
  - required numeric column `y` (no missing values), oldest -> newest row order
  - optional `date` column (Excel date / ISO string)

We produce 4 datasets:
  1) sim_similar.xlsx
       Construction mirrors the original: simulate K correlated log-close
       price paths via correlated GBMs and take a positive linear
       combination. Scale, drift, and volatility are tuned to match
       dataset.xlsx (mean ~1.36, std ~0.008).
  2) sim_diff_trend_seasonal.xlsx
       Strong upward linear trend + weekly + monthly seasonal cycles +
       small AR(1) noise. Magnitude ~150. Calendar-day index.
  3) sim_diff_mean_reverting.xlsx
       Stationary AR(1) with strong mean reversion (phi = 0.85) around
       a fixed level (50). NO `date` column (exercises the optional
       path of the schema).
  4) sim_diff_volatile_jumps.xlsx
       Jump-diffusion (GBM + Poisson jumps) with regime-switching
       volatility. Magnitude ~1000, daily returns std ~3-5%. Business-day
       index.

Run from project root:
    python tools/generate_simulated_datasets.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "simulated_datasets"


def _bdays(start: str, n: int) -> pd.DatetimeIndex:
    """N consecutive business days starting at `start` (inclusive)."""
    return pd.bdate_range(start=start, periods=n)


def _cdays(start: str, n: int) -> pd.DatetimeIndex:
    """N consecutive calendar days starting at `start` (inclusive)."""
    return pd.date_range(start=start, periods=n, freq="D")


def _to_excel(df: pd.DataFrame, path: Path) -> None:
    """Write df to .xlsx; format `date` as yyyy-mm-dd if present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        ws = writer.sheets["data"]
        for col_idx, col_name in enumerate(df.columns, start=1):
            letter = ws.cell(row=1, column=col_idx).column_letter
            if col_name == "date":
                ws.column_dimensions[letter].width = 14
                for cell in ws[letter][1:]:
                    cell.number_format = "yyyy-mm-dd"
            else:
                ws.column_dimensions[letter].width = 16


# ---------------------------------------------------------------------------
# 1) SIMILAR: linear combination of K correlated log-close price paths
# ---------------------------------------------------------------------------
def make_similar(n: int = 500, seed: int = 20260429) -> pd.DataFrame:
    """Replicate the original construction: log(GBM) paths combined linearly.

    The original dataset.xlsx has y mean ~1.359, std ~0.0084 over 500
    business days. We pick K=4 stocks with mild positive correlation,
    drift mu ~ 5-15%/yr, vol sigma ~ 15-25%/yr, and combine via a
    convex weight vector. The mean log-close levels are tuned so the
    weighted sum sits in the ~1.36 range.
    """
    rng = np.random.default_rng(seed)
    K = 4
    dt = 1.0 / 252.0

    # per-stock annualized drift / vol
    mu = rng.uniform(0.04, 0.15, size=K)
    sigma = rng.uniform(0.15, 0.25, size=K)

    # mild positive correlation between stocks
    rho = 0.35
    corr = np.full((K, K), rho)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    # daily correlated normals -> log-returns -> log-prices
    z = rng.standard_normal(size=(n, K))
    z_corr = z @ L.T
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_corr

    # initial log-prices chosen so the convex combination starts ~1.36
    log_p0 = rng.uniform(2.5, 4.0, size=K)
    log_prices = log_p0 + np.cumsum(log_returns, axis=0)

    # convex weights
    w = rng.dirichlet(np.ones(K) * 5.0)

    # raw linear combination
    y_raw = log_prices @ w

    # Affine rescale to match the original (mean 1.359, std 0.0084) so
    # the "similar" dataset is genuinely indistinguishable in summary
    # statistics from dataset.xlsx, while keeping the underlying GBM
    # dynamics intact.
    target_mean, target_std = 1.359, 0.0084
    y = (y_raw - y_raw.mean()) / y_raw.std() * target_std + target_mean

    dates = _bdays("2024-03-18", n)
    return pd.DataFrame({"date": dates.date, "y": y})


# ---------------------------------------------------------------------------
# 2) DIFFERENT: strong trend + seasonality
# ---------------------------------------------------------------------------
def make_trend_seasonal(n: int = 365, seed: int = 7) -> pd.DataFrame:
    """Linear trend + weekly + monthly sinusoids + AR(1) noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    level0 = 100.0
    slope = 0.12          # +0.12 per day -> +44 over a year
    weekly = 4.0 * np.sin(2 * np.pi * t / 7.0)
    monthly = 8.0 * np.sin(2 * np.pi * t / 30.0 + 0.7)

    # AR(1) noise with phi=0.6, innovation std=1.5
    eps = rng.standard_normal(n) * 1.5
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + eps[i]

    y = level0 + slope * t + weekly + monthly + noise

    dates = _cdays("2025-01-01", n)
    return pd.DataFrame({"date": dates.date, "y": y})


# ---------------------------------------------------------------------------
# 3) DIFFERENT: stationary AR(1), no `date` column
# ---------------------------------------------------------------------------
def make_mean_reverting(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """AR(1) around a constant level: y_t = mu + phi*(y_{t-1}-mu) + eps."""
    rng = np.random.default_rng(seed)
    mu = 50.0
    phi = 0.85
    sigma_eps = 1.2

    y = np.empty(n)
    y[0] = mu + rng.standard_normal() * sigma_eps / np.sqrt(1 - phi**2)
    for i in range(1, n):
        y[i] = mu + phi * (y[i - 1] - mu) + rng.standard_normal() * sigma_eps

    return pd.DataFrame({"y": y})


# ---------------------------------------------------------------------------
# 4) DIFFERENT: jump-diffusion + regime-switching volatility
# ---------------------------------------------------------------------------
def make_volatile_jumps(n: int = 800, seed: int = 1234) -> pd.DataFrame:
    """GBM with Poisson jumps and a calm/turbulent regime switch.

    Daily log-returns:
        r_t = mu*dt + sigma_regime(t) * sqrt(dt) * z_t + J_t * 1{N_t = 1}
    where regime is sampled as a 2-state HMM (calm vs turbulent) and
    J_t ~ Normal(jump_mean, jump_std).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = 0.06
    sigma_calm = 0.20
    sigma_turb = 0.55
    jump_intensity = 6.0 / 252.0  # ~6 jumps/yr in expectation
    jump_mean = -0.005
    jump_std = 0.04

    # 2-state Markov regime (calm, turbulent), sticky transitions
    P = np.array([[0.97, 0.03], [0.10, 0.90]])
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        regime[i] = rng.choice(2, p=P[regime[i - 1]])
    sigma_t = np.where(regime == 0, sigma_calm, sigma_turb)

    z = rng.standard_normal(n)
    diffusion = (mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * z

    n_jumps = rng.poisson(lam=jump_intensity, size=n)
    jumps = np.where(
        n_jumps > 0,
        rng.normal(loc=jump_mean, scale=jump_std, size=n) * n_jumps,
        0.0,
    )

    log_returns = diffusion + jumps
    log_p = np.log(1000.0) + np.cumsum(log_returns)
    y = np.exp(log_p)

    dates = _bdays("2023-01-02", n)
    return pd.DataFrame({"date": dates.date, "y": y})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        ("sim_similar.xlsx", make_similar()),
        ("sim_diff_trend_seasonal.xlsx", make_trend_seasonal()),
        ("sim_diff_mean_reverting.xlsx", make_mean_reverting()),
        ("sim_diff_volatile_jumps.xlsx", make_volatile_jumps()),
    ]

    for fname, df in specs:
        path = OUT_DIR / fname
        _to_excel(df, path)
        print(
            f"{fname:40s}  N={len(df):4d}  cols={list(df.columns)}  "
            f"y[mean={df['y'].mean():.4f}, std={df['y'].std():.4f}, "
            f"min={df['y'].min():.4f}, max={df['y'].max():.4f}]  -> {path}"
        )


if __name__ == "__main__":
    main()
