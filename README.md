# Time Series Forecasting Web App (Final Exam Submission)

A Streamlit web application for one-step-ahead forecasting + walk-forward
backtesting of a uploaded univariate time series.

## Live App

- **Web App URL**: *(fill in after deploying to Streamlit Community Cloud)*
- **GitHub**: *(this repository's URL)*

## What the App Does

After the user uploads an Excel file (.xlsx) containing a column `y`
(optionally a `date` column), the app **automatically**:

1. **Part 1 — One-step-ahead forecast**: produces a single point estimate
  for `y_{N+1}` using only `y_1 … y_N`.
2. **Part 2 — Walk-forward backtest**: splits the series 80% train / 20%
  test (by time order) and, for each test index `t`, fits/refreshes the
   model on `y[:t]` only and predicts `y_t` (no data leakage).
3. **Excel export**: a downloadable `.xlsx` whose single column `y` (and
  `date` if the input had it) holds the forecasted values for the test
   segment, in the same row order as the test period.

The user does **not** choose a method — everything runs automatically on
upload, as required by the exam rubric.

## Forecasting Method

Robust **simple-average ensemble** of three classical models, each
re-fit on `y[:t]` only (i.e. no peek into the future):

- **Naive** (random-walk last value) — strong baseline for log-prices.
- **Theta method** (`statsmodels.tsa.forecasting.theta.ThetaModel`) —
M3-competition-grade simple model.
- **Damped Holt** (`statsmodels.tsa.holtwinters.Holt`, damped trend) —
trend smoothing without runaway extrapolation.

Each individual model is wrapped with a try/except that falls back to
`naive` if statsmodels rejects a particular history (e.g. degenerate
short windows). This guarantees the app never crashes on any input.

## Local Run (conda)

```powershell
# 1) create & activate a fresh env
conda create -n tsforecast python=3.11 -y
conda activate tsforecast

# 2) install deps
pip install -r requirements.txt

# 3) run the app
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Excel Schemas

### Input


| column | required | type    | notes                                      |
| ------ | -------- | ------- | ------------------------------------------ |
| `y`    | yes      | numeric | univariate series, oldest → newest, no NaN |
| `date` | no       | date    | display/output alignment only              |


### Output (downloaded after backtest)


| column | always                   | notes                              |
| ------ | ------------------------ | ---------------------------------- |
| `y`    | yes                      | predicted values, one per test row |
| `date` | only if input had `date` | aligned with the test period       |


## Project Layout

```
.
├── app.py                  # Streamlit UI + orchestration
├── forecasting/
│   ├── __init__.py         # public API: forecast_next, run_backtest
│   ├── contracts.py        # @dataclass SeriesInput, ForecastOutput, BacktestOutput
│   ├── io_utils.py         # Excel read/validate/write
│   ├── models.py           # private _naive, _theta, _damped_holt
│   ├── forecaster.py       # public forecast_next()
│   └── backtest.py         # public run_backtest()
├── requirements.txt
├── runtime.txt             # python-3.11 for Streamlit Cloud
├── .streamlit/config.toml
├── dataset.xlsx            # provided sample for testing
└── Exam_question.txt
```

## Anti-Data-Leakage Statement

All forecasts at test index `t` are computed using **only** `y[:t]`
(strict slice, exclusive of `t` itself). See `forecasting/backtest.py`
for the explicit slice and the assert.