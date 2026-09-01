# Time-Series Forecasting — CMPE 255 Assignment 1 Part 2

This directory contains a reproduction-ready, CRISP-DM-based forecasting experiment. The notebook is designed to run against a user-supplied univariate time-series CSV and to stop safely—with actionable setup instructions—when that file is absent.

## Dataset and Execution Disclaimer

The original source dataset is **not included or executed** because of dataset-access constraints. The complete forecasting methodology is implemented, but actual forecasts, metric values, residual findings, and a best-model conclusion are intentionally omitted. Running the notebook with an appropriate dataset is required before making numerical or model-selection claims.

## Setup

```bash
cd Assignment_1_Part_2/06_Time_Series_Forecasting
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook time_series_forecasting.ipynb
```

## Supply the data

1. Put a CSV at `data/time_series.csv` (the `data/` directory is intentionally not supplied), **or** set `TS_DATA_PATH` to another CSV path.
2. The CSV must contain one datetime column and one numeric target column.
3. By default, the notebook tries common names and otherwise asks you to set `TS_DATE_COLUMN` and `TS_TARGET_COLUMN` environment variables.
4. Optionally set `TS_FREQUENCY` (for example `D`, `W`, or `MS`), `TS_TEST_SIZE`, `TS_SEASONAL_PERIOD`, and `TS_MA_WINDOW`.
5. Restart the kernel and run all cells. Generated charts are saved under `images/` only after the pipeline runs.

Example:

```bash
TS_DATA_PATH=/absolute/path/series.csv \
TS_DATE_COLUMN=date TS_TARGET_COLUMN=value TS_FREQUENCY=D \
jupyter notebook time_series_forecasting.ipynb
```

## Methodology

The notebook follows CRISP-DM: business understanding, data understanding, data preparation, modeling, evaluation, and deployment considerations. It includes datetime validation, duplicate and missing-date checks, training-only missing-value handling, time-series EDA, rolling statistics, trend/seasonality diagnostics, and a chronological holdout split. It evaluates naive, moving-average, Simple Exponential Smoothing, Holt trend, Holt-Winters seasonal, and optional lag-feature regression forecasts with MAE, RMSE, and zero-safe MAPE.

Temporal leakage is explicitly controlled: the test set remains chronologically later than training; imputation parameters and regression features are learned or constructed without future observations; and forecasting models are fit only on training data.

## Reproducibility notes

- Configuration and random seed are centralized near the top of the notebook.
- Model eligibility is checked against available history and seasonal period.
- MAPE is reported only over nonzero actuals and is marked unavailable if no such observations exist.
- Model selection uses computed holdout metrics only; no model is declared best in this repository state.
- Forecast and residual plots are produced only when real data has been supplied and evaluation has run.

## Generated files

- `README.md` — setup, data contract, methodology, and disclaimer.
- `time_series_forecasting.ipynb` — executable end-to-end experiment.
- `requirements.txt` — Python dependencies.
- `prompt.txt` — exact assignment prompt.
- `images/` — destination for generated figures (initially empty).

## Integrity statement

No numerical forecasts, MAE, RMSE, MAPE, residual conclusions, or best-model claims were fabricated. Only files inside `Assignment_1_Part_2/06_Time_Series_Forecasting` were created or changed for this work.
