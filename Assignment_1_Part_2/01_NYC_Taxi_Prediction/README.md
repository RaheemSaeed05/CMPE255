# NYC Taxi Trip Duration Prediction

An end-to-end, reproducible supervised-regression experiment inspired by Kaggle's **NYC Taxi Trip Duration** competition. The analysis is organized around CRISP-DM and deliberately reports no scores until it is run against the real competition data.

> [!IMPORTANT]
> ## Dataset Disclaimer
> The original Kaggle `train.csv` is **not included** because access is governed by
> Kaggle's competition terms and the file is too large for this repository. This
> project is a complete, executable experiment template: it reproduces the full
> methodology, validates the expected input, and runs end to end once the original
> file is supplied. The notebook has **not** been trained or evaluated here, so
> numerical metrics, model rankings, and a winning model are intentionally not
> reported. When the file is absent, the notebook prints supply instructions and
> safely skips data-dependent cells instead of crashing or generating synthetic data.

## Objective

Predict `trip_duration` (seconds) from information available at pickup time. A useful model could support dispatch planning, rider ETAs, and fleet-capacity decisions. MAE is treated as the primary business-facing metric, with RMSE and R² providing complementary views of large errors and explained variance.

## Dataset

Download the competition data from the [Kaggle NYC Taxi Trip Duration data page](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data), accept Kaggle's competition rules if prompted, and place **`train.csv`** in this directory:

```text
Assignment_1_Part_2/01_NYC_Taxi_Prediction/train.csv
```

The dataset is not committed to this repository. The notebook checks the expected schema and, if the file is absent, explains exactly where to obtain and place it before safely skipping the experiment. It never substitutes invented data. To keep runtime reasonable when data is supplied, it draws a deterministic sample of up to 75,000 rows (`random_state=42`); smaller files are used in full.

## CRISP-DM methodology

1. **Business understanding:** defines the prediction objective, intended value, metric interpretation, and leakage constraints.
2. **Data understanding:** validates the source and schema, summarizes types and descriptive statistics, and explores target, time, distance, and passenger behavior.
3. **Data preparation:** analyzes missing values and duplicates, applies defensible validity/outlier rules, and engineers pickup-time and Haversine-distance features.
4. **Modeling:** creates one held-out split and compares a median baseline, linear regression, random forest, and histogram gradient boosting with preprocessing learned only from training data.
5. **Evaluation:** compares MAE, RMSE, and R², plots predictions and residuals, and inspects supported feature importances.
6. **Deployment/conclusion:** recommends a model only after measured results exist and documents limitations and next steps.

## Features

- Pickup hour, day of week, and month
- Passenger count
- Pickup and dropoff longitude/latitude
- Great-circle pickup-to-dropoff distance using the Haversine formula

Identifiers, `dropoff_datetime`, `store_and_fwd_flag`, and the target are excluded from predictors. In particular, dropoff time is unavailable at prediction time and would leak the answer.

## Models tested

- Median `DummyRegressor` baseline
- `LinearRegression`
- `RandomForestRegressor` (bounded tree count/depth and parallel execution)
- `HistGradientBoostingRegressor`

All models use scikit-learn pipelines with median imputation; linear regression additionally standardizes inputs.

## Evaluation metrics

- **MAE:** typical absolute error in seconds; primary selection metric
- **RMSE:** penalizes unusually large errors more heavily
- **R²:** fraction of held-out target variance explained

## Final results

**Pending execution with the real Kaggle `train.csv`.** No model score or winner is claimed in advance. When the notebook is run, its comparison table is sorted by held-out MAE, the best measured model is recommended, and plots are exported to `images/`.

### What is implemented

- Input discovery, expected-schema validation, deterministic sampling, descriptive summaries, and exploratory plots
- Missing-value, exact-duplicate, duplicate-ID, and domain-informed outlier analysis
- Explicit cleaning plus pickup hour/day/month and vectorized Haversine-distance features
- Leakage-safe holdout creation and preprocessing pipelines learned only from training rows
- Median baseline, linear regression, random forest, and histogram gradient boosting
- Held-out MAE, RMSE, and R² comparison; actual-versus-predicted and residual plots
- Random-forest feature importance and data-driven final recommendation logic
- Limitations, future improvements, and a CRISP-DM conclusion

### What could not be executed here

Without the original `train.csv`, no dataset inspection, cleaning counts, charts, model fitting, metric calculation, feature-importance calculation, or final model selection can be performed. Accordingly, the committed notebook has no fabricated or cached outputs. Its control flow can still be run without the data to verify that it exits cleanly with actionable instructions.

## How to run

From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook nyc_taxi_analysis.ipynb
```

Then choose **Kernel → Restart & Run All**. Expected artifacts include `images/target_distribution.png`, `images/trips_by_hour.png`, `images/distance_vs_duration.png`, `images/model_comparison.png`, `images/actual_vs_predicted.png`, and (where supported) `images/feature_importance.png`.

## Reproducibility and practical notes

- Random sampling and splitting use seed `42`.
- The test set is isolated before any learned preprocessing or model fitting.
- Cleaning thresholds are explicit in the notebook and should be revisited for other taxi datasets.
- Runtime depends on hardware but is constrained through sampling and conservative model settings.
- Generated images and notebook outputs can be refreshed by rerunning all cells after installing the requirements.
