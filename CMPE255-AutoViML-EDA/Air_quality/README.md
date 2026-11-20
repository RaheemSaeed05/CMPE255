# Air Quality Analysis and Next-Hour CO Forecasting

Can air-quality sensors predict next hour’s carbon monoxide (CO) concentration more accurately than assuming it will stay at its current level? On 1,781 test forecasts, AutoViML CatBoost reduced mean absolute error by about 15% compared with that baseline. It tracked the daily rise and fall in CO, though it underestimated several sharp peaks.

## Project files

| File | Contents |
| --- | --- |
| [02_Air_Quality_TimeSeries.ipynb](./02_Air_Quality_TimeSeries.ipynb) | Executed Colab notebook with the analysis, model comparisons, and plots |
| [AirQuality.csv](./AirQuality.csv) | Original hourly pollutant, sensor, temperature, and humidity readings |
| [air_quality_artifacts.zip](./air_quality_artifacts.zip) | Prepared datasets, model scores, timestamped predictions, diagnostic plot, feature rankings, and anomaly audit |

## Data preparation and exploration

After removing empty rows and columns, the dataset contained 9,357 consecutive hourly timestamps from March 2004 through April 2005. I treated `-200` as a missing reading and dropped `NMHC(GT)` because more than 80% of its training values were missing.

CO concentrations tended to peak around 9 AM and 7 PM. To capture that pattern, I combined the hour of day with current sensor readings, previous CO readings, and rolling averages. Each forecast used only information available through the current hour. I kept the splits in chronological order and scored predictions only when the next hour had an observed CO value.

K-means identified two groups with median CO concentrations of 1.3 and 3.1 mg/m³. Isolation Forest flagged 42 unusual training hours. Removing those hours increased Ridge’s validation error, so I kept them in the final training data.

## Forecasting results

I compared Ridge regression, AutoViML Random Forest, and AutoViML CatBoost against two baselines. CatBoost had the lowest validation mean absolute error (MAE), so I selected it for the final test period. Both error metrics below are in mg/m³. Lower values are better.

| Model | Test MAE | Test RMSE |
| --- | ---: | ---: |
| Training-median baseline | 1.019 | 1.399 |
| Persistence baseline (carry the current CO reading forward) | 0.521 | 0.809 |
| **AutoViML CatBoost** | **0.441** | **0.652** |

## Run the notebook

1. Open `02_Air_Quality_TimeSeries.ipynb` in Google Colab and upload `AirQuality.csv` when prompted.
2. Run the preparation cells and save the forecast checkpoint ZIP when it downloads.
3. Run the dependency installation cell. If Colab prompts you to restart, restart the session and run the checkpoint restore cell with the saved ZIP.
4. Continue through model comparison, test evaluation, and the final export cell to generate `air_quality_artifacts.zip`.

