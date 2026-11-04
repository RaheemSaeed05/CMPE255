# CRISP-DM — Telecom churn outreach demonstration

The selected Random Forest identified 102 historical churners among 140 selected test accounts: **72.86% precision**, **27.27% recall**, and **46.31 percentage points above random expected precision**. This is a simulated historical ranking exercise. Retention gains and future-month performance have not been established.

## Contents

| Path | Purpose |
|---|---|
| `report.md` | Consolidated six-phase report, results, limitations, and completion status |
| `01_CRISP_DM_Telco_Churn.ipynb` | Executed Colab notebook with phase drafts, critiques, revisions, and checks |
| `requirements.txt` | Package versions recorded in the notebook |
| `artifacts/churn_pipeline.joblib` | Training-fitted preprocessing and Random Forest |
| `artifacts/input_schema.json` | Required input columns and ranking policy |
| `artifacts/environment.json` | Recorded Python and package versions |
| `artifacts/simulated_account_ranking.csv` | All 1,409 demonstration accounts, ranked |
| `artifacts/simulated_outreach_list.csv` | Top 140 accounts |
| `artifacts/validation_results.csv` | Two-model selection comparison |
| `artifacts/test_results.csv` | Fixed model's final historical evaluation |
| `SHA256SUMS.txt` | Integrity hashes for the package files |

## Reproduce the full experiment in Colab

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from https://www.kaggle.com/datasets/blastchar/telco-customer-churn. The raw dataset is not bundled. Review the source's usage terms before redistributing it.
2. Upload/open the notebook in Colab. Compare the runtime versions with `artifacts/environment.json`. `requirements.txt` records the original versions; a fresh installation has not been tested in this package. If dependencies are changed, restart the runtime before executing.
3. Run cells in order. When the CSV upload cell prompts, upload the dataset. It uses `google.colab.files`, so the full notebook is intended for Colab.
4. Read the markdown and verification outputs alongside each phase. The notebook makes the stratified 60/20/20 split using seed 42, explores training relationships, prepares data, compares two models, and evaluates the frozen training-only Random Forest.
5. Run the deployment and export-check cells. They create `01_CRISP_DM/artifacts` relative to the current working directory. In a normal Colab runtime this is under `/content`; avoid treating that path as the root of this downloaded package.
6. Confirm the recorded counts, score-reload check, ranking alignment, and CSV comparisons. Save the executed notebook and exports. Changed runtime versions may affect reproducibility; record discrepancies rather than replacing the original evidence silently.

## Score with the saved model without retraining

Use a separate notebook in a compatible environment. Set `artifact_dir` to the extracted artifacts folder, then initialize:

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

artifact_dir = Path('01_CRISP_DM/artifacts')  # adjust to the actual location
schema = json.loads((artifact_dir / 'input_schema.json').read_text())
predictor_columns = [c for c in schema['required_columns'] if c != 'customerID']
final_contact_fraction = schema['selection_fraction']
loaded_model = joblib.load(artifact_dir / 'churn_pipeline.joblib')
```

Copy **only the `score_accounts` function definition** from the notebook's deployment code cell into this new notebook. Do not run that original cell wholesale: its demonstration also refers to training/test variables. Load an input CSV containing all 20 required columns listed in the schema, then call:

```python
accounts = pd.read_csv('input_accounts.csv')
ranking, selected = score_accounts(accounts, loaded_model)
ranking.to_csv('simulated_account_ranking.csv', index=False)
selected.to_csv('simulated_outreach_list.csv', index=False)
```

`Churn` is not required. The function separates identifiers, converts TotalCharges to numeric and SeniorCitizen to strings, and passes the 19 predictors to the pipeline. Imputation, scaling, encoding, and model scoring occur inside the pipeline. The function chooses class-1 scores, sorts by score descending then customerID ascending, and selects `floor(0.10 * N)` accounts. Batches smaller than ten therefore yield an empty selected list.

The function blocks missing columns, empty inputs, and missing/blank/duplicate customer IDs. Additional numeric/category validation described in the monitoring plan is not implemented. Numeric coercion can turn malformed TotalCharges text into an imputed value, so this is a demonstration workflow rather than a production intake system.

## Fixed evaluation policy

Validation selects the model using precision at the 10% limit, then average precision and ROC-AUC for model ties. Customer-score ties use customerID ascending. The selected forest retains the original training-only fit. Historical success means precision at least 10 percentage points above the cohort churn rate. Scores are ranking scores; calibration was not assessed. Saved/reloaded scores were compared with `rtol=1e-9`, `atol=1e-12`.

## Publication checklist

- Publish this entire `01_CRISP_DM` folder in the assignment repository.
- Add the real repository/Colab links after publication.
- Write and publish the required Medium article and add its URL here.
- Record and publish the code/artifact walkthrough and add its YouTube URL here.
- Keep SEMMA and KDD in their own separate directories, using different datasets.
- Link all project deliverables from the final submitted website URL.

Only the local CRISP-DM report/package is assembled here. No repository, article, video, or assignment submission has been published by this packaging step. Monitoring remains a documented future plan.
