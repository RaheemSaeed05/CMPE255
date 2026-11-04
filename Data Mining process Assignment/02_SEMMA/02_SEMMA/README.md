# SEMMA — Student Grade Prediction

Predict G3 using G1 and G2. The selected Linear Regression achieved **1.3886 test MAE** versus **3.8608** for the training-median baseline, a **64.03% reduction** on 79 historical records. Two predictions fell outside 0–20 and remain flagged, unrounded, and unclipped. Future-cohort accuracy and educational benefits are unverified.

## Files

| File | Purpose |
|---|---|
| `02_SEMMA_Student_Performance.ipynb` | Executed five-phase notebook, plots, checks, and two critique rounds per phase |
| `report.md` | Consolidated methodology, findings, limitations, and delivery status |
| `score_students.py` | Scoring function extracted unchanged from the executed notebook |
| `artifacts/grade_model.joblib` | Selected training-fitted Linear Regression |
| `artifacts/simulated_grade_predictions.csv` | 79 historical predictions, source row references, and range flags |
| `artifacts/validation_results.csv` | Three-estimator validation comparison |
| `artifacts/test_results.csv` | Selected model and baseline final test metrics |
| `artifacts/input_schema.json` | Input requirements and prediction policy |
| `artifacts/environment.json` | Recorded runtime versions |

## Full experiment in Colab

1. Obtain `student-mat.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption). Use the same comma-separated mathematics file, with 395 rows and 33 columns. Raw data is not bundled.
2. Open the notebook in Colab. Compare its runtime with `artifacts/environment.json`; a fresh installation of those exact versions was not tested during packaging.
3. Run cells in order, uploading the CSV when prompted. Read the phase documentation, critiques, and outputs. The fixed split is 237/79/79 with seed 42. Training-only Linear Regression and Random Forest are compared using validation MAE; test assesses the selected model and median baseline.
4. Run the delivery cells to save and reload the selected model, verify score reproduction, and export artifacts. Outputs go to `02_SEMMA/artifacts` relative to the working directory, normally `/content` in Colab. The final download cell assumes this standard Colab location.
5. Save the executed notebook and artifact ZIP. Retain all critique responses and revision notes.

## Saved-model scoring without retraining

In a compatible Python environment, work from the extracted `02_SEMMA` folder. The scoring module needs pandas and NumPy; model loading additionally needs joblib and scikit-learn with its dependencies.

```python
import joblib
import pandas as pd
from score_students import score_students

model = joblib.load('artifacts/grade_model.joblib')
accounts = pd.read_csv('input_grades.csv')
result = score_students(accounts, model)
result.rename_axis('Source row index').reset_index().to_csv(
    'predictions.csv', index=False
)
```

Input must contain G1 and G2, numeric, finite, nonmissing, and within 0–20. G3 is not required. The function checks unique input indices, selects the required predictor order, converts inputs to floats, and flags out-of-range outputs. CSV row indices are record references, not student identifiers. The serialized estimator alone does not include the function's input checks. No scaling, imputation, or encoding is required for this workflow.

## Review history and package checks

Sample Round 1 was absent from the uploaded notebook and was restored from the user's supplied review in the conversation, with an explicit restoration note. Heading spacing was corrected. Existing code and outputs were preserved. For phases with no required corrections, revision notes retain the prior work rather than inventing changes.

Package verification checked CSV/source-row alignment, finite scores, range flags, and test MAE for the selected model and baseline. It did not rerun model fitting or reload the estimator. The original execution evidence is retained in the notebook.

## Submission status

This folder is ready for upload to the assignment repository as `02_SEMMA`. Python/Colab was used; SAS was not used. Repository publication, the Medium article, and the YouTube walkthrough remain pending. Add the actual links below when available:

GitHub:

Colab:

Medium:

YouTube:

Keep CRISP-DM and KDD in separate subdirectories and link all required deliverables from the final submitted URL.
