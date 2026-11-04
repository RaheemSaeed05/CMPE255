# Predicting Mathematics Grades with SEMMA

## Summary

This historical demonstration predicts final mathematics grade G3 from earlier grades G1 and G2. Linear Regression was selected using validation MAE and retained its training-only fit. On 79 test records, its MAE was 1.3886 grade points versus 3.8608 for a training-median baseline: a reduction of 2.4722 points, or 64.03%. The predefined historical success criterion was met. This does not demonstrate improved student outcomes or future-cohort accuracy.

## 1. Sample

The source is [Student Alcohol Consumption on Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption), using only student-mat.csv. Despite the dataset title, this project predicts grades, not alcohol consumption. The file contains 395 mathematics records and 33 columns. The exact download date was not recorded in the notebook.

All records were retained because the dataset is small enough to process in full. G1 and G2 were selected as two prior-grade predictors; G3 was separated as the numeric target. Other fields were excluded to focus on prior academic performance. The intended prediction point is after the second grading period. Availability of finalized G1 and G2 before G3 is an assumption, not an independently verified operational fact.

Two random splits with random_state=42 produced 237 training, 79 validation, and 79 test rows (60/20/20). All original indices were accounted for without overlap. No stratification was used. Training supports exploration and fitting, validation selects the predictive model, and test provides the final held-out assessment. The available mathematics records are not assumed to represent all students, schools, or future cohorts.

Initial checks found no pandas-recognized nulls, blank text values, or duplicate full rows. Observed full-file ranges were G1 3–19, G2 0–19, and G3 0–20. Index separation verifies distinct file records; the file has no dedicated student identifier used to independently verify person-level uniqueness.

MAE in grade points is the primary metric. The baseline predicts the training median. Success was defined before testing as test MAE strictly below this baseline; no claim of educationally meaningful or statistically significant improvement was prespecified.

## 2. Explore

The exploration used the 237 training records only.

All three grades have a median of 11 and an interquartile range
from 9 to 13. Mean grades are 11.04 for G1, 10.76 for G2, and
10.49 for G3. G3 has the largest standard deviation, at 4.51.

No training grades fall outside 0–20. There are no zero values
in G1, 10 in G2, and 23 in G3. The histograms show a separate
concentration at zero for G2 and G3. These values will be
retained because the checks do not establish that they are
errors or missing-value codes. Their underlying reason remains
unverified.

Pearson correlations with G3 are 0.774 for G1 and 0.908 for G2.
G1 and G2 are also correlated with each other at 0.808.
G2 therefore has the stronger individual linear association
with the target in this training sample. These correlations
do not establish causation or held-out predictive accuracy.

Both planned predictors will be retained for modeling.
Validation will assess their combined predictive usefulness.
No records or values were changed during exploration.

The notebook includes the three grade histograms and the two predictor-versus-target scatterplots. No validation or test relationships were used in exploration.

## 3. Modify

G1 and G2 were selected in that fixed order, with G3 stored separately. Numeric conversion used errors='raise', then floating-point formatting. Finite-value and 0–20 bounds checks covered all selected grades. Assertions verified predictor order, target exclusion, original index alignment, and unchanged numeric values. These structural checks did not evaluate test performance.

All records and zero grades were retained. No imputation or categorical encoding was needed. Both predictors share a grade scale, and their original units were retained for ordinary Linear Regression and Random Forest. No scaling, binning, or engineered features were introduced. No preparation statistics were learned from any partition.

Preparation preserved 237 training, 79 validation, and 79 test
records. Each predictor matrix contains G1 and G2 in that order,
with G3 stored separately as the target.

All selected values are numeric, finite, and within 0–20.
No missing values were found. Row indices remain aligned,
and checks confirmed that conversion to floating-point format
did not change the grade values.

No records were removed, zeros were retained, and no learned
transformations were fitted. Model training and performance
evaluation remain pending.

## 4. Model

Linear Regression models a linear relationship between earlier and final grades; Random Forest allows nonlinear relationships. The baseline used DummyRegressor(strategy='median'), fitted on training data. Its prediction was 11. LinearRegression used fit_intercept=True, copy_X=True, positive=False, and n_jobs=None. Random Forest used 100 trees, max_depth=4, min_samples_leaf=5, random_state=42, and n_jobs=-1. The depth and leaf limits restrict complexity. Full parameters and scikit-learn 1.6.1 are retained in the notebook.

The lowest validation MAE among the two predictive models determined selection, with an exact tie favoring Linear Regression. The baseline provided a comparison rather than entering that two-model selection rule. Predictions were continuous and neither rounded nor clipped. Settings were documented before validation; no tuning or cross-validation was performed.

| Estimator | Validation MAE | Predictions outside 0–20 |
|---|---:|---:|
| Median baseline | 3.3165 | 0 |
| Linear Regression | 1.0949 | 1 |
| Random Forest | 1.3107 | 0 |

Linear Regression was selected and retained its original training-only fit. The code checked prediction length, index alignment, finiteness, baseline agreement with the training median, and manual MAE agreement. The test set was not scored during selection.

## 5. Assess

| Estimator | Test records | Test MAE | Predictions outside 0–20 |
|---|---:|---:|---:|
| Linear Regression | 79 | 1.3886 | 2 |
| Median baseline | 79 | 3.8608 | 0 |

Both estimators retained their training fits. MAE equals the mean absolute difference between predicted and recorded G3. Full-precision computation yielded a 2.4722-grade-point reduction and 64.03% relative reduction. The selected model's MAE rose from 1.0949 on validation to 1.3886 on test. A single split cannot establish the cause or significance of that difference and does not alone prove overfitting.

Two model predictions were outside the grade scale. They were retained under the fixed evaluation policy and subsequently flagged in the scoring export. For example, source row 153 with G1=5 and G2=0 received approximately -1.0171. This is a model output, not a valid recorded grade. The model was not modified after testing.

The historical target was met, but the small test sample, limited population coverage, assumed predictor availability, unverified meaning of zero grades, and unrestricted linear outputs limit operational conclusions. Future-cohort accuracy and educational benefits remain unverified.

## Delivery demonstration

The notebook saved and reloaded artifacts/grade_model.joblib and scored 79 historical test inputs without G3. Predictions reproduced within rtol=1e-9 and atol=1e-12. The scoring function validates required columns, a nonempty batch, unique input indices, numeric finite values, and input bounds. It selects G1/G2 in fixed order, preserves row indices, and flags predictions outside 0–20. The saved estimator supplies prediction; validation and formatting belong to score_students, not the estimator itself.

The same function is supplied unchanged in score_students.py for reproduction. Source row indices reference records in the original CSV, not student identities. All three CSVs were reloaded and matched notebook DataFrames; schema and environment JSON round-trip checks passed. This is a local scoring demonstration, not a hosted service. SEMMA was implemented in Python/Colab; SAS was not used.

## Critique and revision evidence

Each of the five phases has two supplied reviewer responses and a revision/adoption record in the packaged notebook. Sample Round 1 requested two clarifications: predictor availability and representativeness. Both were added. Later reviews identified no essential changes, and adoption notes accurately state that no extra analysis was performed. The missing Sample Round 1 response was restored from the supplied conversation during packaging, clearly labeled; no review was invented. The remaining notebook code and recorded outputs are unchanged.

## Artifacts, reproduction, and status

README.md maps every artifact to its purpose and provides separate full-experiment and saved-model instructions. environment.json records Python 3.13.18, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.4.6, and joblib 1.6.0. These recorded versions are not a freshly tested installation specification.

During packaging, the 79 exported source indices were checked for uniqueness and joined to the supplied source CSV. G1/G2 agreement, score finiteness, both out-of-range flags, model test MAE, and the median-11 baseline test MAE were independently checked. The extracted scoring function was syntax-checked. The notebook contains no saved error outputs. Model training and joblib reload were not independently rerun during packaging; their execution evidence remains in the notebook.

Completed: five methodology phases, critique/revision history, executed notebook, saved-model demonstration, artifacts, this report, and a local GitHub-ready folder. Pending: GitHub publication, Medium article, YouTube walkthrough, and final submission URL. KDD is a separate remaining project.

## References

- Dataset: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
- Assignment-provided SEMMA example: https://support.sas.com/resources/papers/proceedings13/069-2013.pdf
- Assignment-provided SEMMA overview: https://sceweb.uhcl.edu/boetticher/ml_datamining/sas-semma.pdf

References identify the assignment resources; numerical results come from the executed notebook and exports. No new external source verification was performed during packaging.
