# CRISP-DM: Prioritizing a simulated telecom outreach list

## Executive summary

This project applies all six CRISP-DM phases to IBM sample telecom churn data distributed through Kaggle. It asks which historical accounts a capacity-limited retention team should prioritize. The monthly schedule and 10% contact capacity are assumptions. The goal is a historical ranking demonstration, not proof that outreach prevents churn.

Two models were compared using validation precision at the contact limit. The selected Random Forest retained its training-only fit. On the final test sample, its top 140 of 1,409 accounts contained 102 historical churners: 72.86% precision, 27.27% recall, and a 46.31-percentage-point gain over the 26.54% random-selection baseline. This exceeds the proposed minimum gain of 10 percentage points. The saved pipeline reproduced scores in the notebook's reload check and generated a simulated outreach list without requiring churn labels.

This report consolidates the notebook's reviewed revisions. Drafts, two critique rounds for each phase, revision responses, code, and recorded outputs remain in the accompanying notebook. No additional model fitting or tuning was performed during packaging.

## 1. Business Understanding

**1. Business objective.** At the start of each month, the retention team wants to decide which active customers to call first. The goal is to reach customers at risk of leaving before the next monthly call list is created. Measuring whether a call actually prevented a cancellation would require a separate outreach experiment.

**2. Situation and constraints.** I assume the team can call up to 10% of the accounts active at the start of each month. The monthly schedule and call limit are project assumptions. The CSV contains account details and a historical churn label, but it has no dated account snapshots to connect each prediction to a departure in the following month. I will use held-out accounts to simulate one outreach cycle. This lets me evaluate how well the ranking identifies historical churners. Its accuracy for the intended one-month window remains unverified.

**3. Data mining goal and success criteria.** I will compare the churn rate among the top-ranked 10% with the churn rate across all held-out accounts, which is the expected rate under random selection. My proposed success target is a difference of at least 10 percentage points. For example, if 26% of the held-out accounts churned, at least 36% of the selected accounts would need to have churned to meet the target.

**4. Project plan.** I will check data quality, prepare the features, and compare models on validation data. I will use the chosen model to score the held-out accounts, sort them by risk, and select the top 10%, rounding down to stay within capacity. The sample call list will include customer ID, risk score, and rank. After creating the list, I will use the held-out churn labels to calculate whether it meets the success target.

## 2. Data Understanding

#### 1. Collect initial data

I downloaded WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle and loaded it into Google Colab using pandas.

Download date — not recorded. The dataset was uploaded to the project on November 2, 2026. This follows the final revision and supersedes the earlier draft date.

Source — https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The Kaggle description attributes the dataset to IBM sample data and describes each row as one customer. Churn indicates whether the customer left within the last month. The source supports using the sample to explore customer retention.

The CSV contains no dated account snapshots or cancellation dates. The exact calendar period and whether these accounts represent a current telecom customer population remain unverified. Historical evaluation therefore cannot establish performance for the following month.

#### 2. Describe data

The dataset contains 7,043 customer accounts and 21 columns. There are 1,869 Yes churn labels and 5,174 No labels, giving an overall churn rate of approximately 26.54%.

customerID is the identifier, Churn is the target, and tenure, MonthlyCharges, and TotalCharges are numeric measurements. SeniorCitizen is a binary category stored as 0 or 1. The remaining predictors describe demographics, services, contracts, and payment arrangements.

Observed categories include:
- Contract — Month-to-month, One year, Two year.
- InternetService — DSL, Fiber optic, No.
- MultipleLines — Yes, No, No phone service.
- Internet add-on services — Yes, No, No internet service.
- PaymentMethod — Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check.
- gender — Female, Male.
- Partner, Dependents, PhoneService, PaperlessBilling, and Churn — Yes, No.

“No internet service” and “No phone service” describe the absence of the underlying service. They are meaningful categories and will be retained separately from “No.”

Tenure is interpreted as months with the company. MonthlyCharges and TotalCharges represent monthly and accumulated charges. The currency has not been verified from the source documentation consulted.

customerID will identify accounts in the final outreach list and will be excluded from model inputs.

#### 3. Explore data

I used two stratified splits with random_state=42 to create the following partitions.

| Partition | Accounts | Churn Yes | Churn percentage |
|---|---:|---:|---:|
| Training | 4,225 | 1,121 | 26.53% |
| Validation | 1,409 | 374 | 26.54% |
| Test | 1,409 | 374 | 26.54% |

All accounts are included, and no customer IDs overlap between partitions. Training data supports exploration and fitting, validation data supports model selection, and test data is reserved for final model evaluation.

In the training set, churn rates are 43.06% for month-to-month contracts, 11.05% for one-year contracts, and 2.93% for two-year contracts.

Customers who churned have a median tenure of 10 months and median monthly charges of 80.00. Customers who stayed have a median tenure of 38 months and median monthly charges of 64.40.

The charts show associations between these variables and churn, with overlapping distributions. Their combined predictive value will be assessed during modeling.

#### 4. Verify data quality

Completed checks found:
- Zero pandas-recognized nulls in the original data.
- Eleven whitespace-only TotalCharges entries and zero blank strings in other text columns.
- Eleven missing values after numeric conversion of TotalCharges, corresponding to the blank entries.
- Zero duplicate rows and zero duplicate customer IDs.
- Zero surrounding-space issues or category labels merged by normalizing capitalization and spaces in the checked categorical fields.
- Zero negative values in tenure, MonthlyCharges, or numeric TotalCharges.
- Only 0 and 1 in SeniorCitizen.
- Zero non-integer tenure values.

Observed numeric ranges are:
- tenure — 0 to 72.
- MonthlyCharges — 18.25 to 118.75.
- Nonmissing TotalCharges — 18.80 to 8,684.80.

All 11 accounts with blank TotalCharges have zero tenure. Their status as new accounts is a possible explanation, but the reason for the blanks remains unconfirmed.

These checks establish basic consistency within the file. They do not independently verify every recorded value against customer records. Diagnostic numeric conversion supported these checks; modeling-input conversion and treatment are described in Data Preparation. TotalCharges originally loaded as object/text even though it represents a numeric measurement.

Structural and quality checks covered the full dataset. Detailed relationships with churn were explored using training data.

## 3. Data Preparation

#### 1. Select data

I retained all accounts in the existing partitions and selected the 19 account, service, and demographic fields as initial candidate predictors. These fields may help explain differences in churn risk. Their availability before churn cannot be fully verified because the dataset lacks dated snapshots.

customerID is stored separately for the outreach list, and Churn is the target. Checks confirmed that neither field appears in the preprocessing inputs or generated feature names.

#### 2. Clean data

TotalCharges was converted from text to numeric. This produced seven missing values in training, one in validation, and three in test.

I retained training-median imputation as a provisional baseline. Inspection confirmed that all seven affected training accounts have zero tenure and received an imputed TotalCharges value of 1,373.625 before scaling. This amount may poorly represent their actual charges. The missing amounts remain unknown, and zero tenure alone does not establish a correct replacement.

The numeric imputer uses training medians of 29 for tenure, 70.55 for MonthlyCharges, and 1,373.625 for TotalCharges. Validation and test data use these same learned replacements.

The most-frequent categorical imputer is configured as a fallback. No categorical missing values were observed.

#### 3. Construct data

Churn was mapped from Yes to 1 and No to 0. Row-level comparisons verified the mapping, and class counts were preserved.

| Partition | Original Yes / Encoded 1 | Original No / Encoded 0 |
|---|---:|---:|
| Training | 1,121 | 3,104 |
| Validation | 374 | 1,035 |
| Test | 374 | 1,035 |

SeniorCitizen was converted to a categorical representation. The initial models use the original predictors to establish a reproducible baseline. Additional derived features would require a documented reason and validation.

#### 4. Integrate data

The project uses one CSV, so no joins were required. The prepared matrices were wrapped in DataFrames using the original partition indices and the fitted transformer's feature names.

Assertions confirmed matching indices and lengths for prepared data, targets, and customer identifiers. A spot check also confirmed that the first row in each prepared matrix matches transforming that account individually. These labeled DataFrames preserve the connection needed to associate model scores with customer IDs.

#### 5. Format data

The numeric pipeline applies median imputation followed by standardization. The categorical pipeline applies most-frequent imputation followed by one-hot encoding. Service-absence categories remain separate.

The same fitted preprocessor generated all three matrices, each with 46 features in the same order. All prepared values are finite. Output feature names were recorded.

Checks confirmed that scaler means and variances match the imputed training data and that encoder categories match the categorical training data.

Unseen categories are handled using all-zero indicators for the affected field. This permits transformation, but the model has not learned a separate effect for that unseen category.

#### Cross-validation handling

If cross-validation is used during modeling, the preprocessing and estimator will be placed together in a Pipeline. Each fold will receive predictors before learned imputation, scaling, and encoding, allowing those steps to be fitted separately on that fold's training rows. The already-transformed matrices will not be used as cross-validation inputs.

## 4. Modeling

#### 1. Select modeling techniques

I compared Logistic Regression and Random Forest using scikit-learn 1.6.1. The notebook records both fitted estimators' full parameter configurations alongside the training code.

Logistic Regression uses L2 regularization with C=1.0, the lbfgs solver, and max_iter=2000. Random Forest uses 300 trees, max_depth=8, min_samples_leaf=5, and random_state=42. The depth and leaf-size restrictions were chosen to limit complexity.

#### 2. Generate test design

Models were fitted on 4,225 training accounts and compared on 1,409 validation accounts. The primary selection metric was precision among the top 140 accounts, respecting the 10% outreach limit after rounding down.

Customer scores are sorted descending, with customerID ascending to resolve ties. Model ties are resolved using average precision, then ROC-AUC.

Before examining test performance, I fixed the final procedure to evaluate the selected Random Forest using its existing training-only fit and preprocessing. The final outreach list will use the same ranking rule and 10% limit.

The success criterion remains precision at least 10 percentage points above the test-set churn rate.

#### 3. Build models

Each model was fitted in a Pipeline containing its own cloned preprocessor. Preprocessing settings were learned from training data.

Verification confirmed that each model produced 1,409 finite validation scores between 0 and 1. Predictor indices, customer IDs, and target labels matched the original validation accounts. Recomputed rankings matched the stored rankings, and each selected list contained exactly 140 unique accounts.

A repeated Logistic Regression fit with the same settings produced zero convergence warnings and matched the original validation scores. This documents the repeat run. Warnings from the original fit were not captured.

#### 4. Assess models

Random Forest selected 111 churners among 140 accounts, giving 79.29% precision and 29.68% recall. Logistic Regression selected 106 churners, giving 75.71% precision and 28.34% recall.

Expected random-selection precision was 26.54%. Random Forest improved on this by 52.74 percentage points, and Logistic Regression improved by 49.17 percentage points.

Logistic Regression had slightly higher average precision and ROC-AUC, at 0.6777 and 0.8599. Random Forest achieved 0.6746 and 0.8580. These metrics were recomputed from the full validation labels and scores and matched the recorded results.

Random Forest was selected because it identified five additional churners at the prespecified contact limit. Its advantage is based on one validation split, so its stability across samples remains unmeasured.

The outputs are used as ranking scores. Probability calibration and actual retention benefits remain unverified. Final test performance is reported in the next phase. Cross-validation and hyperparameter tuning have not been performed.

### Recorded validation comparison

| Model | Selected churners / accounts | Precision | Recall | Gain (pp) | Average precision | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 111 / 140 | 79.29% | 29.68% | 52.74 | 0.674553 | 0.858012 |
| Logistic Regression | 106 / 140 | 75.71% | 28.34% | 49.17 | 0.677735 | 0.859929 |


## 5. Evaluation

#### 1. Evaluate results

The selected Random Forest identified 102 historical churners among the 140 highest-ranked test accounts. Precision was 72.86%, and recall was 27.27%.

The overall test churn rate was 26.54%, giving a required precision of approximately 36.54% under the proposed 10-percentage-point improvement target. The model exceeded the baseline by 46.31 percentage points and met the historical ranking target on this test sample.

Random selection would identify approximately 37.16 churners on average among 140 accounts. The model identified approximately 64.84 more than this expectation.

The 38 selected non-churners occupied places in the simulated outreach list. The 272 unselected churners represent missed coverage under the contact limit. These counts describe historical selection outcomes. Whether any selected customer would respond to outreach remains unknown.

Reported test average precision was 0.6326, and ROC-AUC was 0.8315. These were computed from the full test labels and scores.

Precision at the contact limit decreased from 79.29% on validation to 72.86% on test, a decrease of approximately 6.43 percentage points. One split cannot establish the cause or statistical significance of this difference. Sampling variation and optimism from selecting the validation winner are possible explanations. The decrease alone is insufficient evidence of overfitting.

#### 2. Review the process

Model selection used validation results. The selected model's training-only fit, preprocessing, ranking rule, and contact limit were fixed before final testing.

Executed checks confirmed account-label alignment, finite scores within bounds, and selection of exactly 140 unique accounts. The notebook retains the code and outputs for inspection.

Two methodological limitations remain. Missing observation dates prevent confirmation that every predictor was recorded before churn. The realism of median-imputed TotalCharges for zero-tenure accounts also remains uncertain. These limitations restrict conclusions about future operational performance.

The current test result will remain documented. Further model development would require fresh evaluation data for another independent final assessment.

#### 3. Determine next steps

The evaluation decision was to proceed to a deployment demonstration using the selected fitted pipeline.

The following acceptance checks were specified before the deployment demonstration:
- Saving and reloading the pipeline reproduces the original scores within numerical tolerance.
- The scoring workflow accepts customerID and the required account predictors without requiring Churn.
- Customer IDs remain correctly associated with scores.
- The workflow selects floor(0.10 × the number of simulated input accounts).
- Ranking follows score descending, with customerID ascending to resolve ties.

These checks subsequently passed in the deployment demonstration, with score comparison using rtol=1e-9 and atol=1e-12.

Historical test accounts served as a simulated input group. Their current eligibility for contact is unknown. Actual retention gains, financial benefits, probability calibration, and future-month predictive performance remain unverified.

## 6. Deployment

#### 1. Plan deployment

The scoring demonstration accepts customerID and the 19 original predictors. The input schema and environment versions are recorded in input_schema.json and environment.json.

The notebook's score_accounts function validates required columns and customer IDs, selects the predictor columns, converts TotalCharges to numeric, and converts SeniorCitizen to strings. It passes these predictors to the saved pipeline.

Inside the saved pipeline, numeric values are imputed and scaled, categorical values are imputed and encoded, and Random Forest generates scores. All learned preprocessing settings come from the training fit.

The notebook function selects the score corresponding to class 1, associates scores with customer IDs, sorts by score descending and customerID ascending, and selects floor(0.10 × simulated input accounts).

Reproduction requires the saved pipeline and the notebook's scoring function, including its predictor-column list and contact-fraction setting.

The demonstration reproduced scores after reload within rtol=1e-9 and atol=1e-12. It scored 1,409 historical accounts without Churn and selected 140. Account alignment and ranking checks passed.

The two ranking files contain Rank, customerID, and Risk score. The validation and test results files contain model metrics. A verification cell reloads all four CSVs and compares their contents with the corresponding notebook data.

#### 2. Plan monitoring and maintenance

The following procedures are proposed for future operational use.

Blocking conditions include missing required columns, empty batches, missing or duplicate customer IDs, invalid numeric text other than supported blank TotalCharges values, and invalid SeniorCitizen values. The data analyst would investigate these failures, and the designated project owner would authorize resuming scoring after correction and successful checks.

Supported missing values would be handled through imputation. Increased missingness, unseen categories, and changes in score distributions would trigger review according to thresholds agreed by the analyst and project owner before operational use. These thresholds have not yet been established.

Outcome monitoring would require complete churn labels for the entire scored cohort over a defined observation period. This enables calculation of recall, the cohort churn baseline, and precision gain. Any outreach would be recorded because observed outcomes could reflect its effects.

A precision gain below the proposed 10-percentage-point target would trigger investigation. A replacement model would require fresh evaluation, documentation, and project-owner approval before adoption.

Monitoring remains planned. The demonstration implements the documented column, batch-size, and customer-ID checks. Additional operational validation and review procedures remain pending.

#### 3. Produce the final report

The accompanying artifacts connect each output to its purpose and evidence:
- Executed notebook — six phases, critique revisions, code, and verification outputs.
- churn_pipeline.joblib — fitted preprocessing and selected Random Forest.
- input_schema.json — required scoring inputs and ranking rules.
- environment.json — software versions used.
- simulated_account_ranking.csv — ranking of all demonstration accounts.
- simulated_outreach_list.csv — selected 140-account list.
- validation_results.csv — evidence supporting model selection.
- test_results.csv — final historical evaluation results.

README.md provides separate instructions for reproducing the full experiment and scoring with the saved pipeline.

The comparison below documents validation and test performance under the fixed selection procedure. Test precision was 72.86%, recall was 27.27%, and gain over random expected precision was 46.31 percentage points. Reported computed average precision and ROC-AUC were 0.6326 and 0.8315.

These results support the historical scoring demonstration. Retention and financial benefits remain unmeasured.

This report and the local GitHub-ready package are assembled. Publishing the repository, the Medium article, and the YouTube walkthrough remain pending.

#### 4. Review the project

Blank TotalCharges entries showed why pandas null checks need additional whitespace and numeric-conversion checks. Future ingestion should include these checks.

The imputation review showed that a correctly implemented replacement can still have uncertain business meaning. Future work should obtain billing definitions or verified values before changing the rule.

The lack of dated snapshots limits claims about future churn. Future evaluation should use predictors captured before a defined outcome period.

Model comparison showed why selection metrics should be fixed in advance. Random Forest performed better at the outreach limit, while Logistic Regression performed slightly better on broader validation ranking metrics.

Saving and reloading the model showed that reproducibility also depends on input formatting, software versions, and customer-ID alignment. These dependencies are now documented.

## Reproducibility and delivery status

The environment export records Python 3.13.18, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.4.6, SciPy 1.16.3, and joblib 1.6.0. These are recorded notebook versions, not a claim that a fresh installation was tested during packaging. Full model parameters are retained in the notebook. Requirements are listed separately for convenient environment preparation.

Package checks independently confirmed the two ranking CSVs' sizes, unique identifiers, score bounds, deterministic ordering, sequential ranks, and exact top-140 agreement. Joining exported customer IDs to the supplied source CSV confirmed 374 churners in the simulated cohort and 102 among selected accounts. Count-derived test precision, recall, and percentage-point gain were recalculated. These package checks did not rerun model training, reload the model, or independently recompute average precision and ROC-AUC; the notebook retains the recorded execution evidence for those results. No saved notebook error outputs were found.

| Deliverable | Status |
|---|---|
| Six CRISP-DM phases and phase critique/revision history | Present in executed notebook |
| Fitted pipeline, schema, environment, ranking and metric exports | Included |
| Consolidated report and reproduction README | Included |
| GitHub publication and working Colab link | Pending |
| CRISP-DM Medium article | Pending |
| CRISP-DM YouTube code/artifact walkthrough | Pending |
| SEMMA and KDD projects with different datasets | Not completed in this package |
| Final assignment website URL submission | Pending |

## Sources and evidence

- Dataset source recorded in the notebook: https://www.kaggle.com/datasets/blastchar/telco-customer-churn — IBM sample attribution and historical churn definition are documented in the reviewed Data Understanding phase. The exact download date was not recorded.
- Assignment-provided CRISP-DM reference: https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview
- Assignment-provided course slides: https://docs.google.com/presentation/d/1NozxdUtcAWOBq-VVkrJwy5mK4EHAjR1vhhbF50APS8c/edit
- Primary execution evidence: 01_CRISP_DM_Telco_Churn.ipynb and the artifacts directory. References above identify the supplied resources; no new source verification was performed during packaging.

The notebook contains the actual supplied critic responses. Their text is preserved without inventing new review rounds or claims about the model that authored them.
