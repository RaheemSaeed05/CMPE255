# CMPE 255 Assignment 1, Part 2: Data Science Experiments

This submission presents six reproduction-ready data science experiments spanning supervised learning, unsupervised learning, pattern mining, anomaly detection, automated machine learning, and forecasting. Each project is organized as a self-contained, master's-level study with its own documentation, dependency specification, original prompt, Jupyter notebook, and directory for generated visualizations.

The experiments use the **CRISP-DM** lifecycle to connect a clearly defined analytical objective to data understanding, data preparation, modeling, evaluation, and deployment considerations. Together, they demonstrate how the same disciplined methodology can be adapted to distinct data types, modeling assumptions, and evaluation strategies.

## Experiment Overview

| Experiment | Data Science Type | Main Techniques |
|---|---|---|
| [NYC Taxi Prediction](01_NYC_Taxi_Prediction/) | Regression | Feature Engineering, Linear Regression, Random Forest, Gradient Boosting |
| [Customer Segmentation](02_Customer_Segmentation/) | Unsupervised Learning | StandardScaler, K-Means, PCA, Silhouette Analysis |
| [Association Rule Mining](03_Association_Rule_Mining/) | Pattern Mining | Apriori, Support, Confidence, Lift |
| [Anomaly Detection](04_Anomaly_Detection/) | Anomaly Detection | Isolation Forest, LOF, One-Class SVM |
| [AutoML](05_AutoML/) | Automated Machine Learning | AutoGluon, Ensembling, Model Leaderboard |
| [Time-Series Forecasting](06_Time_Series_Forecasting/) | Forecasting | Naive Baseline, Moving Average, Exponential Smoothing, Holt-Winters |

## Experiments

### 1. NYC Taxi Prediction

Builds a supervised regression pipeline for predicting taxi-trip duration. The workflow emphasizes temporal and geospatial feature engineering, data-quality controls, reproducible train/test preparation, comparison of linear and tree-based estimators, and evaluation with appropriate regression metrics.

### 2. Customer Segmentation

Develops an unsupervised customer-profiling workflow. Features are standardized before K-Means clustering, candidate cluster counts are assessed with silhouette analysis, and PCA supports two-dimensional visualization and interpretation of the resulting customer groups.

### 3. Association Rule Mining

Implements a market-basket pattern-mining pipeline. Transaction data is transformed into basket form, frequent itemsets are identified with Apriori, and candidate relationships are assessed using support, confidence, and lift before business interpretation.

### 4. Anomaly Detection

Compares complementary unsupervised anomaly-detection methods. Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM provide different perspectives on unusual observations, with labels reserved for evaluation when an authorized dataset is supplied.

### 5. AutoML

Demonstrates an automated tabular machine-learning workflow using AutoGluon. The experiment covers guarded data preparation, a manual baseline, automated model training and ensembling, leaderboard-based comparison, feature importance, and responsible deployment considerations.

### 6. Time-Series Forecasting

Constructs a forecasting workflow with chronological validation. Naive and moving-average baselines are compared with exponential-smoothing methods, including Holt-Winters, while diagnostics and error metrics are generated only from a supplied source series.

## Repository Structure

Each experiment contains:

- a `README.md` describing its objective, methodology, data requirements, and reproduction steps;
- a Jupyter notebook implementing the end-to-end analytical workflow;
- a `requirements.txt` dependency specification;
- a `prompt.txt` recording the experiment requirements; and
- an `images/` directory reserved for plots produced during genuine execution.

Consult each project's README before running its notebook because expected dataset names, schemas, environment variables, and setup procedures differ by experiment.

## Dataset and Reproduction Note

Due to dataset-access and file-size constraints, these projects reproduce the instructor's data science methodologies, code structure, CRISP-DM workflows, modeling pipelines, and evaluation procedures. The original source datasets were not executed, so no fabricated numerical results are presented. Dataset-dependent tables, metrics, plots, rankings, discovered patterns, forecasts, and conclusions remain pending until an authorized user supplies the required data and executes the corresponding notebook.

Using reproducible templates without fabricated outputs is preferable to reporting unverified results because it preserves academic integrity and keeps every empirical claim traceable to an actual execution. It also allows another researcher to inspect the assumptions, provide the authorized source data, rerun the analysis, and independently verify the resulting evidence rather than relying on numbers that cannot be reproduced.

## Reproduction Guidance

1. Select an experiment and review its project-level `README.md`.
2. Create an isolated Python environment and install the project's `requirements.txt`.
3. Obtain the source dataset through its authorized provider and place it at the documented path.
4. Open the Jupyter notebook and run all cells in order.
5. Review the generated metrics and visualizations in context before making data-dependent conclusions.

## YouTube Walkthrough

[YouTube Video Link — To Be Added]
