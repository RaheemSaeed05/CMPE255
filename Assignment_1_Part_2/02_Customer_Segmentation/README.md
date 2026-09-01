# Customer Segmentation — CMPE 255 Assignment 1, Part 2

This project is a reproducible, master's-level customer segmentation experiment inspired by the widely studied Kaggle **Mall Customers** dataset. The notebook follows CRISP-DM from business framing through data understanding, preparation, modeling, evaluation, and deployment considerations.

## Dataset and Execution Disclaimer

> **The original Kaggle dataset is not included and was not executed because of dataset-access constraints.** This project reproduces the complete CRISP-DM methodology and clustering workflow, but numerical results are intentionally not reported. No silhouette score, cluster size, optimal value of K, PCA result, or data-dependent business conclusion has been invented. Those outputs are calculated only after the user supplies an authorized dataset and runs the notebook.

The notebook checks for the expected CSV and exits analytical cells safely with clear guidance when it is absent.

## Project Contents

- `customer_segmentation.ipynb` — end-to-end, documented CRISP-DM experiment.
- `requirements.txt` — Python dependencies with reproducible major-version bounds.
- `prompt.txt` — the assignment prompt used to produce this project.
- `images/` — output location for model-selection and PCA figures generated at runtime.

## Expected Dataset

Obtain the dataset from its authorized Kaggle source and accept any applicable terms. Save the unmodified file beside the notebook as:

```text
Mall_Customers.csv
```

The commonly used version includes fields such as customer ID, gender, age, annual income, and spending score. The notebook discovers the actual schema at runtime, excludes identifier-like columns from clustering, and requires at least two usable numeric features.

Do **not** commit a downloaded dataset if its license or course policy prohibits redistribution.

## Reproduce the Experiment

From this directory, create an isolated environment and launch Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook customer_segmentation.ipynb
```

Then use **Kernel → Restart & Run All**. With no CSV, the notebook completes without a file-read crash and explains how to proceed. With the authorized CSV, it computes diagnostics and writes generated figures to `images/`.

## Methodology

The notebook implements:

1. business and ethical framing;
2. schema inspection, descriptive EDA, missing-value and duplicate analysis;
3. auditable cleaning, IQR-based outlier diagnostics, and explicit feature selection;
4. `StandardScaler` preprocessing;
5. deterministic K-Means candidates across multiple feasible K values;
6. elbow and silhouette diagnostics plus cluster-balance comparison;
7. runtime model selection based on the observed candidate silhouette scores;
8. two-component PCA for visualization only;
9. runtime cluster profiling and relative centroid-based interpretation; and
10. recommendations, limitations, future improvements, and a CRISP-DM conclusion.

## Reproducibility and Responsible Use

A fixed random seed and explicit K-Means initialization count make repeated runs comparable. PCA is not used to fit the clusters. Generated segments remain exploratory: deployment requires stability analysis, privacy and fairness review, domain validation, monitoring, and controlled experiments demonstrating incremental business value.
