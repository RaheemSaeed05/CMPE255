# Anomaly Detection — CMPE 255 Assignment 1, Part 2

This directory contains a reproduction-ready CRISP-DM experiment inspired by Kaggle's **Credit Card Fraud Detection** dataset. It compares three unsupervised detectors while reserving the binary fraud label strictly for evaluation.

> # Dataset and Execution Disclaimer
>
> **The original dataset is not included and was not executed.** Dataset access and acceptance of its terms are the responsibility of the user. The methodology, validation, visualization, and evaluation pipeline are fully implemented, but numerical anomaly-detection results—including anomaly counts, contamination outcomes, precision, recall, F1 scores, confusion matrices, and model rankings—are intentionally omitted. They are generated only after a valid dataset is supplied and the notebook is executed. No numerical result has been fabricated.

## Artifacts

- `anomaly_detection.ipynb` — documented CRISP-DM workflow with safe dataset detection.
- `requirements.txt` — bounded direct Python dependencies.
- `prompt.txt` — exact assignment prompt used to create this project.
- `images/` — destination for plots generated during genuine execution; intentionally empty except for `.gitkeep` before execution.

## Dataset setup

1. Download `creditcard.csv` from Kaggle's Credit Card Fraud Detection dataset after accepting the dataset terms.
2. Put the file in this directory **without committing it**, or set an environment variable:
   ```bash
   export ANOMALY_DATA_PATH=/absolute/path/to/creditcard.csv
   ```
3. The expected columns are `Time`, `Amount`, `V1` through `V28`, and binary label `Class`.

If the file is absent or has the wrong schema, the notebook prints useful setup guidance and safely skips every data-dependent stage instead of crashing.

## Reproduce

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook anomaly_detection.ipynb
```

In Jupyter, choose **Restart Kernel and Run All Cells**. Runtime and memory depend on the supplied file; One-Class SVM and LOF may be relatively expensive.

## Implemented CRISP-DM workflow

1. Business objectives, stakeholders, costs, and success criteria.
2. Schema validation, data understanding, EDA, missing-value analysis, and duplicate analysis.
3. Numeric validation, duplicate handling, median imputation, leakage-free feature preparation, log amount transformation, and robust scaling.
4. Non-destructive IQR outlier diagnostics.
5. Isolation Forest, Local Outlier Factor, and RBF One-Class SVM.
6. Explicit contamination semantics and operational threshold guidance.
7. When labels exist: precision, recall, F1, anomaly counts, and confusion matrices.
8. PCA anomaly visualization, ranked-case interpretation, technique trade-offs, limitations, future improvements, and CRISP-DM conclusion.

## Reproducibility and responsible interpretation

The notebook fixes random state where supported and leaves all code-cell outputs empty in version control. `Class` is never included in the training feature matrix. A flagged anomaly is an investigation candidate, not proof of fraud. Threshold selection must account for review capacity, false-positive cost, false-negative cost, temporal drift, and label quality.
