# AutoML Experiment — CMPE 255 Assignment 1 Part 2

This directory contains a reproduction-ready, CRISP-DM-structured tabular classification experiment inspired by the instructor's AutoGluon data science example.

## Dataset and Execution Disclaimer

The original dataset is **not included** because of dataset-access constraints. The complete AutoML workflow is implemented, but AutoGluon training was **not executed**. Therefore, this submission reports **no leaderboard values, numerical model results, model rankings, feature-importance values, accuracy, or F1 scores**. Nothing has been fabricated.

When an authorized dataset is supplied, notebook outputs are calculated dynamically. Existing notebook cells have no saved execution outputs.

## Files

- `automl_experiment.ipynb` — CRISP-DM analysis, guarded data loading, manual baseline, and opt-in AutoGluon workflow.
- `requirements.txt` — Python dependencies.
- `prompt.txt` — exact assignment prompt.
- `images/` — reserved for generated figures or submission screenshots (`.gitkeep` retains the directory).

## Reproduce

1. Create and activate a Python 3.10 or 3.11 virtual environment.
2. From this directory, install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Put the authorized CSV at `data/dataset.csv`, or define its path:

   ```bash
   export AUTOML_DATASET_PATH=/absolute/path/to/dataset.csv
   export TARGET_COLUMN=the_target_column
   ```

4. Open the notebook:

   ```bash
   jupyter lab automl_experiment.ipynb
   ```

5. Run cells from the top. Review schema, target semantics, missingness, leakage columns, split strategy, and the compute budget.
6. To perform AutoML training, set `RUN_AUTOML = True`. Adjust `AUTOML_TIME_LIMIT` and the AutoGluon preset based on available resources.

If the CSV is missing, its target is misconfigured, or AutoGluon is unavailable, guarded cells print corrective instructions instead of crashing.

## Implemented workflow

The notebook includes business/data understanding, EDA, missing-value analysis, conservative cleaning, feature preparation, train/test splitting, leakage controls, a Scikit-learn logistic-regression baseline, AutoGluon `TabularPredictor` training, leaderboard and feature-importance workflows, ensembling/stacking discussion, evaluation and comparison logic, benefits/risks/costs, deployment, limitations, future work, and a CRISP-DM conclusion.

## Reproducibility notes

- Fixed split/model seed: `42`.
- Learned manual preprocessing is fitted inside a Scikit-learn pipeline on training data only.
- AutoGluon sees test data only after fitting.
- AutoML training is explicitly opt-in and time-bounded.
- For temporal or grouped observations, replace the random split with an appropriate time/group-aware validation design before training.
