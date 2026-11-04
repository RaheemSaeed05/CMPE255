"""Scoring function extracted unchanged from the executed notebook."""
import numpy as np
import pandas as pd

def score_students(accounts, model):
    required = ["G1", "G2"]

    if not set(required).issubset(accounts.columns):
        raise ValueError("Input must contain G1 and G2.")
    if accounts.empty:
        raise ValueError("Input must contain at least one record.")
    if not accounts.index.is_unique:
        raise ValueError("Input row indices must be unique.")

    predictors = accounts[required].apply(
        pd.to_numeric, errors="raise"
    ).astype(float)

    if not np.isfinite(predictors.to_numpy()).all():
        raise ValueError("Grades must be finite and nonmissing.")
    if not predictors.ge(0).all().all() or not predictors.le(20).all().all():
        raise ValueError("Input grades must be within 0–20.")

    predictions = model.predict(predictors)

    if not np.isfinite(predictions).all():
        raise ValueError("Model produced nonfinite predictions.")

    result = predictors.copy()
    result["Predicted G3"] = predictions
    result["Outside 0–20"] = (predictions < 0) | (predictions > 20)
    return result
