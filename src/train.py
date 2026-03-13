"""Model training and cross-validation utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold


def cross_validate_models(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: list[str] | None = None,
) -> pd.DataFrame:
    """Run stratified k-fold cross-validation for multiple models.

    Parameters
    ----------
    models : dict
        Mapping of model name -> sklearn estimator (or Pipeline).
    X : array-like
        Feature matrix (already preprocessed or wrapped in a Pipeline).
    y : array-like
        Target vector.
    cv : int
        Number of folds.
    scoring : list of str, optional
        Metrics to evaluate. Defaults to accuracy, f1, roc_auc.

    Returns
    -------
    pd.DataFrame
        Summary with mean ± std for each metric per model.
    """
    if scoring is None:
        scoring = ["accuracy", "f1", "roc_auc"]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        cv_results = cross_validate(
            model, X, y, cv=skf, scoring=scoring, return_train_score=False
        )
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = cv_results[key].mean()
            row[f"{metric}_std"] = cv_results[key].std()
        results.append(row)

    return pd.DataFrame(results)
