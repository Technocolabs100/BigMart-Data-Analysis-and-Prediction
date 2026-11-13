"""Cross-validated comparison of the models, hold-out predictions and feature importance.

Every product and every store of `Test.csv` also appears in `Train.csv`, so a shuffled
K-fold split is the evaluation that matches the real task. RMSE is the metric of the
original competition; MAE and R2 are reported next to it.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from bigmart.data import RANDOM_STATE
from bigmart.models import LABELS, MODEL_NAMES, build_model

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE and R2 of one set of predictions."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    names: list[str] | None = None,
    folds: int = 5,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """K-fold CV of each model: mean and std of RMSE, MAE, R2, sorted by RMSE."""
    cv = KFold(folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in names or MODEL_NAMES:
        start = time.perf_counter()
        scores = cross_validate(build_model(name, n_jobs=n_jobs), X, y, cv=cv, scoring=SCORING)
        row = {"model": name, "label": LABELS[name]}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            values = values if metric == "r2" else -values
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def holdout_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25):
    """The fixed 75/25 split used for plots (predicted vs actual, residuals)."""
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def holdout_predictions(X: pd.DataFrame, y: pd.Series, names: list[str]) -> pd.DataFrame:
    """Fit each model on the 75% split; return actual and predicted sales on the 25%."""
    X_tr, X_va, y_tr, y_va = holdout_split(X, y)
    out = pd.DataFrame({"actual": y_va.to_numpy()}, index=X_va.index)
    for name in names:
        out[name] = build_model(name).fit(X_tr, y_tr).predict(X_va)
    return out


def feature_importance(
    model, X: pd.DataFrame, y: pd.Series, repeats: int = 5, n_jobs: int = 2
) -> pd.DataFrame:
    """Permutation importance on raw columns: RMSE increase when a column is shuffled.

    `model` must already be fitted; pass validation rows so it measures generalisation.
    """
    result = permutation_importance(
        model, X, y, scoring="neg_root_mean_squared_error", n_repeats=repeats,
        random_state=RANDOM_STATE, n_jobs=n_jobs,
    )
    table = pd.DataFrame(
        {"feature": X.columns, "rmse_increase": result.importances_mean, "std": result.importances_std}
    )
    return table.sort_values("rmse_increase", ascending=False).reset_index(drop=True)
