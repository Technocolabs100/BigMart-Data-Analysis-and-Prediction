"""The candidate models, each wrapped in the same leakage-free pipeline.

    raw rows -> BigMartPreprocessor -> encoding -> regressor

Linear models get one-hot categories and standardised numbers; tree models get ordinal
codes, which they split on directly.

`StorePriceRegressor` is the model the EDA pointed to: sales are price times units sold,
and units sold depend almost only on the store, so it predicts
`sales = k[store] * Item_MRP` with one multiplier per store (10 numbers in total).

Sales are right-skewed, so `build_model(..., log_target=True)` fits on `log1p(sales)`.
It lowers MAE but raises RMSE (the expm1 of a mean log is a median, which underestimates
the mean), so it is off by default; see docs/methodology.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from bigmart.data import OUTLET_ID, RANDOM_STATE
from bigmart.preprocessing import CATEGORICAL, NUMERIC, BigMartPreprocessor

MODEL_NAMES = [
    "mean_baseline",
    "linear_regression",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "store_price",
]

# Readable names for tables and plots.
LABELS = {
    "mean_baseline": "Mean baseline",
    "linear_regression": "Linear regression",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "store_price": "Store x price",
}

BEST_MODEL = "store_price"


class StorePriceRegressor(BaseEstimator, RegressorMixin):
    """`sales = k[store] * Item_MRP`, one least-squares multiplier per store.

    `k[store]` reads as "average units sold per product" in that store. Stores never seen
    in `fit` get the multiplier fitted on all rows.
    """

    def fit(self, X: pd.DataFrame, y) -> "StorePriceRegressor":
        price = X["Item_MRP"].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        store = X[OUTLET_ID].to_numpy()
        num = pd.Series(price * y).groupby(store).sum()
        den = pd.Series(price**2).groupby(store).sum()
        self.multipliers_ = num / den
        self.global_multiplier_ = float((price * y).sum() / (price**2).sum())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        k = X[OUTLET_ID].map(self.multipliers_).fillna(self.global_multiplier_)
        return (k * X["Item_MRP"]).to_numpy(dtype=float)


def _linear_encoding() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def _tree_encoding() -> ColumnTransformer:
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("cat", ordinal, CATEGORICAL), ("num", "passthrough", NUMERIC)],
        verbose_feature_names_out=False,
    )


def _regressor(name: str, n_jobs: int):
    if name == "mean_baseline":
        return DummyRegressor(strategy="mean")
    if name == "linear_regression":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20, n_jobs=n_jobs, random_state=RANDOM_STATE
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4, min_samples_leaf=40,
            subsample=0.8, random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, n_jobs=n_jobs, random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown model {name!r}, choose from {MODEL_NAMES}")


def build_model(
    name: str = BEST_MODEL, n_jobs: int = 2, log_target: bool = False
) -> Pipeline | TransformedTargetRegressor | StorePriceRegressor:
    """A ready-to-fit estimator that takes raw BigMart rows and predicts sales.

    >>> model = build_model("xgboost").fit(X_train, y_train)
    >>> model.predict(X_test)
    """
    if name == "store_price":
        return StorePriceRegressor()
    linear = name in {"mean_baseline", "linear_regression", "ridge"}
    pipe = Pipeline(
        [
            ("prepare", BigMartPreprocessor()),
            ("encode", _linear_encoding() if linear else _tree_encoding()),
            ("model", _regressor(name, n_jobs)),
        ]
    )
    if name == "mean_baseline" or not log_target:
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
