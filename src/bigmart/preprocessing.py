"""Cleaning and feature engineering as one scikit-learn transformer.

Everything learned from data (item weights, item visibilities, store sizes) is learned in
`fit`, on the training rows only, so the transformer can sit inside a `Pipeline` and be
cross-validated without leaking validation rows into the cleaning step.

What it fixes, found during the EDA:

- `Item_Fat_Content` has five spellings of two values (`Low Fat`, `LF`, `low fat`,
  `Regular`, `reg`), and non-consumables (household, health) cannot have a fat content.
- `Item_Weight` is missing in 17% of rows, but the same product appears in other stores
  with its weight, so it is filled per `Item_Identifier`.
- `Item_Visibility` is 0 in 526 rows, which is impossible for a product that sells; it is
  replaced by the product's mean visibility.
- `Outlet_Size` is missing for three whole stores (OUT010, OUT017, OUT045); it is filled
  with the most common size among stores of the same `Outlet_Type`.

What it adds:

- `Item_Category`: Food / Drinks / Non-Consumable, from the identifier prefix (FD, DR, NC).
- `Outlet_Age`: years since the store opened (sales are from 2013).
- `Item_Visibility_Ratio`: a product's visibility in this store divided by its mean
  visibility across stores (how much shelf space this store gives it, relatively).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from bigmart.data import DATA_YEAR, ITEM_ID, OUTLET_ID, TARGET

FAT_CONTENT = {"low fat": "Low Fat", "lf": "Low Fat", "regular": "Regular", "reg": "Regular"}
CATEGORY = {"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"}

CATEGORICAL = [
    "Item_Fat_Content",
    "Item_Type",
    "Item_Category",
    OUTLET_ID,
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]
NUMERIC = ["Item_Weight", "Item_Visibility", "Item_Visibility_Ratio", "Item_MRP", "Outlet_Age"]
FEATURES = CATEGORICAL + NUMERIC


def normalize_fat_content(fat: pd.Series, category: pd.Series) -> pd.Series:
    """Map the five spellings to Low Fat / Regular, and non-consumables to Non-Edible."""
    fat = fat.str.strip().str.lower().map(FAT_CONTENT).fillna(fat)
    return fat.where(category != "Non-Consumable", "Non-Edible")


class BigMartPreprocessor(BaseEstimator, TransformerMixin):
    """Clean raw BigMart rows and add engineered features.

    Returns a DataFrame with the columns in `FEATURES` (the original identifiers, the
    opening year and the target are dropped). Unknown products fall back to category
    means, so it also works on products never seen in `fit`.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "BigMartPreprocessor":
        X = X.drop(columns=TARGET, errors="ignore")
        self.item_weight_ = X.groupby(ITEM_ID)["Item_Weight"].mean()
        self.type_weight_ = X.groupby("Item_Type")["Item_Weight"].mean()
        visible = X["Item_Visibility"].where(X["Item_Visibility"] > 0)
        self.item_visibility_ = visible.groupby(X[ITEM_ID]).mean()
        self.type_visibility_ = visible.groupby(X["Item_Type"]).mean()
        sizes = X.dropna(subset=["Outlet_Size"]).groupby("Outlet_Type")["Outlet_Size"]
        self.size_by_type_ = sizes.agg(lambda s: s.mode().iloc[0])
        self.default_size_ = X["Outlet_Size"].mode().iloc[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=TARGET, errors="ignore").copy()
        out["Item_Category"] = out[ITEM_ID].str[:2].map(CATEGORY).fillna("Food")
        out["Item_Fat_Content"] = normalize_fat_content(out["Item_Fat_Content"], out["Item_Category"])

        weight = out[ITEM_ID].map(self.item_weight_)
        weight = weight.fillna(out["Item_Type"].map(self.type_weight_))
        out["Item_Weight"] = out["Item_Weight"].fillna(weight).fillna(self.type_weight_.mean())

        item_vis = out[ITEM_ID].map(self.item_visibility_)
        item_vis = item_vis.fillna(out["Item_Type"].map(self.type_visibility_))
        item_vis = item_vis.fillna(self.type_visibility_.mean())
        visibility = out["Item_Visibility"].where(out["Item_Visibility"] > 0, item_vis)
        out["Item_Visibility"] = visibility
        out["Item_Visibility_Ratio"] = visibility / item_vis

        size = out["Outlet_Type"].map(self.size_by_type_).fillna(self.default_size_)
        out["Outlet_Size"] = out["Outlet_Size"].fillna(size)
        out["Outlet_Age"] = DATA_YEAR - out["Outlet_Establishment_Year"]
        return out[FEATURES]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURES, dtype=object)


def clean(train: pd.DataFrame, other: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit the preprocessor on `train` and return `other` (default: `train`) cleaned.

    The target column is kept when present, which is handy for the EDA.
    """
    other = train if other is None else other
    out = BigMartPreprocessor().fit(train).transform(other)
    if TARGET in other:
        out[TARGET] = other[TARGET].to_numpy()
    return out
