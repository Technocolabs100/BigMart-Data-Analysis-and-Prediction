"""Loading the BigMart data and the names of its columns.

The data comes from the Analytics Vidhya "BigMart Sales Prediction" practice problem:
2013 sales of 1,559 products in 10 BigMart stores. `Train.csv` (8,523 rows) has the
target `Item_Outlet_Sales`; `Test.csv` (5,681 rows) has the same product/store pairs
without it. Both files ship in `data/raw/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

TARGET = "Item_Outlet_Sales"
ITEM_ID = "Item_Identifier"
OUTLET_ID = "Outlet_Identifier"

# The year the sales were recorded, used to turn the opening year into a store age.
DATA_YEAR = 2013

RANDOM_STATE = 42


def load_train(path: str | Path | None = None) -> pd.DataFrame:
    """The labelled data: 8,523 product/store rows with `Item_Outlet_Sales`."""
    return pd.read_csv(path or RAW_DIR / "Train.csv")


def load_test(path: str | Path | None = None) -> pd.DataFrame:
    """The unlabelled data: 5,681 product/store rows to predict."""
    return pd.read_csv(path or RAW_DIR / "Test.csv")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target of a labelled frame."""
    return df.drop(columns=TARGET), df[TARGET]
