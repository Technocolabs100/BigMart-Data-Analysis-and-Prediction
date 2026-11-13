"""Load the raw BigMart data and clean it with the package's preprocessor.

Shows the raw data problems (missing weights and store sizes, five spellings of fat
content, zero visibilities) and the cleaned frame with the engineered features.

    python examples/01_load_and_clean.py
"""

from bigmart.data import load_train
from bigmart.preprocessing import clean

raw = load_train()
print(f"raw: {raw.shape[0]} rows x {raw.shape[1]} columns")
print("missing values:", raw.isna().sum()[lambda s: s > 0].to_dict())
print("fat content spellings:", raw["Item_Fat_Content"].value_counts().to_dict())
print("rows with zero visibility:", int((raw["Item_Visibility"] == 0).sum()))

tidy = clean(raw)
print(f"\nclean: {tidy.shape[0]} rows x {tidy.shape[1]} columns, missing values: {int(tidy.isna().sum().sum())}")
print("fat content:", tidy["Item_Fat_Content"].value_counts().to_dict())
print(tidy[["Item_Category", "Item_Visibility_Ratio", "Outlet_Age", "Outlet_Size"]].head())
