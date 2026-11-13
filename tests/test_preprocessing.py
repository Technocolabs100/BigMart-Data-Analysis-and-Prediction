import numpy as np
import pandas as pd

from bigmart.preprocessing import FEATURES, BigMartPreprocessor, clean, normalize_fat_content


def test_fat_content_spellings():
    fat = pd.Series(["LF", "low fat", "Low Fat", "reg", "Regular", "Low Fat"])
    cat = pd.Series(["Food"] * 5 + ["Non-Consumable"])
    assert normalize_fat_content(fat, cat).tolist() == [
        "Low Fat", "Low Fat", "Low Fat", "Regular", "Regular", "Non-Edible"
    ]


def test_clean_has_no_missing_values_and_no_zero_visibility(raw):
    out = clean(raw)
    assert out[FEATURES].isna().sum().sum() == 0
    assert (out["Item_Visibility"] > 0).all()
    assert set(out["Item_Fat_Content"]) == {"Low Fat", "Regular", "Non-Edible"}
    assert out["Outlet_Age"].between(4, 28).all()


def test_weight_is_filled_from_the_same_product(raw):
    out = clean(raw)
    missing = raw["Item_Weight"].isna()
    known = raw.groupby("Item_Identifier")["Item_Weight"].mean()
    expected = raw.loc[missing, "Item_Identifier"].map(known)
    has_ref = expected.notna()
    np.testing.assert_allclose(out.loc[missing[missing].index[has_ref], "Item_Weight"], expected[has_ref])


def test_transform_uses_only_fit_rows(raw):
    # Fit on half the rows, transform the other half: no NaNs even for unseen products.
    half = len(raw) // 2
    pre = BigMartPreprocessor().fit(raw.iloc[:half])
    out = pre.transform(raw.iloc[half:])
    assert out.isna().sum().sum() == 0
    assert list(out.columns) == FEATURES


def test_unknown_product_falls_back(raw):
    pre = BigMartPreprocessor().fit(raw)
    row = raw.iloc[[0]].copy()
    row["Item_Identifier"] = "FDZZZ99"
    row["Item_Weight"] = np.nan
    row["Item_Visibility"] = 0.0
    out = pre.transform(row)
    assert out.notna().all(axis=None)
