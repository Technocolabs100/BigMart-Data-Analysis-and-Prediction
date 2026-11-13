"""The business questions of the project, answered with numbers.

The original notebook started from hypotheses about what drives sales (product quality,
price, health, store size, location, ...). `check_hypotheses` tests the ones the data can
answer and returns one row per hypothesis with the evidence and a verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from bigmart.data import OUTLET_ID, TARGET

UNITS = "Units_Proxy"


def add_units(df: pd.DataFrame) -> pd.DataFrame:
    """Add `Units_Proxy = sales / Item_MRP`: roughly how many units were sold."""
    return df.assign(**{UNITS: df[TARGET] / df["Item_MRP"]})


def store_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per store: type, size, tier, age, products, mean sales and units, revenue share."""
    df = add_units(df)
    table = df.groupby(OUTLET_ID).agg(
        type=("Outlet_Type", "first"),
        size=("Outlet_Size", "first"),
        tier=("Outlet_Location_Type", "first"),
        age=("Outlet_Age", "first"),
        products=(TARGET, "size"),
        mean_sales=(TARGET, "mean"),
        mean_units=(UNITS, "mean"),
        revenue=(TARGET, "sum"),
    )
    table["revenue_share"] = table["revenue"] / table["revenue"].sum()
    return table.sort_values("mean_sales", ascending=False)


def eta_squared(df: pd.DataFrame, column: str, target: str = TARGET) -> float:
    """Share of the target's variance explained by the groups of `column` (0 to 1)."""
    grand = df[target].mean()
    groups = df.groupby(column)[target]
    between = (groups.size() * (groups.mean() - grand) ** 2).sum()
    total = ((df[target] - grand) ** 2).sum()
    return float(between / total)


def _anova_p(df: pd.DataFrame, column: str, target: str = TARGET) -> float:
    samples = [g.to_numpy() for _, g in df.groupby(column)[target]]
    return float(stats.f_oneway(*samples).pvalue)


def check_hypotheses(df: pd.DataFrame) -> pd.DataFrame:
    """Test the project's hypotheses on cleaned training data (see `preprocessing.clean`)."""
    df = add_units(df)
    rows = []

    def add(hypothesis, evidence, verdict):
        rows.append({"hypothesis": hypothesis, "evidence": evidence, "verdict": verdict})

    r_price = df["Item_MRP"].corr(df[TARGET])
    add("Higher-priced products bring more revenue",
        f"corr(MRP, sales) = {r_price:.2f}; MRP alone explains {r_price**2:.0%} of the variance",
        "Confirmed: the strongest driver")

    r_units = df["Item_MRP"].corr(df[UNITS])
    add("Price lowers the number of units sold",
        f"corr(MRP, units) = {r_units:.2f}",
        "Rejected: units sold do not depend on price")

    by_type = df.groupby("Outlet_Type")[UNITS].mean()
    add("Store format drives sales",
        f"units per product: grocery {by_type['Grocery Store']:.1f}, "
        f"supermarkets {by_type.drop('Grocery Store').min():.1f}-{by_type.max():.1f}; "
        f"eta2 = {eta_squared(df, 'Outlet_Type'):.2f}",
        "Confirmed: grocery stores sell ~7x fewer units")

    by_size = df.groupby("Outlet_Size")[TARGET].mean()
    add("Bigger stores sell more",
        ", ".join(f"{k} {v:,.0f}" for k, v in by_size.items()) + f"; eta2 = {eta_squared(df, 'Outlet_Size'):.3f}",
        "Weak: the size label adds little once store type is known")

    by_tier = df.groupby("Outlet_Location_Type")[TARGET].mean()
    add("Location tier (city size) matters",
        ", ".join(f"{k} {v:,.0f}" for k, v in by_tier.items()) + f"; eta2 = {eta_squared(df, 'Outlet_Location_Type'):.3f}",
        "Weak: mostly reflects which store types sit in each tier")

    r_age = df["Outlet_Age"].corr(df[TARGET])
    add("Older, established stores sell more",
        f"corr(store age, sales) = {r_age:.2f}; oldest store OUT027 (1985) is the best seller, "
        f"OUT019 (1985) is among the worst",
        "Rejected: age is not a driver on its own")

    r_vis = df["Item_Visibility"].corr(df[TARGET])
    add("More shelf visibility means more sales",
        f"corr(visibility, sales) = {r_vis:.2f}",
        "Rejected: slightly negative, not positive")

    fat = df[df["Item_Fat_Content"] != "Non-Edible"]
    by_fat = fat.groupby("Item_Fat_Content")[UNITS].mean()
    add("Healthy (low-fat) products sell more",
        f"units: Low Fat {by_fat['Low Fat']:.1f} vs Regular {by_fat['Regular']:.1f}; "
        f"ANOVA p = {_anova_p(fat, 'Item_Fat_Content', UNITS):.2f}",
        "Rejected: no difference")

    by_item = df.groupby("Item_Type")[UNITS].mean()
    add("Some product categories sell much better",
        f"units per product range {by_item.min():.1f}-{by_item.max():.1f} across 16 types; "
        f"eta2 = {eta_squared(df, 'Item_Type', UNITS):.3f}",
        "Rejected: category barely matters")

    return pd.DataFrame(rows)


def units_near_integer_share(df: pd.DataFrame, tolerance: float = 0.05) -> tuple[float, float]:
    """Share of rows whose `sales / MRP` is within `tolerance` of a whole number.

    Returns (observed share, share expected if the fractional part were uniform).
    """
    units = df[TARGET] / df["Item_MRP"]
    frac = np.abs(units - np.round(units))
    return float((frac < tolerance).mean()), 2 * tolerance
