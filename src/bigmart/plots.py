"""The figures of the README and the notebook, in one consistent style.

Every function draws one figure and returns it; `save` writes it as a PNG. Colours follow
the entity (an outlet type keeps its colour in every chart), never its rank.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bigmart.data import OUTLET_ID, TARGET
from bigmart.insights import UNITS, add_units

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a9a8a2"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

OUTLET_TYPES = ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]
TYPE_COLOR = dict(zip(OUTLET_TYPES, [ORANGE, BLUE, AQUA, YELLOW]))
TYPE_MARKER = dict(zip(OUTLET_TYPES, ["o", "s", "^", "D"]))


def use_style() -> None:
    """A light, recessive-grid matplotlib style shared by every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


use_style()


def show(fig: plt.Figure) -> None:
    """Display a figure once in a notebook, then free it."""
    from IPython.display import display

    display(fig)
    plt.close(fig)


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _type_legend(ax, loc="upper left") -> None:
    handles = [
        plt.Line2D([], [], marker=TYPE_MARKER[t], ls="", color=TYPE_COLOR[t], markersize=8, label=t)
        for t in OUTLET_TYPES
    ]
    ax.legend(handles=handles, loc=loc)


# ---------------------------------------------------------------- EDA figures


def missing_values(raw: pd.DataFrame) -> plt.Figure:
    """Share of missing values per column of the raw data (and zero visibilities)."""
    share = raw.isna().mean()
    share["Item_Visibility (= 0)"] = (raw["Item_Visibility"] == 0).mean()
    share = share[share > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(share.index, share.values * 100, color=BLUE, height=0.55)
    for y, v in enumerate(share.values):
        ax.text(v * 100 + 0.5, y, f"{v:.1%}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("% of rows")
    ax.set_title("Raw data problems fixed by the preprocessor")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, share.max() * 100 * 1.2)
    return fig


def sales_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of sales and of the units proxy (sales / MRP)."""
    df = add_units(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(df[TARGET], bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    axes[0].axvline(df[TARGET].mean(), color=INK, lw=1.2, ls="--")
    axes[0].text(df[TARGET].mean() * 1.05, axes[0].get_ylim()[1] * 0.9, f"mean {df[TARGET].mean():,.0f}", color=INK)
    axes[0].set_title("Item_Outlet_Sales is right-skewed")
    axes[0].set_xlabel("sales of one product in one store (2013)")
    axes[0].set_ylabel("rows")
    for t in OUTLET_TYPES:
        axes[1].hist(df.loc[df["Outlet_Type"] == t, UNITS], bins=np.arange(0, 58, 1), density=True,
                     histtype="step", lw=2, color=TYPE_COLOR[t], label=t)
    axes[1].set_title("Units sold (sales / MRP) split by store type")
    axes[1].set_xlabel("units proxy")
    axes[1].set_ylabel("share of the store type's rows")
    axes[1].legend()
    return fig


def sales_by_store(df: pd.DataFrame) -> plt.Figure:
    """Mean sales per store, coloured by store type."""
    stores = df.groupby(OUTLET_ID).agg(sales=(TARGET, "mean"), kind=("Outlet_Type", "first"),
                                       year=("Outlet_Age", "first")).sort_values("sales")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [TYPE_COLOR[k] for k in stores["kind"]]
    ax.barh(stores.index, stores["sales"], color=colors, height=0.6)
    for y, v in enumerate(stores["sales"]):
        ax.text(v + 30, y, f"{v:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("mean sales per product")
    ax.set_title("Store type decides how much a product sells")
    ax.grid(axis="y", visible=False)
    _type_legend(ax, loc="lower right")
    return fig


def price_vs_sales(df: pd.DataFrame, multipliers: pd.Series | None = None) -> plt.Figure:
    """Sales against price, by store type, with the fitted `k[store] * MRP` lines."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for t in OUTLET_TYPES:
        part = df[df["Outlet_Type"] == t]
        ax.scatter(part["Item_MRP"], part[TARGET], s=9, alpha=0.35, color=TYPE_COLOR[t],
                   marker=TYPE_MARKER[t], linewidths=0)
    if multipliers is not None:
        kind = df.groupby(OUTLET_ID)["Outlet_Type"].first()
        price = np.linspace(0, df["Item_MRP"].max() * 1.02, 2)
        for store, k in multipliers.items():
            ax.plot(price, k * price, color=TYPE_COLOR[kind[store]], lw=1.6)
        top = multipliers.idxmax()
        ax.annotate(f"{top}: {multipliers.max():.1f} units / product", (price[1], multipliers.max() * price[1]),
                    xytext=(-4, 10), textcoords="offset points", ha="right", color=INK, fontsize=9)
        low = multipliers.idxmin()
        ax.annotate(f"grocery stores: ~{multipliers.min():.1f} units", (price[1], multipliers.min() * price[1]),
                    xytext=(-4, 10), textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_xlabel("Item_MRP (list price)")
    ax.set_ylabel("Item_Outlet_Sales")
    ax.set_title("Sales = price x units, and units depend on the store")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    _type_legend(ax)
    return fig


def units_by_product(df: pd.DataFrame) -> plt.Figure:
    """Units sold per product type and fat content: almost flat."""
    df = add_units(df)
    by_type = df.groupby("Item_Type")[UNITS].mean().sort_values()
    fat = df.groupby("Item_Fat_Content")[UNITS].mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].barh(by_type.index, by_type.values, color=BLUE, height=0.6)
    axes[0].set_xlim(0, by_type.max() * 1.25)
    axes[0].set_xlabel("mean units per product")
    axes[0].set_title("Product type barely changes units sold")
    axes[0].grid(axis="y", visible=False)
    axes[1].bar(fat.index, fat.values, color=BLUE, width=0.55)
    for x, v in enumerate(fat.values):
        axes[1].text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK)
    axes[1].set_ylim(0, fat.max() * 1.25)
    axes[1].set_title("Neither does fat content")
    axes[1].grid(axis="x", visible=False)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Pearson correlations of the numeric columns (with the target and units proxy)."""
    cols = ["Item_Weight", "Item_Visibility", "Item_Visibility_Ratio", "Item_MRP", "Outlet_Age", UNITS, TARGET]
    corr = add_units(df)[cols].corr()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace("Item_", "").replace("Outlet_", "") for c in cols], rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), [c.replace("Item_", "").replace("Outlet_", "") for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Only price correlates with sales")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


# -------------------------------------------------------------- model figures


def model_comparison(results: pd.DataFrame) -> plt.Figure:
    """Cross-validated RMSE per model, with the fold standard deviation."""
    res = results.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    best = res["rmse"].idxmin()
    colors = [BLUE if i == best else NEUTRAL for i in res.index]
    ax.barh(res["label"], res["rmse"], xerr=res["rmse_std"], color=colors, height=0.6,
            error_kw={"ecolor": INK_SOFT, "capsize": 3, "lw": 1})
    for y, (v, r2) in enumerate(zip(res["rmse"], res["r2"])):
        ax.text(v + res["rmse_std"].max() + 25, y, f"{v:,.0f}  (R2 {r2:.3f})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, res["rmse"].max() * 1.35)
    ax.set_xlabel("5-fold CV RMSE (lower is better)")
    ax.set_title("A 10-number model matches gradient boosting")
    ax.grid(axis="y", visible=False)
    return fig


def predicted_vs_actual(preds: pd.DataFrame, names: list[str], labels: dict[str, str]) -> plt.Figure:
    """Side-by-side predicted vs actual scatter for each model in `names`."""
    lim = float(max(preds["actual"].max(), preds[names].max().max())) * 1.02
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.3), sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        err = preds[name] - preds["actual"]
        rmse = float(np.sqrt((err**2).mean()))
        r2 = 1 - float((err**2).sum() / ((preds["actual"] - preds["actual"].mean()) ** 2).sum())
        ax.scatter(preds["actual"], preds[name], s=7, alpha=0.35, color=BLUE, linewidths=0)
        ax.plot([0, lim], [0, lim], color=INK, lw=1, ls="--")
        ax.set_title(f"{labels[name]}\nRMSE {rmse:,.0f} | R2 {r2:.2f}", fontsize=11)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("actual sales")
    np.atleast_1d(axes)[0].set_ylabel("predicted sales")
    return fig


def feature_importance(table: pd.DataFrame, title: str) -> plt.Figure:
    """Permutation importance bars (RMSE increase when a column is shuffled)."""
    table = table.sort_values("rmse_increase")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(table["feature"], table["rmse_increase"], xerr=table["std"], color=BLUE, height=0.6,
            error_kw={"ecolor": INK_SOFT, "capsize": 2, "lw": 1})
    ax.axvline(0, color=INK_SOFT, lw=0.8)
    ax.set_xlabel("RMSE increase when the column is shuffled")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return fig


def residuals(preds: pd.DataFrame, name: str, label: str) -> plt.Figure:
    """Residuals against predictions: the error grows with the prediction."""
    fitted = preds[name]
    resid = preds["actual"] - fitted
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.scatter(fitted, resid, s=7, alpha=0.35, color=BLUE, linewidths=0)
    ax.axhline(0, color=INK, lw=1, ls="--")
    bins = pd.qcut(fitted, 10)
    spread = resid.groupby(bins, observed=True).std()
    centers = fitted.groupby(bins, observed=True).mean()
    ax.plot(centers, spread, color=ORANGE, lw=2, marker="o", markersize=5, label="+1 std of residuals")
    ax.plot(centers, -spread, color=ORANGE, lw=2, marker="o", markersize=5)
    ax.set_xlabel("predicted sales")
    ax.set_ylabel("actual - predicted")
    ax.set_title(f"{label}: errors grow with the sales level")
    ax.legend(loc="upper left")
    return fig
