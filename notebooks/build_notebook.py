"""Build notebooks/bigmart_sales_analysis.ipynb, the end-to-end analysis notebook.

The notebook tells the project's story (data problems -> drivers of sales -> models ->
conclusions) by calling the `bigmart` package, so the notebook and the package never
disagree. It runs locally from the repo, on Colab and on Kaggle (it installs the package
from GitHub when it is not importable). Regenerate and execute after changing src/:

    python notebooks/build_notebook.py --execute
"""

import argparse
import os
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "bigmart_sales_analysis.ipynb"
REPO = "https://github.com/benisalla/BigMart-Data-Analysis-and-Prediction"

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(f"""
# BigMart sales: what drives them, and how well can we predict them?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benisalla/BigMart-Data-Analysis-and-Prediction/blob/main/notebooks/bigmart_sales_analysis.ipynb)

BigMart recorded the 2013 sales of **1,559 products in 10 stores**. The goal is to
predict `Item_Outlet_Sales` for each product/store pair and, more usefully, to find
**which product and store properties drive sales**, so the business knows where to act.

This notebook is the full analysis, end to end. All the code lives in the `bigmart`
package of the repository ({REPO}); the notebook calls it, so you can read the story
here and reuse every step from Python.

1. The data and its problems
2. Hypotheses: what should drive sales?
3. Exploratory analysis: what actually drives sales
4. Models, compared with 5-fold cross-validation
5. Conclusions and recommendations
""")

code("""
# Use the local package when run from the repo; install it from GitHub on Colab / Kaggle.
try:
    import bigmart
except ImportError:
    %pip install -q git+https://github.com/benisalla/BigMart-Data-Analysis-and-Prediction
    import bigmart

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

plt.ioff()  # figures are shown once, by plots.show

from bigmart import plots
from bigmart.data import load_test, load_train, split_xy

# The data ships in the repo; on Colab / Kaggle the files are fetched from GitHub.
RAW = "https://raw.githubusercontent.com/benisalla/BigMart-Data-Analysis-and-Prediction/main/data/raw"
local = Path(bigmart.__file__).resolve().parents[2] / "data" / "raw" / "Train.csv"
train_path, test_path = (None, None) if local.exists() else (f"{RAW}/Train.csv", f"{RAW}/Test.csv")

raw = load_train(train_path)
test = load_test(test_path)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 110)
print(f"train {raw.shape}, test {test.shape}")
raw.head()
""")

md("""
## 1. The data and its problems

Each row is one product in one store. Eleven features describe the product (identifier,
weight, fat content, visibility, type, list price `Item_MRP`) and the store (identifier,
opening year, size, location tier, type). The data has four quality problems:
""")

code("""
plots.show(plots.missing_values(raw))
print(raw["Item_Fat_Content"].value_counts().to_dict())
print("stores with no Outlet_Size:", sorted(raw.loc[raw.Outlet_Size.isna(), "Outlet_Identifier"].unique()))
""")

md("""
| Problem | Fix (in `BigMartPreprocessor`) |
|---|---|
| `Item_Weight` missing in 17% of rows | the same product is sold in other stores with its weight: fill per `Item_Identifier` |
| `Outlet_Size` missing for 3 whole stores | most common size among stores of the same `Outlet_Type` |
| `Item_Visibility` = 0 in 526 rows (impossible for a product that sells) | the product's mean visibility |
| `Item_Fat_Content` has 5 spellings; household/health products have a "fat content" | map to Low Fat / Regular, and Non-Edible for non-consumables |

Three features are added: `Item_Category` (Food / Drinks / Non-Consumable, from the
identifier prefix), `Outlet_Age` and `Item_Visibility_Ratio`. Everything the preprocessor
learns (weights, visibilities, sizes) is learned in `fit`, on training rows only, so it
can sit inside a cross-validated pipeline without leaking validation data.
""")

code("""
from bigmart.preprocessing import clean

tidy = clean(raw)
print("missing values after cleaning:", int(tidy.isna().sum().sum()))
tidy.head()
""")

md("""
## 2. Hypotheses: what should drive sales?

Before looking at the data, the project listed what *should* matter: product price,
quality and health (fat content), shelf visibility, product category; store size,
location (city tier), format and age. Each one is tested below.

## 3. Exploratory analysis

### Sales are price x units

The target is skewed, but dividing it by the list price gives a "units sold" proxy that
is much better behaved, and that clearly separates the store types.
""")

code("""
plots.show(plots.sales_distribution(tidy))
""")

code("""
from bigmart.models import build_model

store_price = build_model("store_price").fit(*split_xy(raw))
plots.show(plots.price_vs_sales(tidy, store_price.multipliers_))
""")

md("""
Each line is one store's least-squares fit of `sales = k * MRP`. Within a store, the
points fan out around a straight line through the origin: revenue grows with price
because **the number of units sold does not fall with price**. The slope `k` (units per
product) is set by the store: ~2.4 in the two grocery stores, 14-17 in supermarkets,
26 in the one Type 3 supermarket (OUT027).

The price axis also shows four price bands (gaps near 70, 135 and 200), a sign that
BigMart prices in tiers.
""")

code("""
from bigmart.insights import store_summary, units_near_integer_share

plots.show(plots.sales_by_store(tidy))
store_summary(tidy).round(3)
""")

code("""
near, expected = units_near_integer_share(raw)
print(f"sales / MRP is within 0.05 of a whole number in {near:.0%} of rows (uniform would give {expected:.0%})")
""")

md("""
That last check supports the reading of `sales / MRP` as units: a third of the rows sit
close to a whole number of units, three times more than chance.

### Product attributes barely matter
""")

code("""
plots.show(plots.units_by_product(tidy))
plots.show(plots.correlation_heatmap(tidy))
""")

md("""
### The hypotheses, tested
""")

code("""
from bigmart.insights import check_hypotheses

check_hypotheses(tidy)
""")

md("""
## 4. Models

Every product and every store of `Test.csv` also appears in `Train.csv`, so shuffled
5-fold cross-validation matches the real task. RMSE (the competition metric) is the main
score. Every model is a full pipeline, refitted on each fold:

    raw rows -> BigMartPreprocessor -> encoding -> regressor

The candidates: a mean baseline, linear and ridge regression (one-hot features), random
forest, gradient boosting and XGBoost (ordinal features), and **Store x price**, the
10-parameter model the EDA points to (`sales = k[store] * Item_MRP`).
""")

code("""
from bigmart.evaluate import compare_models

X, y = split_xy(raw)
results = compare_models(X, y)
plots.show(plots.model_comparison(results))
results[["label", "rmse", "rmse_std", "mae", "r2"]].round(3)
""")

code("""
from bigmart.evaluate import holdout_predictions
from bigmart.models import LABELS

preds = holdout_predictions(X, y, ["mean_baseline", "linear_regression", "store_price"])
plots.show(plots.predicted_vs_actual(preds, ["mean_baseline", "linear_regression", "store_price"], LABELS))
plots.show(plots.residuals(preds, "store_price", LABELS["store_price"]))
""")

md("""
The error grows with the sales level: a product's units vary a lot around its store's
average, and nothing in the data explains that variation. This is why every model,
however flexible, stops near R2 = 0.60.

### What the flexible models learn
""")

code("""
from bigmart.evaluate import feature_importance, holdout_split

X_tr, X_va, y_tr, y_va = holdout_split(X, y)
gbm = build_model("gradient_boosting").fit(X_tr, y_tr)
importance = feature_importance(gbm, X_va, y_va)
plots.show(plots.feature_importance(importance, "What gradient boosting actually uses"))
importance.round(1)
""")

md("""
Gradient boosting, given all eleven columns, ends up using two: price and store type
(the opening year acts as a store identifier). It rediscovers the Store x price model,
with more parameters and slightly more variance.

### Submission for Test.csv
""")

code("""
submission = test[["Item_Identifier", "Outlet_Identifier"]].assign(
    Item_Outlet_Sales=store_price.predict(test).clip(min=0)
)
submission.head()
""")

md("""
## 5. Conclusions

- **Sales = list price x units sold.** Price alone explains 32% of the variance; units do
  not fall as price rises.
- **The store decides the units.** A product sells ~2.4 units in a grocery store, ~16 in a
  Type 1 supermarket and ~26 in the Type 3 supermarket (OUT027), which brings 19% of
  revenue alone. Store format explains far more than size, city tier or age.
- **Product attributes do not move units**: fat content, category, weight and shelf
  visibility have no meaningful effect.
- **A 10-number model is the best predictor**: Store x price reaches CV RMSE 1,072
  (R2 0.605), ahead of random forest, gradient boosting and XGBoost (1,084-1,088), with
  no tuning, in milliseconds, and fully explainable.
- **The ceiling is the data, not the model.** The remaining 40% of the variance is
  product-in-store variation that no column describes. Promotions, shelf position,
  local demand or stock-outs would be the data to collect next.

**For the business:** growth comes from the store side (converting grocery stores and
Type 1 supermarkets toward the Type 3 format, which sells 1.6x more units per product),
not from product attributes like fat content or packaging visibility.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the notebook so it ships with outputs")
    args = parser.parse_args()
    if args.execute:
        from nbclient import NotebookClient

        os.environ.setdefault("OMP_NUM_THREADS", "2")
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, OUT)
    print("wrote", OUT)
