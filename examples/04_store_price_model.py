"""Fit the 10-number Store x price model and read it like a business report.

`sales = k[store] * Item_MRP`: k is roughly the number of units one product sells in
that store per year. Also shows that the model's predictions on a held-out 25% are as
good as gradient boosting's.

    python examples/04_store_price_model.py
"""

from bigmart.data import load_train, split_xy
from bigmart.evaluate import holdout_split, regression_metrics
from bigmart.models import build_model

X, y = split_xy(load_train())
X_tr, X_va, y_tr, y_va = holdout_split(X, y)

store_price = build_model("store_price").fit(X_tr, y_tr)
print("units sold per product, by store:")
kind = X.groupby("Outlet_Identifier")["Outlet_Type"].first()
for store, k in store_price.multipliers_.sort_values(ascending=False).items():
    print(f"  {store}  {kind[store]:18s} {k:5.1f}")

for name in ["store_price", "gradient_boosting"]:
    model = store_price if name == "store_price" else build_model(name).fit(X_tr, y_tr)
    m = regression_metrics(y_va, model.predict(X_va))
    print(f"{name:18s} hold-out RMSE {m['rmse']:,.0f} | MAE {m['mae']:,.0f} | R2 {m['r2']:.3f}")
