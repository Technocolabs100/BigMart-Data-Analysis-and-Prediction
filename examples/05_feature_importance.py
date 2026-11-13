"""Which columns does a gradient-boosting model really use? (permutation importance)

Fits the model on 75% of the rows and shuffles each raw column of the other 25%:
the RMSE increase is that column's importance. Price and store dominate.

    python examples/05_feature_importance.py
"""

from pathlib import Path

from bigmart import plots
from bigmart.data import load_train, split_xy
from bigmart.evaluate import feature_importance, holdout_split
from bigmart.models import build_model

OUT = Path(__file__).parent / "output"

X, y = split_xy(load_train())
X_tr, X_va, y_tr, y_va = holdout_split(X, y)
model = build_model("gradient_boosting").fit(X_tr, y_tr)
table = feature_importance(model, X_va, y_va)
print(table.round(1).to_string(index=False))
print("saved", plots.save(plots.feature_importance(table, "Permutation importance"), OUT / "feature_importance.png"))
