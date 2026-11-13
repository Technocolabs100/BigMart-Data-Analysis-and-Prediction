"""Rebuild everything in reports/: model comparison, hypotheses, figures, submission.

    python scripts/make_report.py

Takes about two minutes on a laptop CPU. The numbers in the README come from the files
this script writes (reports/metrics.json, reports/model_comparison.csv, reports/hypotheses.md).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from bigmart import plots
from bigmart.data import ITEM_ID, OUTLET_ID, ROOT, TARGET, load_test, load_train, split_xy
from bigmart.evaluate import (
    compare_models,
    feature_importance,
    holdout_predictions,
    holdout_split,
    regression_metrics,
)
from bigmart.insights import check_hypotheses, store_summary, units_near_integer_share
from bigmart.models import BEST_MODEL, LABELS, build_model
from bigmart.preprocessing import clean

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main() -> None:
    raw = load_train()
    X, y = split_xy(raw)
    tidy = clean(raw)

    print("1/5 hypotheses and store table")
    hypotheses = check_hypotheses(tidy)
    stores = store_summary(tidy)
    near, expected = units_near_integer_share(raw)
    lines = ["| Hypothesis | Evidence | Verdict |", "|---|---|---|"]
    lines += [f"| {r.hypothesis} | {r.evidence} | {r.verdict} |" for r in hypotheses.itertuples()]
    (REPORTS / "hypotheses.md").write_text("\n".join(lines) + "\n")
    stores.round(3).to_csv(REPORTS / "store_summary.csv")

    print("2/5 cross-validating all models")
    results = compare_models(X, y)
    results.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)
    print(results[["label", "rmse", "rmse_std", "mae", "r2", "seconds"]].round(3).to_string(index=False))

    print("3/5 hold-out predictions and feature importance")
    shown = ["mean_baseline", "linear_regression", "gradient_boosting", BEST_MODEL]
    preds = holdout_predictions(X, y, shown)
    X_tr, X_va, y_tr, y_va = holdout_split(X, y)
    gbm = build_model("gradient_boosting").fit(X_tr, y_tr)
    importance = feature_importance(gbm, X_va, y_va)
    importance.round(2).to_csv(REPORTS / "feature_importance.csv", index=False)

    print("4/5 figures")
    full_model = build_model(BEST_MODEL).fit(X, y)
    plots.save(plots.missing_values(raw), FIGURES / "missing_values.png")
    plots.save(plots.sales_distribution(tidy), FIGURES / "sales_distribution.png")
    plots.save(plots.sales_by_store(tidy), FIGURES / "sales_by_store.png")
    plots.save(plots.price_vs_sales(tidy, full_model.multipliers_), FIGURES / "price_vs_sales.png")
    plots.save(plots.units_by_product(tidy), FIGURES / "units_by_product.png")
    plots.save(plots.correlation_heatmap(tidy), FIGURES / "correlation_heatmap.png")
    plots.save(plots.model_comparison(results), FIGURES / "model_comparison.png")
    plots.save(plots.predicted_vs_actual(preds, ["mean_baseline", "linear_regression", BEST_MODEL], LABELS),
               FIGURES / "before_after.png")
    plots.save(plots.feature_importance(importance, "What gradient boosting actually uses"),
               FIGURES / "feature_importance.png")
    plots.save(plots.residuals(preds, BEST_MODEL, LABELS[BEST_MODEL]), FIGURES / "residuals.png")

    print("5/5 submission for Test.csv")
    test = load_test()
    submission = pd.DataFrame({ITEM_ID: test[ITEM_ID], OUTLET_ID: test[OUTLET_ID],
                               TARGET: full_model.predict(test).clip(min=0)})
    submission.to_csv(REPORTS / "submission.csv", index=False)

    metrics = {
        "cv_folds": 5,
        "models": results.set_index("model")[["rmse", "rmse_std", "mae", "r2"]].round(4).to_dict("index"),
        "best_model": BEST_MODEL,
        "store_multipliers": full_model.multipliers_.round(3).to_dict(),
        "units_near_integer_share": round(near, 4),
        "units_near_integer_expected": expected,
        "holdout": {
            name: {k: round(v, 4) for k, v in regression_metrics(preds["actual"], preds[name]).items()}
            for name in shown
        },
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("done:", ", ".join(sorted(p.name for p in REPORTS.iterdir())))


if __name__ == "__main__":
    sys.exit(main())
