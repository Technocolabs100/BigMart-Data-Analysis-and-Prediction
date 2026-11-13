"""Command-line entry points: `bigmart-train` and `bigmart-predict`."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from bigmart.data import ITEM_ID, OUTLET_ID, ROOT, TARGET, load_test, load_train, split_xy
from bigmart.evaluate import compare_models
from bigmart.models import BEST_MODEL, MODEL_NAMES, build_model

DEFAULT_MODEL_PATH = ROOT / "models" / "bigmart_model.pkl"


def train(argv: list[str] | None = None) -> None:
    """Cross-validate a model on Train.csv, then fit it on all rows and save it."""
    parser = argparse.ArgumentParser(description=train.__doc__)
    parser.add_argument("--model", default=BEST_MODEL, choices=MODEL_NAMES)
    parser.add_argument("--train", type=Path, default=None, help="labelled CSV (default: data/raw/Train.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    X, y = split_xy(load_train(args.train))
    scores = compare_models(X, y, names=[args.model], folds=args.folds).iloc[0]
    print(f"{args.model}: CV RMSE {scores.rmse:,.1f} +- {scores.rmse_std:.1f} | "
          f"MAE {scores.mae:,.1f} | R2 {scores.r2:.4f}")

    model = build_model(args.model).fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"name": args.model, "model": model}, f)
    meta = {k: float(scores[k]) for k in ("rmse", "rmse_std", "mae", "r2")}
    args.out.with_suffix(".json").write_text(json.dumps({"model": args.model, "cv": meta}, indent=2))
    print(f"saved {args.out}")


def predict(argv: list[str] | None = None) -> None:
    """Predict Item_Outlet_Sales for every row of a CSV with a saved model."""
    parser = argparse.ArgumentParser(description=predict.__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=None, help="CSV to predict (default: data/raw/Test.csv)")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args(argv)

    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    rows = load_test(args.input)
    pred = saved["model"].predict(rows).clip(min=0)
    out = pd.DataFrame({ITEM_ID: rows[ITEM_ID], OUTLET_ID: rows[OUTLET_ID], TARGET: pred})
    out.to_csv(args.out, index=False)
    print(f"{len(out)} predictions from {saved['name']} -> {args.out}")
