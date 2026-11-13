"""Cross-validate all models (5-fold) and print RMSE, MAE and R2.

Every model is a full pipeline (cleaning -> encoding -> regressor) fitted inside each
fold, so the scores are free of leakage. Takes about two minutes on a CPU.

    python examples/03_compare_models.py
"""

from bigmart.data import load_train, split_xy
from bigmart.evaluate import compare_models

X, y = split_xy(load_train())
results = compare_models(X, y)
print(results[["label", "rmse", "rmse_std", "mae", "r2", "seconds"]].round(3).to_string(index=False))
