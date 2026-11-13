# Examples

Short scripts that use the `bigmart` package from Python, in the order you would use it.
Run them from the repository root after `pip install -e .`; each one runs on a laptop CPU
in seconds (03 takes about two minutes). Files they write go to `examples/output/`.

| Script | What it shows |
|---|---|
| [`01_load_and_clean.py`](01_load_and_clean.py) | the raw data problems and the cleaned, feature-engineered frame |
| [`02_explore_the_drivers.py`](02_explore_the_drivers.py) | store table and the hypothesis checks: what drives sales |
| [`03_compare_models.py`](03_compare_models.py) | 5-fold CV of seven models: RMSE, MAE, R2 |
| [`04_store_price_model.py`](04_store_price_model.py) | the 10-number `k[store] x price` model, read as units per store |
| [`05_feature_importance.py`](05_feature_importance.py) | permutation importance of a gradient-boosting model |
| [`06_train_and_predict.py`](06_train_and_predict.py) | train on Train.csv, write a submission for Test.csv |
| [`07_make_figures.py`](07_make_figures.py) | regenerate the main EDA figures |
