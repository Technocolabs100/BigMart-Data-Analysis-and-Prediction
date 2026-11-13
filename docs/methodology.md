# Methodology

## Evaluation

- **Split.** Every product and every store in `Test.csv` is also in `Train.csv`, so the
  task is interpolation inside a known product x store grid. Shuffled 5-fold
  cross-validation (`KFold(5, shuffle=True, random_state=42)`) reproduces that setting.
  Plots that need one set of predictions use a fixed 75/25 split.
- **Metrics.** RMSE (the competition metric, in sales units) is the main score; MAE and
  R2 are reported next to it, each as mean +- std over the folds.
- **No leakage.** Each model is a single scikit-learn estimator,
  `BigMartPreprocessor -> ColumnTransformer -> regressor`, refitted inside every fold. The
  values the cleaning step learns (per-product weights and visibilities, per-type store
  sizes, scaler statistics, encoder categories) come from the training fold only.

## Models

| Name | Encoding | Notes |
|---|---|---|
| Mean baseline | - | predicts the training mean |
| Linear regression, Ridge | one-hot + standardised numbers | Ridge alpha = 1 |
| Random forest | ordinal codes | 300 trees, depth 8, min 20 rows per leaf |
| Gradient boosting | ordinal codes | 300 trees, lr 0.03, depth 4, subsample 0.8 |
| XGBoost | ordinal codes | 400 trees, lr 0.03, depth 4, column + row subsampling |
| **Store x price** | raw `Outlet_Identifier`, `Item_MRP` | `sales = k[store] * MRP`, `k` fitted by least squares per store (10 parameters) |

The tree models were given conservative, regularised settings; tuning them further moved
CV RMSE by only a few units, since they are already at the noise floor (see Results).

## Experiments that did not help

| Change | CV RMSE (gradient boosting) |
|---|---:|
| all features (default) | 1,084.9 |
| + target-encoded `Item_Identifier` (cross-fitted) | 1,084.3 |
| only `Outlet_Identifier`, `Outlet_Type`, `Item_MRP` | 1,083.3 |
| predict units (`sales / MRP`) then multiply by MRP | 1,081.5 |
| average of gradient boosting and Store x price | 1,074.8 |
| **Store x price alone** | **1,071.6** |

Fitting on `log1p(sales)` (`build_model(..., log_target=True)`) lowers MAE (789 vs 839 for
linear regression) but raises RMSE (1,145 vs 1,132), because `expm1` of a mean in log space
is closer to a median and underestimates the mean. RMSE is the target metric, so the
default is the raw scale.

## What changed from the original notebook

The first version of this project (kept as `notebooks/original_colab_notebook.ipynb`)
reached R2 0.56-0.60 on one validation split. The rebuild keeps its hypotheses and its
cleaning ideas and fixes these issues:

| Original notebook | Rebuild |
|---|---|
| the cell that created `train_X` / `val_X` was missing, so the notebook could not run top to bottom | one pipeline, runs end to end (`scripts/make_report.py`, notebook executed) |
| `StandardScaler` and the mode imputation refitted on the validation set | every statistic learned on the training fold only |
| `Item_Weight` filled with the global mode | filled from the same product in other stores |
| `Item_Visibility` zeros kept | replaced by the product's mean visibility |
| `Outlet_Identifier` and `Item_Type` dropped | kept; the store identity is the strongest signal after price |
| one train/validation split | 5-fold CV with mean +- std |
| model chosen by R2 on the validation split | models compared on RMSE, MAE and R2 |
| hypotheses listed but not tested | each hypothesis tested with numbers (`bigmart.insights`) |
