import numpy as np
import pytest

from bigmart.evaluate import compare_models, regression_metrics
from bigmart.models import MODEL_NAMES, StorePriceRegressor, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name, xy, test_rows):
    X, y = xy
    model = build_model(name, n_jobs=1).fit(X.iloc[:2000], y.iloc[:2000])
    pred = model.predict(test_rows.iloc[:50])
    assert pred.shape == (50,)
    assert np.isfinite(pred).all()


def test_store_price_recovers_known_multipliers(xy):
    X, _ = xy
    k = {store: i + 1.0 for i, store in enumerate(sorted(X["Outlet_Identifier"].unique()))}
    y = X["Outlet_Identifier"].map(k) * X["Item_MRP"]
    model = StorePriceRegressor().fit(X, y)
    for store, value in k.items():
        assert model.multipliers_[store] == pytest.approx(value)


def test_models_beat_the_mean_baseline(xy):
    X, y = xy
    res = compare_models(X, y, names=["mean_baseline", "store_price", "xgboost"], folds=3, n_jobs=1)
    rmse = res.set_index("model")["rmse"]
    assert rmse["store_price"] < 0.7 * rmse["mean_baseline"]
    assert rmse["xgboost"] < 0.7 * rmse["mean_baseline"]
    assert res.set_index("model").loc["store_price", "r2"] > 0.58


def test_log_target_option(xy):
    X, y = xy
    model = build_model("ridge", log_target=True).fit(X, y)
    assert (model.predict(X.iloc[:20]) > 0).all()


def test_regression_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}
