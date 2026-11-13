import pandas as pd

from bigmart import cli
from bigmart.insights import check_hypotheses, eta_squared, store_summary
from bigmart.preprocessing import clean


def test_hypotheses_table(raw):
    table = check_hypotheses(clean(raw))
    assert len(table) >= 8
    assert {"hypothesis", "evidence", "verdict"} <= set(table.columns)


def test_store_summary_shares_sum_to_one(raw):
    table = store_summary(clean(raw))
    assert len(table) == 10
    assert abs(table["revenue_share"].sum() - 1) < 1e-9
    assert table.index[0] == "OUT027"


def test_eta_squared_bounds(raw):
    df = clean(raw)
    assert 0 <= eta_squared(df, "Item_Type") < eta_squared(df, "Outlet_Type") <= 1


def test_train_then_predict(tmp_path):
    model = tmp_path / "m.pkl"
    cli.train(["--model", "store_price", "--folds", "3", "--out", str(model)])
    out = tmp_path / "sub.csv"
    cli.predict(["--model", str(model), "--out", str(out)])
    sub = pd.read_csv(out)
    assert len(sub) == 5681 and (sub["Item_Outlet_Sales"] >= 0).all()
