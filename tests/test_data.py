from bigmart.data import ITEM_ID, OUTLET_ID, TARGET


def test_shapes(raw, test_rows):
    assert raw.shape == (8523, 12)
    assert test_rows.shape == (5681, 11)
    assert TARGET in raw and TARGET not in test_rows


def test_test_products_and_stores_are_known(raw, test_rows):
    # Justifies the shuffled K-fold evaluation: nothing in Test.csv is unseen.
    assert test_rows[ITEM_ID].isin(raw[ITEM_ID]).all()
    assert test_rows[OUTLET_ID].isin(raw[OUTLET_ID]).all()
