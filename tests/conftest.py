import pytest

from bigmart.data import load_test, load_train, split_xy


@pytest.fixture(scope="session")
def raw():
    return load_train()


@pytest.fixture(scope="session")
def xy(raw):
    return split_xy(raw)


@pytest.fixture(scope="session")
def test_rows():
    return load_test()
