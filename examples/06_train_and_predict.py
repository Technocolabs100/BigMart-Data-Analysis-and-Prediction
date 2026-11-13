"""Train the final model on all of Train.csv and predict Test.csv (a submission file).

Uses the same entry points as the `bigmart-train` and `bigmart-predict` commands.

    python examples/06_train_and_predict.py
"""

from pathlib import Path

import pandas as pd

from bigmart import cli

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

cli.train(["--model", "store_price", "--out", str(OUT / "model.pkl")])
cli.predict(["--model", str(OUT / "model.pkl"), "--out", str(OUT / "submission.csv")])
print(pd.read_csv(OUT / "submission.csv").head())
