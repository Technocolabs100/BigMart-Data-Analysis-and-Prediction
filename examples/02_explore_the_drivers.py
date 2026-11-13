"""Answer the project's business questions: what drives sales at BigMart?

Prints the store table (type, size, tier, units per product, revenue share) and the
hypothesis checks with their evidence and verdicts.

    python examples/02_explore_the_drivers.py
"""

import pandas as pd

from bigmart.data import load_train
from bigmart.insights import check_hypotheses, store_summary
from bigmart.preprocessing import clean

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

tidy = clean(load_train())
print(store_summary(tidy)[["type", "size", "tier", "mean_sales", "mean_units", "revenue_share"]].round(2))
print()
for row in check_hypotheses(tidy).itertuples():
    print(f"- {row.hypothesis}\n    {row.evidence}\n    -> {row.verdict}")
