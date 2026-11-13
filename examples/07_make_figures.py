"""Draw the main EDA figures into examples/output/ (the README's figures, regenerated).

    python examples/07_make_figures.py
"""

from pathlib import Path

from bigmart import plots
from bigmart.data import load_train, split_xy
from bigmart.models import build_model
from bigmart.preprocessing import clean

OUT = Path(__file__).parent / "output"

raw = load_train()
tidy = clean(raw)
model = build_model("store_price").fit(*split_xy(raw))
for name, fig in {
    "missing_values": plots.missing_values(raw),
    "sales_by_store": plots.sales_by_store(tidy),
    "price_vs_sales": plots.price_vs_sales(tidy, model.multipliers_),
    "units_by_product": plots.units_by_product(tidy),
}.items():
    print("saved", plots.save(fig, OUT / f"{name}.png"))
