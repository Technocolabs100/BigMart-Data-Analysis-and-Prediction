# Data dictionary

Source: the Analytics Vidhya **BigMart Sales Prediction** practice problem. 2013 sales of
1,559 products in 10 stores. Both files are in `data/raw/`.

| File | Rows | Columns |
|---|---:|---:|
| `Train.csv` | 8,523 | 12 (with the target) |
| `Test.csv` | 5,681 | 11 |

Every product and every store of `Test.csv` also appears in `Train.csv`.

## Raw columns

| Column | Type | Description | Data problems |
|---|---|---|---|
| `Item_Identifier` | id | product id; prefix FD = food, DR = drinks, NC = non-consumable | |
| `Item_Weight` | float | product weight | 1,463 missing (17%) |
| `Item_Fat_Content` | category | Low Fat / Regular | 5 spellings (`LF`, `low fat`, `reg`, ...); set for non-food products too |
| `Item_Visibility` | float | share of the store's display area given to the product | 526 zeros |
| `Item_Type` | category | 16 product categories (Dairy, Snack Foods, Household, ...) | |
| `Item_MRP` | float | maximum retail (list) price, 31 to 267 | |
| `Outlet_Identifier` | id | store id, OUT010 to OUT049 | |
| `Outlet_Establishment_Year` | int | year the store opened, 1985 to 2009 | |
| `Outlet_Size` | category | Small / Medium / High | 2,410 missing: three whole stores (OUT010, OUT017, OUT045) |
| `Outlet_Location_Type` | category | Tier 1 / 2 / 3 city | |
| `Outlet_Type` | category | Grocery Store, Supermarket Type1 / Type2 / Type3 | |
| `Item_Outlet_Sales` | float | **target**: the product's 2013 sales in that store, 33 to 13,087 | right-skewed |

## Engineered columns (`BigMartPreprocessor`)

| Column | Definition |
|---|---|
| `Item_Category` | Food / Drinks / Non-Consumable from the first two letters of `Item_Identifier` |
| `Item_Fat_Content` | normalised to Low Fat / Regular, and `Non-Edible` for non-consumables |
| `Item_Weight` | missing values filled with the product's weight in other stores (fallback: `Item_Type` mean) |
| `Item_Visibility` | zeros replaced by the product's mean non-zero visibility |
| `Item_Visibility_Ratio` | visibility in this store / the product's mean visibility |
| `Outlet_Size` | missing values filled with the most common size of the same `Outlet_Type` |
| `Outlet_Age` | 2013 - `Outlet_Establishment_Year` |
| `Units_Proxy` (EDA only) | `Item_Outlet_Sales / Item_MRP`, roughly the units sold |
