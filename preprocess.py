import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "commodity",
    "date",
    "unit",
    "min_price",
    "max_price",
    "avg_price",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["date"] = pd.to_datetime(df["date"], errors="coerce")
  df = df.dropna(subset=["date"])

  # Standardize units
  unit_map = {
      "kg": "KG",
      "KG": "KG",
      "1 pc": "PIECE",
      "per piece": "PIECE",
      "doz": "DOZEN",
      "per dozen": "DOZEN",
  }
  df["unit"] = df["unit"].astype(str).str.strip()
  df["unit_clean"] = df["unit"].map(lambda u: unit_map.get(u, u.upper()))

  # Base commodity extraction
  extracted = df["commodity"].str.extract(
      r"^(?P<commodity_base>[^(]+?)\s*(\((?P<variety>[^)]*)\))?$"
  )
  df["commodity_base"] = extracted["commodity_base"].str.strip()
  df["variety"] = extracted["variety"].fillna("Standard").str.strip()

  for col in ["min_price", "max_price", "avg_price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  df = df.dropna(subset=["min_price", "max_price", "avg_price"])
  df = df[(df["min_price"] > 0) & (df["max_price"] >= df["min_price"])]
  df = df.drop_duplicates(subset=["commodity", "date"])
  return df.sort_values(["commodity", "date"]).reset_index(drop=True)


def add_seasonal_and_lags(
    df: pd.DataFrame, target_col="avg_price"
) -> pd.DataFrame:
  """Add calendar + lag/rolling features.

  IMPORTANT: lag/rolling features are computed against a real daily
  calendar per commodity (not row position). If a commodity has a gap
  in its history (e.g. no scraped data for a month), the first rows
  after the gap will correctly get NaN lag features (since "yesterday's
  price" isn't actually known) instead of silently pulling in a price
  from months earlier. Those rows are dropped at the end.
  """
  df = df.copy()
  df["date"] = pd.to_datetime(df["date"])

  pieces = []
  for commodity, g in df.groupby("commodity"):
    g = g.sort_values("date").drop_duplicates(subset="date").set_index("date")
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g_full = g.reindex(full_idx)

    s = g_full[target_col]
    g_full["price_lag_1d"] = s.shift(1)
    g_full["price_lag_7d"] = s.shift(7)
    g_full["rolling_7d_avg"] = s.shift(1).rolling(7, min_periods=7).mean()

    # Keep only rows that had real (scraped) data for this date.
    g_full = g_full.loc[g.index]
    g_full["commodity"] = commodity
    pieces.append(g_full.reset_index().rename(columns={"index": "date"}))

  out = pd.concat(pieces, ignore_index=True)

  # Calendar Features
  out["month"] = out["date"].dt.month
  out["day_of_week"] = out["date"].dt.dayofweek
  out["day_of_year"] = out["date"].dt.dayofyear

  # Seasonal Nepal Indicators
  out["is_monsoon"] = out["month"].isin([6, 7, 8, 9]).astype(int)
  out["is_festival_season"] = out["month"].isin([9, 10, 11]).astype(int)

  out["category"] = "Vegetable"

  needed = [
      "price_lag_1d", "price_lag_7d", "rolling_7d_avg", target_col,
  ]
  return out.dropna(subset=needed).reset_index(drop=True)