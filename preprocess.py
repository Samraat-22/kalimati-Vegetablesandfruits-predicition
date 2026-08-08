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
  df = df.copy()

  # Calendar Features
  df["month"] = df["date"].dt.month
  df["day_of_week"] = df["date"].dt.dayofweek
  df["day_of_year"] = df["date"].dt.dayofyear

  # Seasonal Nepal Indicators
  df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
  df["is_festival_season"] = df["month"].isin([9, 10, 11]).astype(int)

  # Lag & Rolling Features
  g = df.groupby("commodity")[target_col]
  df["price_lag_1d"] = g.shift(1)
  df["price_lag_7d"] = g.shift(7)
  df["rolling_7d_avg"] = (
      g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
  )

  df["category"] = "Vegetable"
  return df.dropna().reset_index(drop=True)