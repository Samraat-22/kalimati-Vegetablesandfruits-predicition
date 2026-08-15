"""
Command-line price predictor for the Kalimati market model.

Usage:
    python predict.py "Tomato Big(Nepali)" 2026-12-25
    python predict.py "Potato Red" 2027-01-15
    python predict.py --list                # show all available commodity names

If you omit the date, it defaults to 30 days after the last known date.
"""

import argparse
import sys
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

MODEL_FILE = "veg_price_model_v2.pkl"
DATA_FILE = "vegetable_clean_enhanced.csv"


def seasonal_baseline(hist, target_date, window_days=7, recent_years=3):
  """Robust (median) historical price around this day-of-year, preferring
  recent years. Used to keep long-range forecasts anchored to reality."""
  doy = target_date.dayofyear
  diff = (hist["date"].dt.dayofyear - doy).abs()
  diff = np.minimum(diff, 365 - diff)
  mask = diff <= window_days

  recent_cutoff = target_date.year - recent_years
  recent_mask = mask & (hist["date"].dt.year >= recent_cutoff)

  if recent_mask.sum() >= 5:
    return hist.loc[recent_mask, "avg_price"].median()
  if mask.sum() > 0:
    return hist.loc[mask, "avg_price"].median()
  return None


def build_feature_row(target_date, working_series, last_row, commodity):
  def lag_val(n):
    idx = len(working_series) - n
    return working_series.iloc[idx] if idx >= 0 else working_series.iloc[0]

  row = {
      "price_lag_1d": lag_val(1),
      "price_lag_7d": lag_val(7),
      "rolling_7d_avg": working_series.tail(7).mean(),
      "is_monsoon": int(target_date.month in [6, 7, 8, 9]),
      "is_festival_season": int(target_date.month in [9, 10, 11]),
      "month": target_date.month,
      "day_of_week": target_date.dayofweek,
      "commodity": last_row.get("commodity", commodity),
      "category": last_row.get("category", "Vegetable"),
      "unit": last_row.get("unit", "KG"),
  }
  return pd.DataFrame([row]).fillna(0)


def predict_for_date(hist, last_row, commodity, target_date, model, encoders,
                      feature_cols, cat_features, blend_full_at=90):
  series = hist.set_index("date")["avg_price"].copy()
  last_date = hist["date"].max()
  price_floor = hist["avg_price"].min() * 0.5
  price_cap = hist["avg_price"].max() * 1.5

  days_ahead = (target_date - last_date).days
  if days_ahead < 1:
    days_ahead = 1
    target_date = last_date + timedelta(days=1)

  pred = None
  current_date = last_date
  for step in range(1, days_ahead + 1):
    current_date = current_date + timedelta(days=1)
    row_df = build_feature_row(current_date, series, last_row, commodity)

    for c in cat_features:
      le = encoders[c]
      val = str(row_df.loc[0, c])
      row_df[c + "_enc"] = le.transform([val])[0] if val in le.classes_ else -1

    raw_pred = model.predict(row_df[feature_cols])[0]
    seasonal = seasonal_baseline(hist, current_date)
    if seasonal is not None:
      w = min(step / blend_full_at, 1.0)
      pred = (1 - w) * raw_pred + w * seasonal
    else:
      pred = raw_pred

    pred = float(np.clip(pred, price_floor, price_cap))
    series.loc[current_date] = pred

  return pred, days_ahead, target_date


def main():
  parser = argparse.ArgumentParser(description="Predict a future Kalimati market price.")
  parser.add_argument("commodity", nargs="?", help='e.g. "Tomato Big(Nepali)"')
  parser.add_argument("date", nargs="?", help="Target date, e.g. 2026-12-25")
  parser.add_argument("--list", action="store_true", help="List all available commodity names")
  args = parser.parse_args()

  bundle = joblib.load(MODEL_FILE)
  model = bundle["model"]
  encoders = bundle["encoders"]
  feature_cols = bundle["feature_cols"]
  cat_features = bundle["cat_features"]

  df = pd.read_csv(DATA_FILE, parse_dates=["date"])

  if args.list or not args.commodity:
    print("Available commodities:")
    for c in sorted(df["commodity"].unique()):
      print(f"  - {c}")
    if not args.commodity:
      sys.exit(0 if args.list else 1)

  commodity = args.commodity
  hist = df[df["commodity"] == commodity].sort_values("date")
  if hist.empty:
    print(f'No data found for commodity "{commodity}". Run with --list to see valid names.')
    sys.exit(1)

  last_row = hist.iloc[-1]
  last_date = last_row["date"]

  if args.date:
    target_date = pd.Timestamp(args.date)
  else:
    target_date = last_date + timedelta(days=30)

  pred, days_ahead, target_date = predict_for_date(
      hist, last_row, commodity, target_date, model, encoders, feature_cols, cat_features
  )

  print(f"\nCommodity:        {commodity}")
  print(f"Last known date:  {last_date.date()}  (price: NPR {last_row['avg_price']:.2f})")
  print(f"Target date:      {target_date.date()}  ({days_ahead} days ahead)")
  print(f"Predicted price:  NPR {pred:.2f} / {last_row.get('unit', 'KG')}")
  print(f"Change:           {pred - last_row['avg_price']:+.2f} NPR")

  if days_ahead > 180:
    print("\n⚠ This is 180+ days out -- treat this as a rough seasonal estimate, not a precise forecast.")
  elif days_ahead > 30:
    print("\n⚠ This is 30+ days out -- accuracy drops noticeably this far ahead.")
  elif days_ahead > 7:
    print("\n⚠ This is a week+ out -- treat it as a rough trend.")


if __name__ == "__main__":
  main()