"""
Faster version: fetches days in parallel instead of one at a time.
"""
from datetime import datetime, timedelta
from io import StringIO
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

CSV_FILE = "vegetable_clean_enhanced.csv"
RAW_BASE = "https://raw.githubusercontent.com/ErKiran/kalimati/master/data/csv"


def fetch_day(target_date):
  url = f"{RAW_BASE}/{target_date.year}/{target_date.month:02d}/{target_date.day:02d}.csv"
  try:
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
      return target_date, None
    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
      return target_date, None
    df = df.rename(columns={
        "Date": "date", "Product": "commodity", "Unit": "unit",
        "Min Price": "min_price", "Max Price": "max_price", "Avg Price": "avg_price",
    })
    return target_date, df[["date", "commodity", "unit", "min_price", "max_price", "avg_price"]]
  except requests.RequestException:
    return target_date, None


def update_dataset_range(start_date, end_date, max_workers=20):
  dates = []
  d = start_date
  while d <= end_date:
    dates.append(d)
    d += timedelta(days=1)

  all_new = []
  done = 0
  with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(fetch_day, d): d for d in dates}
    for fut in as_completed(futures):
      d, df = fut.result()
      done += 1
      if df is not None:
        all_new.append(df)
      if done % 50 == 0 or done == len(dates):
        print(f"Progress: {done}/{len(dates)} days checked")

  if not all_new:
    print("No new data retrieved.")
    return

  df_new = pd.concat(all_new, ignore_index=True)
  df_new["date"] = pd.to_datetime(df_new["date"])

  if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE, parse_dates=["date"])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
  else:
    df_combined = df_new

  df_combined = df_combined.drop_duplicates(subset=["date", "commodity"], keep="last")
  df_combined = df_combined.sort_values(["commodity", "date"])
  df_combined.to_csv(CSV_FILE, index=False)
  print(f"Saved. New date range: {df_combined['date'].min()} to {df_combined['date'].max()}")


if __name__ == "__main__":
  start = datetime(2023, 5, 16)
  end = datetime.now()
  update_dataset_range(start, end)