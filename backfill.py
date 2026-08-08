from datetime import datetime, timedelta
import os
from bs4 import BeautifulSoup
import pandas as pd
import requests

CSV_FILE = "vegetable_clean.csv"


def scrape_date(target_date_str):
  url = f"https://kalimatimarket.gov.np/price/date/{target_date_str}"
  try:
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
      return []
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table", {"id": "tblPrice"}) or soup.find("table")
    if not table:
      return []

    rows = table.find_all("tr")
    new_rows = []
    for r in rows:
      cols = [td.text.strip() for td in r.find_all(["td", "th"])]
      if len(cols) >= 5:
        try:
          if "Commodity" in cols[0] or "१" in cols[0] or "कृषि" in cols[0]:
            continue
          new_rows.append({
              "date": target_date_str,
              "commodity": cols[0],
              "unit": cols[1],
              "min_price": float(
                  cols[2].replace("Rs", "").replace("रू", "").strip()
              ),
              "max_price": float(
                  cols[3].replace("Rs", "").replace("रू", "").strip()
              ),
              "avg_price": float(
                  cols[4].replace("Rs", "").replace("रू", "").strip()
              ),
          })
        except ValueError:
          continue
    return new_rows
  except Exception as e:
    return []


def run_backfill():
  start_date = datetime(2022, 4, 19)
  end_date = datetime.now()

  current = start_date
  all_data = []

  print(
      f"Starting historical backfill from {start_date.strftime('%Y-%m-%d')} to"
      f" {end_date.strftime('%Y-%m-%d')}..."
  )

  while current <= end_date:
    d_str = current.strftime("%Y-%m-%d")
    print(f"Fetching: {d_str}")
    daily_records = scrape_date(d_str)
    if daily_records:
      all_data.extend(daily_records)
    current += timedelta(days=1)

  if all_data:
    df_new = pd.DataFrame(all_data)
    if os.path.exists(CSV_FILE):
      df_existing = pd.read_csv(CSV_FILE)
      df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
      df_combined = df_new

    df_combined = df_combined.drop_duplicates(
        subset=["date", "commodity"], keep="last"
    )
    df_combined["date"] = pd.to_datetime(df_combined["date"])
    df_combined = df_combined.sort_values(["commodity", "date"])
    df_combined.to_csv(CSV_FILE, index=False)
    print("Backfill complete! Saved to vegetable_clean.csv")
  else:
    print("No records retrieved during backfill.")


if __name__ == "__main__":
  run_backfill()