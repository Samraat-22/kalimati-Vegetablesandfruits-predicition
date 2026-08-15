import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from preprocess import add_seasonal_and_lags, clean_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# 1. Load Data
df = pd.read_csv("vegetable_clean_enhanced.csv")
df = clean_data(df)
df = add_seasonal_and_lags(df)

CAT_FEATURES = ["commodity", "category", "unit"]
encoders = {}

for col in CAT_FEATURES:
  le = LabelEncoder()
  df[col + "_enc"] = le.fit_transform(df[col].astype(str))
  encoders[col] = le

FEATURE_COLS = [
    # NOTE: min_price / max_price were removed from this list.
    # avg_price == (min_price + max_price) / 2 in this dataset, so
    # including them let the model just do arithmetic instead of
    # learning to forecast -- that's why R^2 was ~1.0 before. On any
    # future date you also won't know that day's min/max in advance,
    # so they can't be used as prediction inputs anyway.
    "price_lag_1d",
    "price_lag_7d",
    "rolling_7d_avg",
    "is_monsoon",
    "is_festival_season",
    "month",
    "day_of_week",
    "commodity_enc",
    "category_enc",
    "unit_enc",
]

# IMPORTANT: split by a GLOBAL date cutoff, not by row position.
# `df` is grouped by commodity then date, so a positional 85/15 split
# mostly separates *different commodities* into train vs test (only
# ~1 commodity actually overlapped) rather than separating past vs
# future dates. That made the old "test" score meaningless for
# judging forecasting ability. Sorting by date and cutting by date
# means test rows are genuinely later in time than every train row,
# which is what actually matters if you plan to predict the future.
df = df.sort_values("date").reset_index(drop=True)
cutoff_date = df["date"].quantile(0.85, interpolation="nearest")
train_mask = df["date"] <= cutoff_date
test_mask = ~train_mask

X = df[FEATURE_COLS]
y = df["avg_price"]

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(
    f"Train: {train_mask.sum()} rows through {df.loc[train_mask, 'date'].max().date()}"
)
print(
    f"Test:  {test_mask.sum()} rows from {df.loc[test_mask, 'date'].min().date()}"
    f" to {df.loc[test_mask, 'date'].max().date()}"
)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42, verbosity=-1
    ),
    "LinearRegression": LinearRegression(),
}

results = {}
fitted_models = {}

for name, m in models.items():
  m.fit(X_train, y_train)
  preds = m.predict(X_test)
  mae = mean_absolute_error(y_test, preds)
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  r2 = r2_score(y_test, preds)
  results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
  fitted_models[name] = m

best_name = min(results, key=lambda k: results[k]["MAE"])
best_model = fitted_models[best_name]

# Save Artifacts
artifacts = {
    "model": best_model,
    "model_name": best_name,
    "encoders": encoders,
    "feature_cols": FEATURE_COLS,
    "cat_features": CAT_FEATURES,
    "results": results,
    "commodity_choices": sorted(df["commodity"].unique().tolist()),
}

joblib.dump(artifacts, "veg_price_model_v2.pkl")
print(
    f" Best Model '{best_name}' trained and saved to veg_price_model_v2.pkl!"
)
