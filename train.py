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
    "min_price",
    "max_price",
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

X = df[FEATURE_COLS]
y = df["avg_price"]

# Time Split (85% Train / 15% Test)
split_idx = int(len(df) * 0.85)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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
    f"✅ Best Model '{best_name}' trained and saved to veg_price_model_v2.pkl!"
)