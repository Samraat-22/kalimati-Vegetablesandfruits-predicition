import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

FRUITS = {
    "Amla",
    "Apple",
    "Avocado",
    "Banana",
    "Grapes",
    "Guava",
    "Jack Fruit",
    "Kinnow",
    "Kiwi",
    "Lemon",
    "Lime",
    "Litchi",
    "Mandarin",
    "Mango",
    "Mombin",
    "Orange",
    "Papaya",
    "Pear",
    "Pineapple",
    "Pomegranate",
    "Sarifa",
    "Strawberry",
    "Sweet Orange",
    "Tamarind",
    "Water Melon",
}

OTHER = {"Fish Fresh", "Tofu", "Gundruk", "Okara"}

MIN_ROWS_FOR_PREDICTION = 20


def classify(commodity_base: str) -> str:
    if commodity_base in FRUITS:
        return "Fruit"
    if commodity_base in OTHER:
        return "Other"
    return "Vegetable"


def build_features(item_df: pd.DataFrame) -> pd.DataFrame:
    """Turn a single-commodity price history into a feature table.

    Each row's features describe what was known *before* that day's
    price, so a model trained on this can be reused to forecast the
    next unseen day.
    """
    data = item_df[["date", "avg_price"]].copy()
    data = data.sort_values("date").reset_index(drop=True)

    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["day_of_year"] = data["date"].dt.dayofyear

    # Lag features: yesterday's price, and short-term rolling averages
    data["lag_1"] = data["avg_price"].shift(1)
    data["lag_7"] = data["avg_price"].shift(7)
    data["rolling_mean_7"] = data["avg_price"].shift(1).rolling(7).mean()
    data["rolling_mean_14"] = data["avg_price"].shift(1).rolling(14).mean()

    return data


def train_price_model(item_df: pd.DataFrame):
    """Train a RandomForest on one commodity's price history.

    Returns (model, feature_cols, mae) or (None, None, None) if there
    isn't enough history to train on.
    """
    data = build_features(item_df)

    feature_cols = [
        "day_of_week",
        "month",
        "day_of_year",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_mean_14",
    ]

    train_data = data.dropna(subset=feature_cols + ["avg_price"])

    if len(train_data) < MIN_ROWS_FOR_PREDICTION:
        return None, None, None

    X = train_data[feature_cols]
    y = train_data["avg_price"]

    # Hold out the most recent 10 rows to sanity-check accuracy
    split = max(len(X) - 10, int(len(X) * 0.85))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    mae = None
    if len(X_test) > 0:
        preds_test = model.predict(X_test)
        mae = (preds_test - y_test).abs().mean()

    # Refit on ALL available data before forecasting forward, so the
    # prediction uses every known data point, not just the training split.
    model.fit(X, y)

    return model, feature_cols, mae


def predict_price_for_date(item_df: pd.DataFrame, target_date: pd.Timestamp):
    """Forecast a commodity's price for any future date.

    Works recursively day-by-day from the last known date up to
    target_date, feeding each day's prediction back in as the "lag_1"
    input for the next day. Accuracy naturally degrades the further
    target_date is from the last known data point.

    Returns (predicted_price, mean_abs_error_on_recent_data, days_ahead)
    or (None, None, None) if there isn't enough history to train on.
    """
    model, feature_cols, mae = train_price_model(item_df)
    if model is None:
        return None, None, None

    data = item_df[["date", "avg_price"]].sort_values("date").reset_index(drop=True)
    last_date = data["date"].max()

    days_ahead = (target_date - last_date).days
    if days_ahead < 1:
        days_ahead = 1
        target_date = last_date + pd.Timedelta(days=1)

    # Recursively step forward one day at a time, appending each
    # prediction to the price series so the next step's lag/rolling
    # features are based on the freshest (predicted) values.
    working_prices = data["avg_price"].tolist()
    working_dates = data["date"].tolist()

    predicted_price = None
    for _ in range(days_ahead):
        next_date = working_dates[-1] + pd.Timedelta(days=1)
        series = pd.Series(working_prices)

        next_features = pd.DataFrame(
            [
                {
                    "day_of_week": next_date.dayofweek,
                    "month": next_date.month,
                    "day_of_year": next_date.dayofyear,
                    "lag_1": series.iloc[-1],
                    "lag_7": series.iloc[-7] if len(series) >= 7 else series.iloc[-1],
                    "rolling_mean_7": series.tail(7).mean(),
                    "rolling_mean_14": series.tail(14).mean(),
                }
            ]
        )

        predicted_price = model.predict(next_features[feature_cols])[0]
        working_prices.append(predicted_price)
        working_dates.append(next_date)

    return predicted_price, mae, days_ahead


def main():
    st.title("Fruit vs Vegetable Price Analysis")

    # IMPORTANT: this file must be in the SAME folder as this script,
    # and committed to your GitHub repo.
    file_path = "vegetable.csv"

    columns = [
        "commodity",
        "date",
        "unit",
        "min_price",
        "max_price",
        "avg_price",
    ]

    try:
        df = pd.read_csv(
            file_path,
            header=None,
            names=columns,
            parse_dates=["date"],
            skipinitialspace=True,
        )
    except FileNotFoundError:
        st.error(
            f"Could not find '{file_path}'. Make sure vegetable.csv is uploaded "
            "to the same GitHub folder as this script."
        )
        st.stop()

    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df = df.dropna(subset=["avg_price", "date"])

    df["commodity_base"] = (
        df["commodity"].astype(str).str.split("(").str[0].str.strip()
    )

    df["category"] = df["commodity_base"].apply(classify)
    df = df[df["category"] != "Other"]

    # ------------------------------------------------------------------
    # Commodity search (URL-shareable)
    # ------------------------------------------------------------------
    st.subheader("Search a specific item")

    commodity_list = sorted(df["commodity"].dropna().unique())

    # Read commodity from the URL if present (e.g. ?commodity=Tomato Big(Nepali))
    query_commodity = st.query_params.get("commodity", None)
    default_index = (
        commodity_list.index(query_commodity)
        if query_commodity in commodity_list
        else 0
    )

    selected_commodity = st.selectbox(
        "Choose a commodity",
        commodity_list,
        index=default_index,
    )

    # Keep the URL in sync so the link can be copied and shared
    st.query_params["commodity"] = selected_commodity

    item_df = df[df["commodity"] == selected_commodity].sort_values("date")

    if not item_df.empty:
        min_date = item_df["date"].min().date()
        max_date = item_df["date"].max().date()

        st.markdown(f"**Price Trend from {min_date} to {max_date}**")

        fig_item, ax_item = plt.subplots(figsize=(10, 4))
        ax_item.plot(item_df["date"], item_df["avg_price"], color="#1f77b4")
        ax_item.set_xlabel("Date")
        ax_item.set_ylabel("Average Price (NPR)")
        ax_item.set_title(selected_commodity)
        st.pyplot(fig_item)

        summary = pd.DataFrame(
            {
                "commodity": [selected_commodity],
                "Min_Price": [item_df["avg_price"].min()],
                "Max_Price": [item_df["avg_price"].max()],
                "Average_Price": [item_df["avg_price"].mean()],
                "Latest_Price": [item_df["avg_price"].iloc[-1]],
            }
        )
        st.markdown("### 📊 Summary Statistics")
        st.dataframe(summary, hide_index=True)

        # --------------------------------------------------------------
        # Price prediction for a chosen future date
        # --------------------------------------------------------------
        st.markdown("### 🔮 Price Prediction")

        last_known_date = item_df["date"].max().date()
        default_target = last_known_date + pd.Timedelta(days=1)

        target_date = st.date_input(
            "Predict price for this date",
            value=default_target,
            min_value=default_target,
        )
        target_ts = pd.Timestamp(target_date)

        predicted_price, mae, days_ahead = predict_price_for_date(
            item_df, target_ts
        )

        if predicted_price is None:
            st.warning(
                "Not enough historical data for this item yet to make a "
                f"reliable prediction (need at least {MIN_ROWS_FOR_PREDICTION} "
                "recorded days)."
            )
        else:
            last_price = item_df["avg_price"].iloc[-1]
            change = predicted_price - last_price
            change_pct = (change / last_price * 100) if last_price else 0

            pred_col1, pred_col2 = st.columns(2)
            pred_col1.metric(
                f"Predicted price for {target_date} (NPR)",
                f"{predicted_price:.2f}",
                delta=f"{change:+.2f} ({change_pct:+.1f}%) vs last known price",
            )
            if mae is not None:
                pred_col2.metric("Model's typical 1-day error (NPR)", f"±{mae:.2f}")

            st.caption(
                "This is a simple estimate from a RandomForest model trained "
                "on this item's own price history (day of week, month, and "
                "recent price trends). It is not financial advice — actual "
                "market prices can move for reasons the model can't see."
            )

            if days_ahead > 7:
                st.warning(
                    f"This forecast is {days_ahead} days beyond the last "
                    "recorded price. The model predicts day by day and "
                    "feeds each guess into the next, so errors compound — "
                    "treat predictions this far out as a rough trend, not "
                    "a precise number."
                )

        st.info(
            "Copy the link from your browser's address bar to share this "
            f"exact view of **{selected_commodity}** with someone."
        )
    else:
        st.warning("No data found for this commodity.")

    st.divider()

    # ------------------------------------------------------------------
    # Overall Fruit vs Vegetable comparison (original charts)
    # ------------------------------------------------------------------
    st.subheader("Summary statistics")
    st.dataframe(df.groupby("category")["avg_price"].describe())

    daily_avg = df.groupby(["date", "category"])["avg_price"].mean().unstack()

    # Make sure both columns exist even if one category is briefly missing
    # on some dates, and force everything to numeric so plotting never
    # breaks on stray non-numeric values.
    for col in ["Fruit", "Vegetable"]:
        if col not in daily_avg.columns:
            daily_avg[col] = pd.NA
        daily_avg[col] = pd.to_numeric(daily_avg[col], errors="coerce")

    daily_avg["difference"] = daily_avg["Fruit"] - daily_avg["Vegetable"]

    st.subheader("Overall averages")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fruit avg (NPR)", f"{daily_avg['Fruit'].mean():.2f}")
    col2.metric("Vegetable avg (NPR)", f"{daily_avg['Vegetable'].mean():.2f}")
    col3.metric("Difference (NPR)", f"{daily_avg['difference'].mean():.2f}")

    daily_avg.to_csv("fruit_vs_vegetable_daily.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    means = df.groupby("category")["avg_price"].mean()
    axes[0, 0].bar(means.index, means.values, color=["#e07a5f", "#3d9970"])
    axes[0, 0].set_title("Average price: Fruit vs Vegetable")
    axes[0, 0].set_ylabel("Avg price (NPR)")
    for i, v in enumerate(means.values):
        axes[0, 0].text(i, v + 2, f"{v:.1f}", ha="center")

    df.boxplot(
        column="avg_price", by="category", ax=axes[0, 1], showfliers=False
    )
    axes[0, 1].set_title("Price distribution by category")
    axes[0, 1].set_ylabel("Avg price (NPR)")
    axes[0, 1].set_xlabel("")
    plt.suptitle("")

    daily_avg[["Fruit", "Vegetable"]].plot(ax=axes[1, 0])
    axes[1, 0].set_title("Daily average price over time")
    axes[1, 0].set_ylabel("Avg price (NPR)")

    # Drop any rows where "difference" ended up NaN so matplotlib never
    # chokes on a non-numeric / masked value.
    diff_data = daily_avg["difference"].dropna()
    axes[1, 1].plot(diff_data.index, diff_data.values, color="purple")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Price difference over time (Fruit − Vegetable)")
    axes[1, 1].set_ylabel("NPR difference")

    plt.tight_layout()

    st.subheader("Charts")
    st.pyplot(fig)

    plt.savefig("fruit_vs_vegetable_price.png", dpi=110)


if __name__ == "__main__":
    main()