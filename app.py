from datetime import timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Kalimati Vegetable Market Price Predictor",
    page_icon="🥑",
    layout="wide",
)

lang = st.sidebar.radio("🌐 Language / भाषा", ["English", "नेपाली"])

trans = {
    "English": {
        "title": "🥕 Kalimati Vegetable Market — Price Predictor",
        "subtitle": (
            "Forecasts future average price (NPR/unit) for Kalimati market"
            " commodities using machine learning."
        ),
        "tab1": "Predict",
        "tab2": "Price History",
        "tab3": "Model Performance",
        "commodity": "Select Commodity",
        "last_date": "Last known date",
        "last_price": "Last known avg price",
        "prediction_header": "Prediction Result",
        "predicted_price": "Predicted avg price for",
        "change": "Change vs last known price",
        "unit": "Unit",
        "predict_date": "Predict price for this date",
        "predict_date_help": (
            "Predictions further from the last recorded date are"
            " increasingly rough estimates, not precise forecasts."
        ),
        "warn_week": "This forecast is {n} days out — treat it as a rough trend.",
        "warn_month": "This forecast is {n} days out — accuracy drops noticeably this far ahead.",
        "warn_far": (
            "This forecast is {n} days out. At this range the model is"
            " essentially extrapolating a smoothed seasonal pattern — treat"
            " it as a very loose guess, not a precise forecast."
        ),
    },
    "नेपाली": {
        "title": "🥕 कालिमाटी तरकारी बजार — मूल्य पूर्वानुमान",
        "subtitle": (
            "कालिमाटी बजारका वस्तुहरूको भविष्यको औसत मूल्य (रू/इकाई) मसिन"
            " लर्निङ प्रयोग गरी अनुमान गर्दछ।"
        ),
        "tab1": "अनुमान",
        "tab2": "मूल्य इतिहास",
        "tab3": "मोडेल कार्यसम्पादन",
        "commodity": "वस्तु छनोट गर्नुहोस्",
        "last_date": "पछिल्लो मिति",
        "last_price": "पछिल्लो औसत मूल्य",
        "prediction_header": "पूर्वानुमान नतिजा",
        "predicted_price": "अनुमानित औसत मूल्य",
        "change": "पछिल्लो मूल्यको तुलनामा परिवर्तन",
        "unit": "इकाई",
        "predict_date": "यो मितिको मूल्य अनुमान गर्नुहोस्",
        "predict_date_help": (
            "पछिल्लो रेकर्ड मितिबाट जति टाढा जान्छ, अनुमान त्यति नै अनुमानित"
            " (कम सटीक) हुन्छ।"
        ),
        "warn_week": "यो पूर्वानुमान {n} दिन पर हो — यसलाई एक अनुमानित प्रवृत्तिको रूपमा लिनुहोस्।",
        "warn_month": "यो पूर्वानुमान {n} दिन पर हो — यति टाढा गएपछि शुद्धता उल्लेखनीय रूपमा घट्छ।",
        "warn_far": (
            "यो पूर्वानुमान {n} दिन पर हो। यति टाढाको अनुमान वास्तवमा एक"
            " सहज मौसमी प्रवृत्तिको अनुमान मात्र हो — यसलाई एकदमै खुकुलो"
            " अनुमानको रूपमा मात्र लिनुहोस्।"
        ),
    },
}
t = trans[lang]


@st.cache_resource
def load_model():
  return joblib.load("veg_price_model_v2.pkl")


@st.cache_data
def load_data():
  df = pd.read_csv("vegetable_clean_enhanced.csv", parse_dates=["date"])
  return df


def build_feature_row(target_date, working_series, last_row, commodity):
  """Build one feature row for target_date, using working_series (a
  date-indexed price Series that includes real history PLUS any
  predictions already made for earlier future dates)."""

  def lag_val(n):
    idx = len(working_series) - n
    return working_series.iloc[idx] if idx >= 0 else working_series.iloc[0]

  row = {
      "min_price": last_row.get("min_price", last_row["avg_price"] * 0.9),
      "max_price": last_row.get("max_price", last_row["avg_price"] * 1.1),
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


def predict_for_date(hist, last_row, commodity, target_date, model, encoders, feature_cols, cat_features):
  """Recursively forecast forward, one day at a time, from the last known
  date up to target_date. Each step's prediction becomes an input for the
  next step's lag/rolling features."""

  series = hist.set_index("date")["avg_price"].copy()
  last_date = hist["date"].max()

  days_ahead = (target_date - last_date).days
  if days_ahead < 1:
    days_ahead = 1
    target_date = last_date + timedelta(days=1)

  pred = None
  current_date = last_date
  for _ in range(days_ahead):
    current_date = current_date + timedelta(days=1)

    row_df = build_feature_row(current_date, series, last_row, commodity)

    for c in cat_features:
      le = encoders[c]
      val = str(row_df.loc[0, c])
      row_df[c + "_enc"] = le.transform([val])[0] if val in le.classes_ else -1

    X_input = row_df[feature_cols]
    pred = model.predict(X_input)[0]

    # Feed this prediction back in as if it were the newest known price
    series.loc[current_date] = pred

  return pred, days_ahead, target_date


try:
  bundle = load_model()
  model = bundle["model"]
  encoders = bundle["encoders"]
  feature_cols = bundle["feature_cols"]
  cat_features = bundle["cat_features"]
  results = bundle.get("results", {})
  df = load_data()
  data_loaded = True
except Exception as e:
  data_loaded = False
  st.error(f"Error loading model or data. Details: {e}")

st.title(t["title"])
st.caption(t["subtitle"])

if data_loaded:
  tab1, tab2, tab3 = st.tabs([
      f"🔮 {t['tab1']}",
      f"📈 {t['tab2']}",
      f"📊 {t['tab3']}",
  ])

  with tab1:
    st.subheader(t["tab1"])
    commodities = sorted(df["commodity"].unique().tolist())
    default_idx = (
        commodities.index("Tomato Big(Nepali)")
        if "Tomato Big(Nepali)" in commodities
        else 0
    )
    commodity = st.selectbox(t["commodity"], commodities, index=default_idx)

    hist = df[df["commodity"] == commodity].sort_values("date")
    if hist.empty:
      st.warning("No historical data available for this commodity.")
    else:
      st.markdown("### 📅 Day-by-Day Market Price View")
      hist_desc = hist.sort_values("date", ascending=False)
      latest_entry = hist_desc.iloc[0]

      delta_val = 0.0
      if len(hist_desc) > 1:
        delta_val = latest_entry["avg_price"] - hist_desc.iloc[1]["avg_price"]

      st.metric(
          label=f"Latest Price ({latest_entry['date'].strftime('%Y-%m-%d')})",
          value=f"NPR {latest_entry['avg_price']:.2f} / {latest_entry['unit']}",
          delta=f"{delta_val:+.2f} NPR from previous record",
      )

      st.markdown("### 📋 Recent Daily Records")
      st.dataframe(
          hist_desc[["date", "min_price", "avg_price", "max_price"]].head(14),
          width='stretch',
      )

      last_row = hist.iloc[-1]
      last_date = last_row["date"]

      default_target = (last_date + timedelta(days=1)).date()
      max_target = (last_date + timedelta(days=365)).date()

      target_date_input = st.date_input(
          t["predict_date"],
          value=default_target,
          min_value=default_target,
          max_value=max_target,
          help=t["predict_date_help"],
      )
      target_date = pd.Timestamp(target_date_input)

      pred, days_ahead, target_date = predict_for_date(
          hist, last_row, commodity, target_date,
          model, encoders, feature_cols, cat_features,
      )

      st.markdown(f"### {t['prediction_header']}")
      pc1, pc2, pc3 = st.columns(3)
      pc1.metric(
          f"{t['predicted_price']} ({target_date.strftime('%Y-%m-%d')})",
          f"NPR {pred:.2f}",
      )
      delta = pred - last_row["avg_price"]
      pc2.metric(t["change"], f"{delta:+.2f}", delta=f"{delta:+.2f}")
      pc3.metric(t["unit"], last_row.get("unit", "KG"))

      if days_ahead > 180:
        st.error(t["warn_far"].format(n=days_ahead))
      elif days_ahead > 30:
        st.warning(t["warn_month"].format(n=days_ahead))
      elif days_ahead > 7:
        st.warning(t["warn_week"].format(n=days_ahead))

      plot_df = hist.tail(30)[["date", "avg_price"]].copy()
      pred_point = pd.DataFrame({"date": [target_date], "avg_price": [pred]})
      fig = go.Figure()
      fig.add_trace(
          go.Scatter(
              x=plot_df["date"],
              y=plot_df["avg_price"],
              mode="lines+markers",
              name="Actual",
          )
      )
      fig.add_trace(
          go.Scatter(
              x=pred_point["date"],
              y=pred_point["avg_price"],
              mode="markers",
              marker=dict(size=14, symbol="star", color="red"),
              name="Predicted",
          )
      )
      fig.update_layout(
          title=f"{commodity}: Last 30 Days + Forecast for {target_date.strftime('%Y-%m-%d')}",
          xaxis_title="Date",
          yaxis_title="Price (NPR)",
      )
      st.plotly_chart(fig, use_container_width=True)

  with tab2:
    st.subheader(t["tab2"])
    df["date"] = pd.to_datetime(df["date"])

    commodities_multi = st.multiselect(
        "Select Commodities to Compare", commodities, default=[commodity]
    )

    min_df_date = df["date"].min().date()
    max_df_date = df["date"].max().date()

    col_date1, col_date2 = st.columns(2)
    with col_date1:
      start_date = st.date_input(
          "Start Date",
          value=min_df_date,
          min_value=min_df_date,
          max_value=max_df_date,
      )
    with col_date2:
      end_date = st.date_input(
          "End Date",
          value=max_df_date,
          min_value=min_df_date,
          max_value=max_df_date,
      )

    if commodities_multi:
      filtered_df = df[
          (df["commodity"].isin(commodities_multi))
          & (df["date"] >= pd.to_datetime(start_date))
          & (df["date"] <= pd.to_datetime(end_date))
      ].sort_values("date")

      if filtered_df.empty:
        st.warning("No data found for the selected commodity and date range.")
      else:
        fig2 = px.line(
            filtered_df,
            x="date",
            y="avg_price",
            color="commodity",
            title=(
                f"Price Trend from {start_date.strftime('%Y-%m-%d')} to"
                f" {end_date.strftime('%Y-%m-%d')}"
            ),
            labels={"avg_price": "Average Price (NPR)", "date": "Date"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 📊 Summary Statistics")
        stats = (
            filtered_df.groupby("commodity")["avg_price"]
            .agg(
                Min_Price="min",
                Max_Price="max",
                Average_Price="mean",
                Latest_Price="last",
            )
            .reset_index()
        )
        st.dataframe(stats, use_container_width=True)

  with tab3:
    st.subheader(t["tab3"])
    if results:
      res_df = pd.DataFrame(results).T.rename_axis("Model").reset_index()
      st.dataframe(res_df, use_container_width=True)
      fig3 = px.bar(
          res_df, x="Model", y="MAE", title="Mean Absolute Error (Lower is Better)"
      )
      st.plotly_chart(fig3, use_container_width=True)
    else:
      st.info("Model evaluation metrics recorded during training.")

      with tab2:

        st.subheader(t["tab2"])
    df["date"] = pd.to_datetime(df["date"])

    commodities_multi = st.multiselect(
        "Select Commodities to Compare",
        commodities,
        default=[commodity],
        key="history_commodities",
    )

    min_df_date = df["date"].min().date()
    max_df_date = df["date"].max().date()

    # Safety net: if data is somehow empty or corrupted, don't let the
    # date widgets crash the app.
    if pd.isna(min_df_date) or pd.isna(max_df_date):
      st.error("Could not determine a valid date range from the data.")
    else:
      col_date1, col_date2 = st.columns(2)
      with col_date1:
        start_date = st.date_input(
            "Start Date",
            value=min_df_date,
            min_value=min_df_date,
            max_value=max_df_date,
            format="YYYY/MM/DD",
            key="history_start_date",
        )
      with col_date2:
        end_date = st.date_input(
            "End Date",
            value=max_df_date,
            min_value=min_df_date,
            max_value=max_df_date,
            format="YYYY/MM/DD",
            key="history_end_date",
        )

      if start_date > end_date:
        st.error("Start Date must be before End Date.")
      elif commodities_multi:
        filtered_df = df[
            (df["commodity"].isin(commodities_multi))
            & (df["date"] >= pd.to_datetime(start_date))
            & (df["date"] <= pd.to_datetime(end_date))
        ].sort_values("date")

        if filtered_df.empty:
          st.warning("No data found for the selected commodity and date range.")
        else:
          fig2 = px.line(
              filtered_df,
              x="date",
              y="avg_price",
              color="commodity",
              title=(
                  f"Price Trend from {start_date.strftime('%Y-%m-%d')} to"
                  f" {end_date.strftime('%Y-%m-%d')}"
              ),
              labels={"avg_price": "Average Price (NPR)", "date": "Date"},
          )
          st.plotly_chart(fig2, use_container_width=True)

          st.markdown("### 📊 Summary Statistics")
          stats = (
              filtered_df.groupby("commodity")["avg_price"]
              .agg(
                  Min_Price="min",
                  Max_Price="max",
                  Average_Price="mean",
                  Latest_Price="last",
              )
              .reset_index()
          )
          st.dataframe(stats, use_container_width=True)
      else:
        st.info("Pick at least one commodity above to see its price history.")