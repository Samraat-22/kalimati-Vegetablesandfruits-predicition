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
            "Forecasts next-day average price (NPR/unit) for Kalimati market"
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
    },
    "नेपाली": {
        "title": "🥕 कालिमाटी तरकारी बजार — मूल्य पूर्वानुमान",
        "subtitle": (
            "कालिमाटी बजारका वस्तुहरूको भोलिको औसत मूल्य (रू/इकाई) मसिन"
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
      target_date = last_date + timedelta(days=1)

      series = hist.set_index("date")["avg_price"]

      def lag_val(n):
        idx = len(series) - n
        return series.iloc[idx] if idx >= 0 else series.iloc[0]

      row = {
          "min_price": last_row.get("min_price", last_row["avg_price"] * 0.9),
          "max_price": last_row.get("max_price", last_row["avg_price"] * 1.1),
          "price_lag_1d": lag_val(1),
          "price_lag_7d": lag_val(7),
          "rolling_7d_avg": series.tail(7).mean(),
          "is_monsoon": int(target_date.month in [6, 7, 8, 9]),
          "is_festival_season": int(target_date.month in [9, 10, 11]),
          "month": target_date.month,
          "day_of_week": target_date.dayofweek,
          "commodity": last_row.get("commodity", commodity),
          "category": last_row.get("category", "Vegetable"),
          "unit": last_row.get("unit", "KG"),
      }
      row_df = pd.DataFrame([row]).fillna(0)

      for c in cat_features:
        le = encoders[c]
        val = str(row_df.loc[0, c])
        row_df[c + "_enc"] = (
            le.transform([val])[0] if val in le.classes_ else -1
        )

      X_input = row_df[feature_cols]
      pred = model.predict(X_input)[0]

      st.markdown(f"### {t['prediction_header']}")
      pc1, pc2, pc3 = st.columns(3)
      pc1.metric(
          f"{t['predicted_price']} ({target_date.strftime('%Y-%m-%d')})",
          f"NPR {pred:.2f}",
      )
      delta = pred - last_row["avg_price"]
      pc2.metric(t["change"], f"{delta:+.2f}", delta=f"{delta:+.2f}")
      pc3.metric(t["unit"], last_row.get("unit", "KG"))

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
          title=f"{commodity}: Last 30 Days + Next-Day Forecast",
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