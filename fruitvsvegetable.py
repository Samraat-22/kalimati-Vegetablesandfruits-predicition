import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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


def classify(commodity_base: str) -> str:
    if commodity_base in FRUITS:
        return "Fruit"
    if commodity_base in OTHER:
        return "Other"
    return "Vegetable"


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