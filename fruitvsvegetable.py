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
    df["commodity_base"] = (
        df["commodity"].astype(str).str.split("(").str[0].str.strip()
    )

    df["category"] = df["commodity_base"].apply(classify)
    df = df[df["category"] != "Other"]

    # ------------------------------------------------------------------
    # URL-shareable category + commodity selectors
    # ------------------------------------------------------------------
    category_options = ["All", "Fruit", "Vegetable"]
    query_category = st.query_params.get("category", "All")
    if query_category not in category_options:
        query_category = "All"

    selected_category = st.selectbox(
        "Select category",
        category_options,
        index=category_options.index(query_category),
    )

    # Commodity choices depend on the chosen category
    if selected_category == "All":
        available_commodities = sorted(df["commodity_base"].unique().tolist())
    else:
        available_commodities = sorted(
            df.loc[df["category"] == selected_category, "commodity_base"].unique().tolist()
        )
    commodity_options = ["All"] + available_commodities

    query_commodity = st.query_params.get("commodity", "All")
    if query_commodity not in commodity_options:
        query_commodity = "All"

    selected_commodity = st.selectbox(
        "Select a commodity (or 'All')",
        commodity_options,
        index=commodity_options.index(query_commodity),
    )

    # Keep the URL in sync with the current selections so the page can be shared
    if selected_category == "All":
        if "category" in st.query_params:
            del st.query_params["category"]
    else:
        st.query_params["category"] = selected_category

    if selected_commodity == "All":
        if "commodity" in st.query_params:
            del st.query_params["commodity"]
    else:
        st.query_params["commodity"] = selected_commodity

    if selected_category != "All":
        df = df[df["category"] == selected_category]
    if selected_commodity != "All":
        df = df[df["commodity_base"] == selected_commodity]

    if selected_category != "All" or selected_commodity != "All":
        label = " / ".join(
            v for v in [selected_category, selected_commodity] if v != "All"
        )
        st.caption(f"Showing data filtered to: **{label}**")

    st.subheader("Summary statistics")
    st.dataframe(df.groupby("category")["avg_price"].describe())

    daily_avg = df.groupby(["date", "category"])["avg_price"].mean().unstack()

    if "Fruit" not in daily_avg.columns:
        daily_avg["Fruit"] = pd.NA
    if "Vegetable" not in daily_avg.columns:
        daily_avg["Vegetable"] = pd.NA

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

    axes[1, 1].plot(daily_avg.index, daily_avg["difference"], color="purple")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Price difference over time (Fruit − Vegetable)")
    axes[1, 1].set_ylabel("NPR difference")

    plt.tight_layout()

    st.subheader("Charts")
    st.pyplot(fig)

    plt.savefig("fruit_vs_vegetable_price.png", dpi=110)


if __name__ == "__main__":
    main()