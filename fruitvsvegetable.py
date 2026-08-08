import matplotlib.pyplot as plt
import pandas as pd

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
    file_path = r"c:\Users\ACER\Downloads\vegetable.csv"
    columns = [
        "commodity",
        "date",
        "unit",
        "min_price",
        "max_price",
        "avg_price",
    ]

    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
        parse_dates=["date"],
        skipinitialspace=True,
    )

    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df["commodity_base"] = (
        df["commodity"].astype(str).str.split("(").str[0].str.strip()
    )

    df["category"] = df["commodity_base"].apply(classify)
    df = df[df["category"] != "Other"]

    print(df.groupby("category")["avg_price"].describe())

    daily_avg = df.groupby(["date", "category"])["avg_price"].mean().unstack()
    daily_avg["difference"] = daily_avg["Fruit"] - daily_avg["Vegetable"]
    print(
        "\nOverall mean price -> Fruit: {:.2f}, Vegetable: {:.2f}, Diff: {:.2f}".format(
            daily_avg["Fruit"].mean(),
            daily_avg["Vegetable"].mean(),
            daily_avg["difference"].mean(),
        )
    )
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
    plt.savefig("fruit_vs_vegetable_price.png", dpi=110)
    print("Saved fruit_vs_vegetable_price.png")


if __name__ == "__main__":
    main()