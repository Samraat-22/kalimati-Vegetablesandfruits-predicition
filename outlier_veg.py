import numpy as np
import pandas as pd


def main():
    file_path = r"c:\Users\ACER\Downloads\vegetable.csv"

    columns = ["commodity", "date", "unit", "min_price", "max_price", "avg_price"]

    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
        parse_dates=["date"],
        skipinitialspace=True,
    )

    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")

    df["z_score"] = df.groupby("commodity")["avg_price"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )

    flagged = df[df["z_score"].abs() > 4].copy()
    flagged = flagged.sort_values("z_score", key=abs, ascending=False)

    print(
        f"Flagged {len(flagged)} / {len(df)} rows with |z-score| > 4 "
        f"({len(flagged) / len(df):.2%} of data)"
    )
    print("\nTop 15 most extreme:")
    print(
        flagged[["commodity", "date", "avg_price", "z_score"]]
        .head(15)
        .to_string(index=False)
    )

    flagged.to_csv("outlier_report.csv", index=False)
    print(
        "\nSaved outlier_report.csv -- review these manually: they may be "
        "genuine spikes (festival demand, supply shock) or data-entry errors."
    )


if __name__ == "__main__":
    main()