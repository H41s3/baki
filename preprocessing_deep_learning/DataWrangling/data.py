# Load libraries
from pathlib import Path

import pandas as pd
import numpy as np  # Needed for conditional column creation

# The csv lives next to this file, so it loads no matter where we run from
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "marvel-wikia-data.csv"


def load_data(path=DEFAULT_CSV_PATH):
    # Load the local csv file
    return pd.read_csv(path)


def add_bodycount(dataframe):
    # Create the column conditionally:
    # IF the name contains 'Peter Parker', set it to 3.
    # ELSE (for everyone else), set it to 0.
    # Works on a copy so the caller's DataFrame is left untouched.
    dataframe = dataframe.copy()
    dataframe["bodycount"] = np.where(
        dataframe["name"].str.contains("Peter Parker", na=False), 3, 0
    )
    return dataframe


def main():
    dataframe = add_bodycount(load_data())

    # Let's verify it worked:
    # 1. Print Peter Parker's row to see his '3'
    print("--- Peter Parker's Row ---")
    print(
        dataframe[dataframe["name"].str.contains("Peter Parker", na=False)][
            ["name", "bodycount"]
        ]
    )

    # 2. Print the first 5 rows to see that others are '0'
    print("\n--- First 5 Rows ---")
    print(dataframe[["name", "bodycount"]].head(5))


if __name__ == "__main__":
    main()
