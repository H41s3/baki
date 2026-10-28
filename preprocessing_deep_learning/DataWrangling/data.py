# Load libraries
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np  # Needed for conditional column creation

# The csv lives next to this file, so it loads no matter where we run from
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "marvel-wikia-data.csv"


def load_data(path=DEFAULT_CSV_PATH):
    # Load the local csv file, failing early with a clear message if it's missing
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Marvel csv not found at '{path}'. "
            "Pass the correct location with --csv PATH."
        )
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add a bodycount column to the Marvel data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"path to marvel-wikia-data.csv (default: {DEFAULT_CSV_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        dataframe = add_bodycount(load_data(args.csv))
    except FileNotFoundError as err:
        sys.exit(f"error: {err}")

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
