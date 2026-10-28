# Load libraries
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np  # Needed for conditional column creation

# The csv lives next to this file, so it loads no matter where we run from
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "marvel-wikia-data.csv"

# Columns the bodycount transformation depends on
REQUIRED_COLUMNS = ("name",)


class MissingColumnsError(ValueError):
    """Raised when the csv is missing columns the transformation needs."""


def validate_columns(dataframe, required=REQUIRED_COLUMNS):
    # Fail early with a clear message instead of a bare KeyError mid-transform
    missing = [col for col in required if col not in dataframe.columns]
    if not missing:
        return

    # Point out near-misses like 'Name' or ' name ' so the fix is obvious
    by_normalized = {str(col).strip().lower(): col for col in dataframe.columns}
    hints = [
        f"'{by_normalized[col.lower()]}' looks like '{col}'"
        for col in missing
        if col.lower() in by_normalized
    ]
    message = (
        f"missing required column(s): {', '.join(repr(c) for c in missing)}. "
        f"Available columns: {', '.join(repr(str(c)) for c in dataframe.columns)}."
    )
    if hints:
        message += f" Possible match: {'; '.join(hints)} (column names are case-sensitive)."
    raise MissingColumnsError(message)


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
    validate_columns(dataframe)
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
    except (FileNotFoundError, MissingColumnsError) as err:
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
