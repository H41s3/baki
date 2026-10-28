# Load libraries
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np  # Needed for conditional column creation

# The csv lives next to this file, so it loads no matter where we run from
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "marvel-wikia-data.csv"

# Columns the bodycount transformation depends on
REQUIRED_COLUMNS = ("name",)

logger = logging.getLogger(__name__)


class MissingColumnsError(ValueError):
    """Raised when the csv is missing columns the transformation needs."""


def validate_columns(dataframe: pd.DataFrame, required=REQUIRED_COLUMNS) -> None:
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


def load_csv(path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    # Load the local csv file, failing early with a clear message if it's missing
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Marvel csv not found at '{path}'. "
            "Pass the correct location with --csv PATH."
        )
    return pd.read_csv(path)


def add_bodycount(
    dataframe: pd.DataFrame,
    match: str = "Peter Parker",
    hit: int = 3,
    miss: int = 0,
) -> pd.DataFrame:
    # Create the column conditionally:
    # IF the name contains `match` (literal substring), set it to `hit`.
    # ELSE (for everyone else), set it to `miss`.
    # Pure: works on a copy so the caller's DataFrame is left untouched,
    # and needs no file on disk, so it can be tested with an in-memory frame.
    validate_columns(dataframe)
    dataframe = dataframe.copy()
    dataframe["bodycount"] = np.where(
        dataframe["name"].str.contains(match, regex=False, na=False), hit, miss
    )
    return dataframe


def bodycount_preview(dataframe, n=5):
    # Return the slices used to sanity-check the transform, leaving display to the caller:
    # Peter Parker's row(s) (should be 3) and the first n rows (mostly 0).
    columns = ["name", "bodycount"]
    peter_parker = dataframe[dataframe["name"].str.contains("Peter Parker", na=False)][columns]
    return peter_parker, dataframe[columns].head(n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add a bodycount column to the Marvel data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"path to marvel-wikia-data.csv (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging verbosity; use WARNING or higher to suppress the preview (default: INFO)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(message)s")

    try:
        dataframe = add_bodycount(load_csv(args.csv))
    except (FileNotFoundError, MissingColumnsError) as err:
        sys.exit(f"error: {err}")

    # Let's verify it worked: Peter Parker should have a '3', others a '0'
    if logger.isEnabledFor(logging.INFO):
        peter_parker, head = bodycount_preview(dataframe)
        logger.info("--- Peter Parker's Row ---\n%s", peter_parker.to_string())
        logger.info("--- First 5 Rows ---\n%s", head.to_string())


if __name__ == "__main__":
    main()
