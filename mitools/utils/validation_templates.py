import re
from typing import List

import pandas as pd
from pandas import DataFrame, MultiIndex

from mitools.exceptions import ColumnValidationError


def sankey_plot_validation(
    dataframe: DataFrame,
    expected_levels: List[str] = ["year_range", "n-gram", "values"],
):
    if (
        not isinstance(dataframe.columns, MultiIndex)
        or len(dataframe.columns.levels) != 3
    ):
        raise ColumnValidationError(
            "DataFrame columns must have a MultiIndex with exactly 3 levels."
        )
    # Check if the levels are named as expected
    if dataframe.columns.names != expected_levels:
        raise ColumnValidationError(f"Column levels must be named {expected_levels}.")
    # Regular expressions for level validation
    year_range_pattern = re.compile(
        r"^\(\d{4}, \d{4}\)$"
    )  # Matches '(X, Y)' where X and Y are 4-digit years
    ngram_pattern = re.compile(
        r"^\d+_\d+-Gram$"
    )  # Matches 'X_Y-Gram' where X, Y are numbers with X <= Y
    # Loop over each level to validate
    for year_range, ngram, value in dataframe.columns:
        # Level 0 check: "(X, Y)"
        if not year_range_pattern.match(year_range):
            raise ColumnValidationError(
                f"Level 0 column '{year_range}' must match '(X, Y)' where X and Y are years."
            )
        # Level 1 check: "X_Y-Gram" with X <= Y
        if not ngram_pattern.match(ngram):
            raise ColumnValidationError(
                f"Level 1 column '{ngram}' must match 'X_Y-Gram' where X and Y are numbers with X <= Y."
            )
        x, y = map(int, ngram.replace("-Gram", "").split("_")[0:2])
        if x > y:
            raise ColumnValidationError(
                f"Level 1 column '{ngram}' should satisfy X <= Y, where X = {x} and Y = {y}."
            )
        # Level 2 check: values must be 'Gram' or 'Count'
        if value not in ["Gram", "Count"]:
            raise ColumnValidationError(
                f"Level 2 column '{value}' must be either 'Gram' or 'Count'."
            )
        # Data type checks for 'Gram' and 'Count' columns
        if value == "Gram":
            if (
                not dataframe[year_range, ngram, value]
                .apply(lambda x: isinstance(x, str) or pd.isna(x))
                .all()
            ):
                raise ColumnValidationError(
                    "Level 2 'Gram' columns must contain strings or NaN values only."
                )
        elif value == "Count":
            if (
                not dataframe[year_range, ngram, value]
                .apply(lambda x: pd.api.types.is_numeric_dtype(type(x)) or pd.isna(x))
                .all()
            ):
                raise ColumnValidationError(
                    "Level 2 'Count' columns must contain numeric values or NaN only."
                )
    return True
