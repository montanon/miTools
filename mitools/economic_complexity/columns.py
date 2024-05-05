from typing import Callable, List, Optional

from pandas import DataFrame, Index, IndexSlice, MultiIndex

from ..exceptions.custom_exceptions import ArgumentKeyError, ArgumentTypeError

GROWTH_COLUMN_NAME = "_growth_{:d}"
GROWTH_PCT_COLUMN_NAME = "_growth%_{:d}"
SHIFTED_COLUMN_NAME = "{}_shifted_by_{}"

INVALID_COLUMN_ERROR = "One or more of {} are not in DataFrame."
INVALID_TRANSFORMATION_ERROR = "Transformation {} provided is not Callable."


def transform_columns(
    dataframe: DataFrame, columns_names: List[str], transformation: Callable
) -> DataFrame:
    if not all([c in dataframe.columns.get_level_values(-1) for c in columns_names]):
        raise ArgumentKeyError(INVALID_COLUMN_ERROR.format(columns_names))
    if not isinstance(transformation, Callable):
        raise ArgumentTypeError(INVALID_TRANSFORMATION_ERROR.format(transformation))
    transformation_name = str(transformation).split("'")[1]
    selected_columns = dataframe.loc[:, IndexSlice[:, columns_names]]
    if transformation_name in set(("ln", "log")):  # YIKES: robustness needed
        selected_columns = selected_columns.replace(0, 1e-6)
    transformed_columns = selected_columns.apply(transformation)
    transformed_columns.columns = MultiIndex.from_tuples(
        [
            (col_0, f"{col_1}_{transformation_name}")
            if col_1 in columns_names
            else (col_0, col_1)
            for col_0, col_1 in transformed_columns.columns.values
        ],
        names=dataframe.columns.names,
    )
    return transformed_columns


def variation_columns(
    dataframe: DataFrame, columns_names: List[str], t: int, pct: Optional[bool] = False
) -> DataFrame:
    selected_columns = dataframe.loc[:, IndexSlice[:, columns_names]]
    shifted_columns = selected_columns.shift(t)
    variation_columns = selected_columns - shifted_columns
    variation_columns = (
        (variation_columns / selected_columns) * 100.0 if pct else variation_columns
    )
    growth_name = (
        GROWTH_PCT_COLUMN_NAME.format(t) if pct else GROWTH_COLUMN_NAME.format(t)
    )
    variation_columns.columns = MultiIndex.from_tuples(
        [
            (col_0, f"{col_1}{growth_name}")
            if col_1 in columns_names
            else (col_0, col_1)
            for col_0, col_1 in variation_columns.columns.values
        ],
        names=dataframe.columns.names,
    )
    return variation_columns


def shift_columns(dataframe: DataFrame, columns_names: List[str], t: int) -> DataFrame:
    shifted_columns = dataframe.loc[:, IndexSlice[:, columns_names]].shift(-t)
    shifted_columns.columns = MultiIndex.from_tuples(
        [
            (col_0, SHIFTED_COLUMN_NAME.format(col_1, t))
            if col_1 in columns_names
            else (col_0, col_1)
            for col_0, col_1 in shifted_columns.columns.values
        ],
        names=dataframe.columns.names,
    )
    return shifted_columns


def add_columns(
    dataframe: DataFrame, column1: str, column2: str, new_name: str
) -> DataFrame:
    if isinstance(dataframe.index, Index):
        columns1 = dataframe.loc[:, [column1]]
        columns2 = dataframe.loc[:, [column2]]
    else:
        columns1 = dataframe.loc[:, IndexSlice[:, column1]]
        columns2 = dataframe.loc[:, IndexSlice[:, column2]]
    plus_columns = columns1 + columns2.values
    if isinstance(dataframe.index, Index):
        plus_columns.columns = [new_name]
    else:
        plus_columns.columns = MultiIndex.from_tuples(
            [(col_0, new_name) for col_0, _ in plus_columns.columns.values],
            names=dataframe.columns.names,
        )
    return plus_columns


def subtract_columns(
    dataframe: DataFrame, column1: str, column2: str, new_name: str
) -> DataFrame:
    if isinstance(dataframe.index, Index):
        columns1 = dataframe.loc[:, [column1]]
        columns2 = dataframe.loc[:, [column2]]
    else:
        columns1 = dataframe.loc[:, IndexSlice[:, column1]]
        columns2 = dataframe.loc[:, IndexSlice[:, column2]]
    minus_columns = columns1 - columns2.values
    if isinstance(dataframe.index, Index):
        minus_columns.columns = [new_name]
    else:
        minus_columns.columns = MultiIndex.from_tuples(
            [(col_0, new_name) for col_0, _ in minus_columns.columns.values],
            names=dataframe.columns.names,
        )
    return minus_columns


def divide_columns(
    dataframe: DataFrame, num_column: str, den_column: str, new_name: str
) -> DataFrame:
    if isinstance(dataframe.index, Index):
        num_columns = dataframe.loc[:, [num_column]]
        den_columns = dataframe.loc[:, [den_column]]
    else:
        num_columns = dataframe.loc[:, IndexSlice[:, num_column]]
        den_columns = dataframe.loc[:, IndexSlice[:, den_column]]
    div_columns = num_columns / den_columns.values
    if isinstance(dataframe.index, Index):
        div_columns.columns = [new_name]
    else:
        div_columns.columns = MultiIndex.from_tuples(
            [(col_0, new_name) for col_0, _ in div_columns.columns.values],
            names=dataframe.columns.names,
        )
    return div_columns


def multiply_columns(
    dataframe: DataFrame, column1: str, column2: str, new_name: str
) -> DataFrame:
    if isinstance(dataframe.index, Index):
        columns1 = dataframe.loc[:, [column1]]
        columns2 = dataframe.loc[:, [column2]]
    else:
        columns1 = dataframe.loc[:, IndexSlice[:, column1]]
        columns2 = dataframe.loc[:, IndexSlice[:, column2]]
    mul_columns = columns1 * columns2.values
    if isinstance(dataframe.index, Index):
        mul_columns.columns = [new_name]
    else:
        mul_columns.columns = MultiIndex.from_tuples(
            [(col_0, new_name) for col_0, _ in mul_columns.columns.values],
            names=dataframe.columns.names,
        )
    return mul_columns
