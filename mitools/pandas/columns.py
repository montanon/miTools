from typing import Callable, Iterable, List, Optional, Tuple, Union

from pandas import DataFrame, MultiIndex

from mitools.exceptions import ArgumentTypeError, ArgumentValueError

GROWTH_COLUMN_NAME = "growth_{:d}"
GROWTH_PCT_COLUMN_NAME = "growth%_{:d}"
SHIFTED_COLUMN_NAME = "shifted_{:d}"
ADDED_COLUMN_NAME = "+_{}"
SUBTRACTED_COLUMN_NAME = "-_{}"
MULTIPLIED_COLUMN_NAME = "*_{:s}"
DIVIDED_COLUMN_NAME = "/_{:s}"

INVALID_COLUMN_ERROR = "One or more of {} are not in DataFrame."
INVALID_TRANSFORMATION_ERROR = "Transformation {} provided is not Callable."


def select_index(
    dataframe: DataFrame,
    index: Union[str, Tuple, int, List[Union[str, Tuple, int]]],
    level: Optional[Union[str, int]] = None,
) -> DataFrame:
    if level is not None and not isinstance(level, (str, int)):
        raise ArgumentTypeError(
            f"The 'level' must be a string (for named levels) or an integer (for positional levels), not {type(level)}."
        )
    has_multi_index = isinstance(dataframe.index, MultiIndex)
    if level is not None:
        if not has_multi_index:
            raise ArgumentValueError(
                "level can only be specified for DataFrames with multi-index index"
            )
        if isinstance(level, str) and level not in dataframe.index.names:
            raise ArgumentValueError(f"Invalid level name: {level}")
        if isinstance(level, int) and (abs(level) >= dataframe.index.nlevels):
            raise ArgumentValueError(f"Invalid level index: {level}")
        if any(isinstance(col, tuple) for col in index):
            raise ArgumentValueError(
                "Cannot use tuples in index when level is specified"
            )
    index = [index] if isinstance(index, (str, tuple, int)) else index
    if not isinstance(index, list):
        raise ArgumentTypeError(
            "Provided 'index' must be a string, tuple, int, or list."
        )
    if level is not None:
        level_index = (
            level if isinstance(level, int) else dataframe.index.names.index(level)
        )
        index = [idx for idx in dataframe.index if idx[level_index] in index]
        if not index:
            raise ArgumentValueError(f"No 'index' provided are in 'level={level}'!")
    invalid_index = set(index) - set(dataframe.index)
    if invalid_index:
        raise ArgumentValueError(f"Invalid index: {invalid_index}")
    return dataframe.loc[index, :]


def select_columns(
    dataframe: DataFrame,
    columns: Union[str, Tuple, int, List[Union[str, Tuple, int]]],
    level: Optional[Union[str, int]] = None,
) -> DataFrame:
    if level is not None and not isinstance(level, (str, int)):
        raise ArgumentTypeError(
            f"The 'level' must be a string (for named levels) or an integer (for positional levels), not {type(level)}."
        )
    has_multi_index = isinstance(dataframe.columns, MultiIndex)
    if level is not None:
        if not has_multi_index:
            raise ArgumentValueError(
                "level can only be specified for DataFrames with multi-index columns"
            )
        if isinstance(level, str) and level not in dataframe.columns.names:
            raise ArgumentValueError(f"Invalid level name: {level}")
        if isinstance(level, int) and (abs(level) >= dataframe.columns.nlevels):
            raise ArgumentValueError(f"Invalid level index: {level}")
        if any(isinstance(col, tuple) for col in columns):
            raise ArgumentValueError(
                "Cannot use tuples in columns when level is specified"
            )
    columns = [columns] if isinstance(columns, (str, tuple, int)) else columns
    if not isinstance(columns, list):
        raise ArgumentTypeError(
            "Provided 'columns' must be a string, tuple, int, or list."
        )
    if level is not None:
        level_index = (
            level if isinstance(level, int) else dataframe.columns.names.index(level)
        )
        columns = [col for col in dataframe.columns if col[level_index] in columns]
        if not columns:
            raise ArgumentValueError(f"No 'columns' provided are in 'level={level}'!")
    invalid_columns = set(columns) - set(dataframe.columns)
    if invalid_columns:
        raise ArgumentValueError(f"Invalid columns: {invalid_columns}")
    return dataframe.loc[:, columns]


def transform_columns(
    dataframe: DataFrame,
    transformation: Callable,
    columns: Iterable[Union[str, Tuple]],
    level: Optional[Union[str, int]] = None,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    if not callable(transformation):
        raise ArgumentTypeError(INVALID_TRANSFORMATION_ERROR.format(transformation))
    transformation_name = (
        transformation.__name__
        if hasattr(transformation, "__name__")
        else "transformation"
    )
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    if transformation_name in {"ln", "log"}:
        selected_columns = selected_columns.replace(0, 1e-6)
        print("Replaced 0.0 values for 1e-6 to avoid -inf values!")
    try:
        transformed_columns = selected_columns.apply(transformation)
    except Exception as e:
        raise ArgumentValueError(
            f"Error while applying '{transformation_name}' transformation: {e}"
        )
    if rename:
        transformation_name = (
            transformation_name if not isinstance(rename, str) else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            transformed_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{transformation_name}")
                    for col in transformed_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            transformed_columns.columns = [
                f"{col}_{transformation_name}" for col in transformed_columns.columns
            ]
    return transformed_columns


def growth_columns(
    dataframe: DataFrame,
    columns: Iterable[Union[str, Tuple]],
    t: int,
    level: Optional[Union[str, int]] = None,
    pct: Optional[bool] = False,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    if not isinstance(t, int):
        raise ArgumentTypeError("Provided 't' must be an integer.")
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    try:
        shifted_columns = selected_columns.shift(t)
        variation_columns = selected_columns - shifted_columns
    except Exception as e:
        raise ArgumentValueError(f"Error while calculating variation with 't={t}': {e}")
    variation_columns = (
        (variation_columns / selected_columns) * 100.0 if pct else variation_columns
    )
    if rename:
        variation_name = (
            (GROWTH_PCT_COLUMN_NAME.format(t) if pct else GROWTH_COLUMN_NAME.format(t))
            if not isinstance(rename, str)
            else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            variation_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{variation_name}")
                    for col in variation_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            variation_columns.columns = [
                f"{col}_{variation_name}" for col in variation_columns.columns
            ]
    return variation_columns


def shift_columns(
    dataframe: DataFrame,
    columns: Iterable[Union[str, Tuple]],
    t: int,
    level: Optional[Union[str, int]] = None,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    if not isinstance(t, int):
        raise ArgumentTypeError("Provided 't' must be an integer.")
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    try:
        shifted_columns = selected_columns.shift(t)
    except Exception as e:
        raise ArgumentValueError(f"Error while calculating variation with 't={t}': {e}")
    if rename:
        shifted_name = (
            SHIFTED_COLUMN_NAME.format(t) if not isinstance(rename, str) else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            shifted_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{shifted_name}")
                    for col in shifted_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            shifted_columns.columns = [
                f"{col}_{shifted_name}" for col in shifted_columns.columns
            ]
    return shifted_columns


def add_columns(
    dataframe: DataFrame,
    columns: Iterable[Union[str, Tuple]],
    column_to_add: Union[str, Tuple],
    level: Optional[Union[str, int]] = None,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    column_to_add = select_columns(
        dataframe=dataframe, columns=column_to_add, level=level
    )
    if column_to_add.shape[1] > 1:
        raise ArgumentValueError(
            f"Column to add '{column_to_add}' with 'level={level}' must be a single column."
        )
    try:
        plus_columns = selected_columns + column_to_add.values
    except Exception as e:
        raise ArgumentValueError(
            f"Error while adding column '{column_to_add}' to columns '{columns}': {e}"
        )
    if rename:
        added_name = (
            ADDED_COLUMN_NAME.format(
                column_to_add
                if isinstance(column_to_add, str)
                else ",".join(*column_to_add)
            )
            if not isinstance(rename, str)
            else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            plus_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{added_name}")
                    for col in plus_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            plus_columns.columns = [
                f"{col}_{added_name}" for col in plus_columns.columns
            ]
    return plus_columns


def subtract_columns(
    dataframe: DataFrame,
    columns: Iterable[Union[str, Tuple]],
    column_to_subtract: Union[str, Tuple],
    level: Optional[Union[str, int]] = None,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    column_to_subtract = select_columns(
        dataframe=dataframe, columns=column_to_subtract, level=level
    )
    if column_to_subtract.shape[1] > 1:
        raise ArgumentValueError(
            f"Column to subtract '{column_to_subtract}' with 'level={level}' must be a single column."
        )
    try:
        subtracted_columns = selected_columns - column_to_subtract.values
    except Exception as e:
        raise ArgumentValueError(
            f"Error while subtracting column '{column_to_subtract}' from columns '{columns}': {e}"
        )
    if rename:
        subtracted_name = (
            SUBTRACTED_COLUMN_NAME.format(
                column_to_subtract
                if isinstance(column_to_subtract, str)
                else ",".join(*column_to_subtract)
            )
            if not isinstance(rename, str)
            else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            subtracted_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{subtracted_name}")
                    for col in subtracted_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            subtracted_columns.columns = [
                f"{col}_{subtracted_name}" for col in subtracted_columns.columns
            ]
    return subtracted_columns


def divide_columns(
    dataframe: DataFrame,
    columns: Iterable[Union[str, Tuple]],
    column_to_divide: Union[str, Tuple],
    level: Optional[Union[str, int]] = None,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    column_to_divide = select_columns(
        dataframe=dataframe, columns=column_to_divide, level=level
    )
    if column_to_divide.shape[1] > 1:
        raise ArgumentValueError(
            f"Column to divide '{column_to_divide}' with 'level={level}' must be a single column."
        )
    try:
        divided_columns = selected_columns / column_to_divide.values
    except ZeroDivisionError:
        raise ArgumentValueError("Division by zero encountered.")
    except Exception as e:
        raise ArgumentValueError(
            f"Error while dividing columns '{columns}' by '{column_to_divide}': {e}"
        )
    if rename:
        divided_name = (
            DIVIDED_COLUMN_NAME.format(
                column_to_divide
                if isinstance(column_to_divide, str)
                else ",".join(*column_to_divide)
            )
            if not isinstance(rename, str)
            else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            divided_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{divided_name}")
                    for col in divided_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            divided_columns.columns = [
                f"{col}_{divided_name}" for col in divided_columns.columns
            ]

    return divided_columns


def multiply_columns(
    dataframe: DataFrame,
    columns: Iterable[Union[str, Tuple]],
    column_to_multiply: Union[str, Tuple],
    level: Optional[Union[str, int]] = None,
    rename: Optional[Union[str, bool]] = True,
) -> DataFrame:
    selected_columns = select_columns(dataframe=dataframe, columns=columns, level=level)
    column_to_multiply = select_columns(
        dataframe=dataframe, columns=column_to_multiply, level=level
    )
    if column_to_multiply.shape[1] > 1:
        raise ArgumentValueError(
            f"Column to multiply '{column_to_multiply}' with 'level={level}' must be a single column."
        )
    try:
        multiplied_columns = selected_columns * column_to_multiply.values
    except Exception as e:
        raise ArgumentValueError(
            f"Error while multiplying column '{column_to_multiply}' with columns '{columns}': {e}"
        )
    if rename:
        multiplied_name = (
            MULTIPLIED_COLUMN_NAME.format(
                column_to_multiply
                if isinstance(column_to_multiply, str)
                else ",".join(*column_to_multiply)
            )
            if not isinstance(rename, str)
            else rename
        )
        if isinstance(dataframe.columns, MultiIndex):
            multiplied_columns.columns = MultiIndex.from_tuples(
                [
                    (*col[:-1], f"{col[-1]}_{multiplied_name}")
                    for col in multiplied_columns.columns
                ],
                names=dataframe.columns.names,
            )
        else:
            multiplied_columns.columns = [
                f"{col}_{multiplied_name}" for col in multiplied_columns.columns
            ]
    return multiplied_columns
