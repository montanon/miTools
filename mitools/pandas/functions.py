from os import PathLike
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Union

import pandas as pd
from pandas import DataFrame
from pandas._libs.tslibs.parsing import DateParseError
from tqdm import tqdm

from mitools.exceptions.custom_exceptions import ArgumentTypeError, ArgumentValueError

INT_COL_ERROR = "Value or values in any of columns={} cannnot be converted into int."
NON_DATE_COL_ERROR = (
    "Column {} has values that cannot be converted to datetime objects."
)


def prepare_int_cols(
    dataframe: DataFrame,
    cols: Union[Iterable[str], str],
    nan_placeholder: int,
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
) -> DataFrame:
    if errors not in ["raise", "coerce", "ignore"]:
        raise ArgumentValueError(
            "Argument 'errors' must be one of ['raise', 'coerce', 'ignore']."
        )
    cols = [cols] if isinstance(cols, str) else cols
    if not isinstance(cols, Iterable) or not all(isinstance(c, str) for c in cols):
        raise ArgumentTypeError(
            "Argument 'cols' must be a string or an iterable of strings."
        )
    missing_cols = [col for col in cols if col not in dataframe.columns]
    if missing_cols:
        raise ArgumentValueError(f"Columns {missing_cols} not found in DataFrame.")
    try:
        for col in cols:
            dataframe[col] = pd.to_numeric(
                dataframe[col], errors=errors, downcast="integer"
            )
            if errors != "ignore":
                dataframe[col] = dataframe[col].fillna(nan_placeholder)
                dataframe[col] = dataframe[col].astype(int)
    except (ValueError, KeyError) as e:
        raise ArgumentTypeError(f"{INT_COL_ERROR.format(col)}, Details: {str(e)}")
    return dataframe


def prepare_str_cols(
    dataframe: DataFrame, cols: Union[Iterable[str], str]
) -> DataFrame:
    cols = [cols] if isinstance(cols, str) else cols
    if not isinstance(cols, Iterable) or not all(isinstance(c, str) for c in cols):
        raise ArgumentTypeError(
            "Argument 'cols' must be a string or an iterable of strings."
        )
    missing_cols = [col for col in cols if col not in dataframe.columns]
    if missing_cols:
        raise ArgumentValueError(f"Columns {missing_cols} not found in DataFrame.")
    dataframe[cols] = dataframe[cols].astype(str)
    return dataframe


def prepare_date_cols(
    dataframe: DataFrame,
    cols: Union[Iterable[str], str],
    nan_placeholder: Union[str, pd.Timestamp],
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
    date_format: str = None,
) -> DataFrame:
    if errors not in ["raise", "coerce", "ignore"]:
        raise ArgumentValueError(
            "Argument 'errors' must be one of ['raise', 'coerce', 'ignore']."
        )
    cols = [cols] if isinstance(cols, str) else cols
    if not isinstance(cols, Iterable) or not all(isinstance(c, str) for c in cols):
        raise ArgumentTypeError(
            "Argument 'cols' must be a string or an iterable of strings."
        )
    missing_cols = [col for col in cols if col not in dataframe.columns]
    if missing_cols:
        raise ArgumentValueError(f"Columns {missing_cols} not found in DataFrame.")
    try:
        for col in cols:
            dataframe[col] = pd.to_datetime(
                dataframe[col], errors=errors, format=date_format
            )
            if errors != "ignore" and nan_placeholder is not None:
                dataframe[col] = dataframe[col].fillna(pd.to_datetime(nan_placeholder))
    except (ValueError, DateParseError) as e:
        raise ArgumentTypeError(f"{NON_DATE_COL_ERROR.format(col)}, Details: {str(e)}")
    return dataframe


def prepare_bool_cols(df: DataFrame, cols: Union[Iterable[str], str]) -> DataFrame:
    df[cols] = df[cols].astype(bool)
    return df


def build_group_subgroup(
    df: DataFrame,
    value_col: str,
    entity: str,
    group_col: str,
    subgroup_col: str,
    time_col,
) -> DataFrame:
    group_df = df.query(f'{group_col} == "{entity}"')
    group_subgroups = (
        group_df.groupby(by=[time_col, group_col, subgroup_col])[[value_col]]
        .first()
        .reset_index()
    )
    group_subgroups = group_subgroups.pivot(
        index=time_col, columns=subgroup_col, values=value_col
    )
    group_subgroups.index.name = entity
    return group_subgroups


def build_groups_subgroups(
    df: DataFrame, group_col: str, value_col: str, subgroup_col: str, time_col: str
) -> DataFrame:
    groups_subgroups = {}
    for entity in tqdm(df[group_col].unique()):
        groups_subgroups[entity] = build_group_subgroup(
            df, entity, value_col, group_col, subgroup_col, time_col
        )
    for entity, subgroups in groups_subgroups.items():
        subgroups.columns = pd.MultiIndex.from_product([[entity], subgroups.columns])
    groups_subgroups = pd.concat(groups_subgroups.values(), axis=1)
    groups_subgroups.columns.names = [group_col, groups_subgroups.columns.names[1]]
    return groups_subgroups


def build_group(
    df: DataFrame, value_cols: List[str], entity: str, group_col: str, time_col: str
) -> DataFrame:
    group_df = df.query(f'{group_col} == "{entity}"')
    group = group_df.groupby(by=[time_col])[[*value_cols]].first().reset_index()
    group.set_index(time_col, inplace=True)
    group.index.name = entity
    return group


def build_groups(
    df: DataFrame, value_cols: List[str], group_col: str, time_col: str
) -> DataFrame:
    groups = {}
    for entity in tqdm(df[group_col].unique()):
        groups[entity] = build_group(df, entity, value_cols, group_col, time_col)
    for entity, group in groups.items():
        group.columns = pd.MultiIndex.from_product([[entity], group.columns])
    return pd.concat(groups.values(), axis=1)


def melt_dataframe(
    _dataframe: DataFrame,
    value_col: str,
    entities: Iterable,
    group_col: str,
    subgroup_col: str,
    time_col: str,
) -> DataFrame:
    value_cols = [c for c in _dataframe.columns if c[-1].split(" ")[-1] == value_col]
    dataframe = _dataframe.loc[pd.IndexSlice[:, value_cols]]
    dataframe = dataframe.copy(deep=True).T
    dataframes = []
    for entity in entities:
        entity_df = dataframe.loc[entity].copy(deep=True)
        entity_df[group_col] = entity
        dataframes.append(entity_df)
    combined_df = pd.concat(dataframes)
    melted_df = pd.melt(
        combined_df.reset_index(),
        id_vars=[subgroup_col, group_col],
        var_name=time_col,
        value_name=value_col,
    )
    return melted_df


def store_dataframe_by_level(
    df: DataFrame, base_path: Union[str, PathLike], level: Union[str, int]
) -> None:
    if not isinstance(df, DataFrame):
        raise Exception("Error: df is not a pandas DataFrame.")
    if not isinstance(base_path, (str, PathLike)):
        raise ValueError("Error: base_path is not a string or a PathLike object.")
    if not isinstance(level, (int, str)):
        raise ValueError("Error: level is not an integer.")
    if isinstance(level, int) and (level < 0 or level >= df.columns.nlevels):
        raise ValueError(
            f"Error: level {level} is not a valid level for the DataFrame."
        )
    if isinstance(level, str) and level not in df.columns.names:
        raise ValueError(
            f"Error: level {level} is not a valid level for the DataFrame."
        )
    level_values = df.columns.get_level_values(level).unique()
    for n, value in enumerate(level_values):
        if value not in df.columns.get_level_values(level):
            raise ValueError(
                f"Error: value {value} is not in level {level} of the DataFrame."
            )
        sub_df = df.xs(value, axis=1, level=level, drop_level=False)
        sub_path = Path(str(base_path).replace(".parquet", f"{n}_sub.parquet"))
        sub_df.to_parquet(sub_path)


def load_level_destructured_dataframe(
    base_path: Union[str, PathLike], level: Union[str, int]
) -> DataFrame:
    if not isinstance(base_path, (str, PathLike)):
        raise ValueError("base_path must be a string or a PathLike object.")
    if not isinstance(level, int):
        raise ValueError("level must be an integer.")
    if isinstance(base_path, str):
        base_path = Path(base_path)
    base_dir, base_filename = base_path.parent, base_path.stem
    parquet_files = list(base_dir.glob(f"{base_filename}*_sub.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {base_dir} with prefix {base_filename}."
        )
    df = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(df, axis=1)
    return df


def idxslice(
    df: DataFrame, level: Union[int, str], value: Union[List[Any], Any], axis: int
) -> pd.IndexSlice:
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 for index or 1 for columns")
    value = [value] if not isinstance(value, list) else value
    multiidx = df.index if axis == 0 else df.columns
    if isinstance(level, str):
        if level not in multiidx.names:
            raise ValueError("level is not in the axis index provided")
        level = multiidx.names.index(level)
    slices = [slice(None)] * multiidx.nlevels
    slices[level] = value
    return pd.IndexSlice[tuple(slices)]


def quantize_group(group, column, N):
    quantiles = pd.qcut(group[column], N, labels=False)
    quantiles = quantiles / (N - 1)
    group[column] = quantiles + 0.1
    return group
