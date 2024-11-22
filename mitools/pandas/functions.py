from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import pandas as pd
from pandas import DataFrame
from pandas._libs.tslibs.parsing import DateParseError
from tqdm import tqdm

from mitools.exceptions.custom_exceptions import ArgumentTypeError, ArgumentValueError

INT_COL_ERROR = "Value or values in any of columns={} cannnot be converted into int."
BOOL_COL_ERROR = "Value or values in any of columns={} cannnot be converted into bool."
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


def prepare_bool_cols(
    dataframe: DataFrame, cols: Union[Iterable[str], str], nan_placeholder: bool = False
) -> DataFrame:
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
            dataframe[col] = dataframe[col].fillna(nan_placeholder)
            dataframe[col] = dataframe[col].astype(bool)
    except Exception as e:
        raise ArgumentTypeError(
            f"{BOOL_COL_ERROR.format(col)}: {cols}. Details: {str(e)}"
        )
    return dataframe


def reshape_country_indicators(
    data: DataFrame,
    country: str,
    indicator_column: str,
    country_column: str,
    region_column: str,
    year_column: str,
    aggregation_function: str = "first",
) -> DataFrame:
    """
    Reshapes data for a specific country by aggregating and pivoting regional indicators over years.

    Args:
        data (DataFrame): The input DataFrame containing country data.
        country (str): The name of the country to filter data for.
        indicator_column (str): The column containing indicator values (e.g., GDP, population).
        country_column (str): The column identifying countries.
        region_column (str): The column identifying regions or sub-regions within countries.
        year_column (str): The column representing years.
        agg_func (str): The aggregation function to apply to the indicators (default is 'first').

    Returns:
        DataFrame: A pivoted DataFrame with `year_column` as the index,
                   `region_column` values as columns, and aggregated indicators.
    """
    return reshape_group_data(
        dataframe=data,
        filter_value=country,
        value_column=indicator_column,
        group_column=country_column,
        subgroup_column=region_column,
        time_column=year_column,
        agg_func=aggregation_function,
    )


def reshape_group_data(
    dataframe: DataFrame,
    filter_value: str,
    value_column: str,
    group_column: str,
    subgroup_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {value_column, group_column, subgroup_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    filtered_data = dataframe.query(f"{group_column} == @filter_value")
    if filtered_data.empty:
        raise ArgumentValueError(
            f"No data found for group '{filter_value}' in column '{group_column}'."
        )
    grouped_data = (
        filtered_data.groupby(by=[time_column, group_column, subgroup_column])[
            [value_column]
        ]
        .agg(agg_func)
        .reset_index()
    )
    pivoted_data = grouped_data.pivot(
        index=time_column, columns=subgroup_column, values=value_column
    )
    all_times = dataframe[time_column].unique()
    pivoted_data = pivoted_data.reindex(all_times, fill_value=None)
    pivoted_data = pivoted_data.sort_index()
    pivoted_data = pivoted_data.sort_index(axis=1)
    pivoted_data.index.name = filter_value
    return pivoted_data


def reshape_countries_indicators(
    data: DataFrame,
    country_column: str,
    indicator_column: str,
    region_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    """
    Reshapes data for countries by aggregating and pivoting regional indicators over years.

    Args:
        data (DataFrame): The input DataFrame.
        country_column (str): The column identifying the country.
        indicator_column (str): The column containing the indicator values to aggregate.
        region_column (str): The column identifying regions within each country.
        time_column (str): The column representing time.
        agg_func (str): Aggregation function to apply (default is "first").

    Returns:
        DataFrame: A multi-index DataFrame with countries as primary columns
                   and industries as secondary columns.
    """
    return reshape_groups_subgroups(
        dataframe=data,
        group_column=country_column,
        value_column=indicator_column,
        subgroup_column=region_column,
        time_column=time_column,
        agg_func=agg_func,
    )


def reshape_groups_subgroups(
    dataframe: DataFrame,
    group_column: str,
    value_column: str,
    subgroup_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {value_column, group_column, subgroup_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    groups_subgroups: Dict[str, DataFrame] = {}
    for group in tqdm(dataframe[group_column].unique(), desc="Processing groups"):
        try:
            groups_subgroups[group] = reshape_group_data(
                dataframe=dataframe,
                filter_value=group,
                value_column=value_column,
                group_column=group_column,
                subgroup_column=subgroup_column,
                time_column=time_column,
                agg_func=agg_func,
            )
        except ArgumentValueError as e:
            raise ArgumentValueError(f"Error processing group '{group}': {str(e)}")
    for group, subgroups in groups_subgroups.items():
        subgroups.columns = pd.MultiIndex.from_product([[group], subgroups.columns])
    try:
        combined_groups = pd.concat(groups_subgroups.values(), axis=1)
    except ValueError as e:
        raise ArgumentValueError(f"Error concatenating groups: {str(e)}")
    combined_groups = combined_groups.sort_index()
    combined_groups = combined_groups.sort_index(axis=1)
    combined_groups.columns.names = [group_column, subgroup_column]
    return combined_groups


def get_entity_data(
    dataframe: DataFrame,
    data_columns: List[str],
    entity: str,
    entity_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {*data_columns, entity_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    filtered_data = dataframe.query(f"{entity_column} == @entity")
    if filtered_data.empty:
        raise ArgumentValueError(
            f"No data found for entity '{entity}' in column '{entity_column}'."
        )
    grouped_data = (
        filtered_data.groupby(by=[time_column])[data_columns]
        .agg(agg_func)
        .reset_index()
    )
    grouped_data = grouped_data.set_index(time_column)
    all_times = dataframe[time_column].unique()
    grouped_data = grouped_data.reindex(all_times, fill_value=None)
    grouped_data = grouped_data.sort_index()
    grouped_data = grouped_data.sort_index(axis=1)
    grouped_data.index.name = entity
    return grouped_data


def get_entities_data(
    dataframe: DataFrame,
    data_columns: List[str],
    entity_column: str,
    time_column: str,
    entities: List[str] = None,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {*data_columns, entity_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    entities = entities or dataframe[entity_column].unique()
    entities_data = {}
    for entity in tqdm(entities, desc="Processing entities"):
        try:
            entities_data[entity] = get_entity_data(
                dataframe=dataframe,
                data_columns=data_columns,
                entity=entity,
                entity_column=entity_column,
                time_column=time_column,
                agg_func=agg_func,
            )
            entities_data[entity].columns = pd.MultiIndex.from_product(
                [[entity], entities_data[entity].columns]
            )
        except ArgumentValueError as e:
            raise ArgumentValueError(f"Error processing entity '{entity}': {str(e)}")
    combined_data = pd.concat(entities_data.values(), axis=1)
    combined_data = combined_data.sort_index()
    combined_data = combined_data.sort_index(axis=1)
    combined_data.columns.names = [entity_column, "indicator"]
    return combined_data


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
