from typing import Iterable, List, Optional, Union

import pandas as pd
from pandas import DataFrame
from pandas._libs.tslibs.parsing import DateParseError
from tqdm import tqdm

from ..exceptions.custom_exceptions import (ArgumentTypeError,
                                            ArgumentValueError)

INT_COL_ERROR = 'Value or values in any of columns={} cannnot be converted into int.'
NON_DATE_COL_ERROR = 'Column {} has values that cannot be converted to datetime objects.'


def prepare_int_cols(df: DataFrame, cols: Union[Iterable[str],str], 
                     nan_placeholder: int,
                     errors: Optional[str]='coerce') -> DataFrame:
    try:
        df[cols] = df[cols].apply(pd.to_numeric, args=(errors,))
        df[cols] = df[cols].fillna(nan_placeholder)
    except (ValueError, KeyError):
        raise ArgumentTypeError(INT_COL_ERROR) 
    if not errors == 'ignore':
        df[cols] = df[cols].astype(int)                       
    return df

def prepare_str_cols(df: DataFrame, cols: Union[Iterable[str],str]) -> DataFrame:
    df[cols] = df[cols].astype(str)
    return df

def prepare_date_cols(df: DataFrame, cols: Union[Iterable[str],str]) -> DataFrame:
    try:
        df[cols] = df[cols].apply(pd.to_datetime)
    except (ValueError, DateParseError):
        raise ArgumentValueError(NON_DATE_COL_ERROR)
    return df

def prepare_bool_cols(df: DataFrame, cols: Union[Iterable[str],str]) -> DataFrame:
    df[cols] = df[cols].astype(bool)
    return df

def build_group_subgroup(df: DataFrame, value_col: str, entity: str, group_col: str, 
                         subgroup_col: str, time_col) -> DataFrame:
    group_df = df.query(f'{group_col} == "{entity}"')
    group_subgroups = group_df.groupby(by=[time_col, group_col, subgroup_col])[[value_col]].first().reset_index()
    group_subgroups = group_subgroups.pivot(index=time_col, columns=subgroup_col, values=value_col)
    group_subgroups.index.name = entity
    return group_subgroups

def build_groups_subgroups(df: DataFrame, group_col: str, value_col: str,
                           subgroup_col: str, time_col: str) -> DataFrame:
    groups_subgroups = {}
    for entity in tqdm(df[group_col].unique()):
        groups_subgroups[entity] = build_group_subgroup(df, entity, value_col, group_col, subgroup_col, time_col)
    for entity, subgroups in groups_subgroups.items():
        subgroups.columns = pd.MultiIndex.from_product([[entity], subgroups.columns])
    groups_subgroups = pd.concat(groups_subgroups.values(), axis=1)
    groups_subgroups.columns.names = [group_col, groups_subgroups.columns.names[1]]
    return groups_subgroups

def build_group(df: DataFrame, value_cols: List[str], entity: str,
                 group_col: str, time_col: str) -> DataFrame:
    group_df = df.query(f'{group_col} == "{entity}"')
    group = group_df.groupby(by=[time_col])[[*value_cols]].first().reset_index()
    group.set_index(time_col, inplace=True)
    group.index.name = entity
    return group

def build_groups(df: DataFrame, value_cols: List[str], group_col: str,
                 time_col: str) -> DataFrame:
    groups = {}
    for entity in tqdm(df[group_col].unique()):
        groups[entity] = build_group(df, entity, value_cols, group_col, time_col)
    for entity, group in groups.items():
        group.columns = pd.MultiIndex.from_product([[entity], group.columns])
    return pd.concat(groups.values(), axis=1)

def melt_dataframe(_dataframe: DataFrame, value_col: str, entities: Iterable, group_col: str, 
                   subgroup_col: str, time_col: str) -> DataFrame:
    value_cols = [
        c for c in _dataframe.columns if c[-1].split(' ')[-1] == value_col]
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
        value_name=value_col
        )
    return melted_df
