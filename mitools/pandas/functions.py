from typing import Iterable, Optional, Union

import pandas as pd
from pandas import DataFrame


def prepare_int_cols(df: DataFrame, cols: Union[Iterable[str],str], 
                     nan_placeholder: int,
                     errors: Optional[str]='coerce') -> DataFrame:
    df[cols] = (df[cols].apply(pd.to_numeric, args=(errors,))
                .fillna(nan_placeholder)
                .astype(int)
    )
    return df

def prepare_str_cols(df: DataFrame, cols: Union[Iterable[str],str]) -> DataFrame:
    df[cols] = df[cols].astype(str)
    return df


def prepare_date_cols(df: DataFrame, cols: Union[Iterable[str],str]) -> DataFrame:
    df[cols] = df[cols].apply(pd.to_datetime)
    return df

def prepare_bool_cols(df: DataFrame, cols: Union[Iterable[str],str]) -> DataFrame:
    df[cols] = df[cols].astype(bool)
    return df
