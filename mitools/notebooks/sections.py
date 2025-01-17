from os import PathLike
from pathlib import Path
from typing import Dict, List, NewType, Optional, Pattern

import pandas as pd
from pandas import DataFrame

from mitools.context import DISPLAY
from mitools.exceptions import ArgumentKeyError
from mitools.notebooks.objects import NotebookSection
from mitools.utils import iprint

FULL_TEXT_COLUMN = "full_text"
TEXT_COLUMNS = ["Title", "Abstract"]
LENS_ARTICLES_COLS_MAP = {
    "Lens ID": "lens_id",
    "Title": "title",
    "Date Published": "date_published",
    "Publication Year": "year_published",
    "Publication Type": "publication_type",
    "Source Title": "source_title",
    "ISSNs": "issns",
    "Publisher": "publisher",
    "Source Country": "source_country",
    "Author/s": "authors",
    "Abstract": "abstract",
    "Volume": "volume",
    "Issue Number": None,
    "Start Page": "start_page",
    "End Page": "end_page",
    "Fields of Study": "fields_of_study",
    "Keywords": "keywords",
    "MeSH Terms": "mesh_terms",
    "Chemicals": None,
    "Funding": None,
    "Source URLs": "source_urls",
    "External URL": None,
    "PMID": "pmid",
    "DOI": "doi",
    "Microsoft Academic ID": "magid",
    "PMCID": "pmcid",
    "Citing Patents Count": None,
    "References": "references",
    "References Count": "references_count",
    "Citing Works Count": "scholarly_citations_count",
    "Is Open Access": "isopen_access",
    "Open Access License": None,
    "Open Access Colour": "open_access",
}
INVALID_COLUMN_ERROR = "Columns {} are not in the DataFrame"

RenameColumnsMap = NewType("RenameColumnsMap", Dict[str, str])


def read_and_concat_csvs(folder: PathLike, axis: Optional[int] = 0) -> DataFrame:
    csv_files = [
        Path(folder) / file
        for file in Path(folder).iterdir()
        if file.is_file() and file.suffix == ".csv"
    ]
    csv_dfs = [pd.read_csv(file, index_col=0) for file in csv_files]
    combined_df = pd.concat(csv_dfs, axis=axis)
    return combined_df


def create_full_text_column(dataframe: DataFrame, text_columns: List[str]) -> DataFrame:
    if not all([c in dataframe for c in text_columns]):
        raise ArgumentKeyError(INVALID_COLUMN_ERROR.format(text_columns))
    dataframe[FULL_TEXT_COLUMN] = dataframe[text_columns].apply(
        lambda row: " ".join(str(value) for value in row.values), axis=1
    )
    return dataframe


def rename_columns(
    dataframe: DataFrame,
    columns_map: RenameColumnsMap,
    inverse_map: Optional[bool] = False,
) -> DataFrame:
    if inverse_map:
        columns_map = {v: k for k, v in columns_map.items() if v is not None}
    unique_names = len(set(columns_map.values())) == len(columns_map)
    # all_in_dataframe = all([c in dataframe.columns for c in columns_map.keys()])
    no_preexisting_names = all(
        val not in dataframe.columns for val in columns_map.values()
    )
    if not all([unique_names, no_preexisting_names]):
        raise ArgumentKeyError(INVALID_COLUMN_ERROR.format(columns_map.keys()))
    return dataframe.rename(columns=columns_map)


def filter_text_rows_by_pattern(
    dataframe: DataFrame,
    filter_col: str,
    pattern: Pattern,
    case: Optional[bool] = False,
) -> DataFrame:
    if filter_col not in dataframe.columns:
        raise ArgumentKeyError(INVALID_COLUMN_ERROR.format(filter_col))
    condition = dataframe[filter_col]
    condition = condition.str.contains(pattern, case=case, regex=True)
    return dataframe[condition]


def merge_csvs_into_dataframe(csvs_folder: PathLike) -> DataFrame:
    dataframe = (
        read_and_concat_csvs(csvs_folder).drop_duplicates().reset_index(drop=True)
    )
    return dataframe


def etl(
    df_path, csvs_folder, columns_map, text_columns, pattern, filter_col, recalculate
):
    if not Path(df_path).exists() or recalculate:
        dataframe = merge_csvs_into_dataframe(csvs_folder)
        if columns_map:
            dataframe = rename_columns(dataframe, columns_map)
        dataframe = create_full_text_column(dataframe, text_columns)
        dataframe = filter_text_rows_by_pattern(dataframe, filter_col, pattern)
        dataframe.to_parquet(df_path)
    else:
        dataframe = pd.read_parquet(df_path)
    return dataframe
