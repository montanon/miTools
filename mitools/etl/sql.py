import os
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, OperationalError
from typing import Iterable, List, Optional, Union

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from ..utils import suppress_user_warning


class CustomConnection(Connection):
    def __init__(self, path: PathLike, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.path = Path(path).absolute()

    @property
    def __class__(self):
        return Connection


class MainConnection(CustomConnection):
    _instances = {}

    def __new__(cls, path: PathLike):
        path = Path(path).absolute()
        if path not in cls._instances:
            cls._instances[path] = super(MainConnection, cls).__new__(cls)
            cls._instances[path]._initialized = False
        return cls._instances[path]

    def __init__(self, path: PathLike, *args, **kwargs):
        if not self._initialized:
            super().__init__(path, *args, **kwargs)
            self._initialized = True


def check_if_tables(conn: Connection, tables_names: Iterable[str]) -> List[bool]:
    return [check_if_table(conn, table_name) for table_name in tables_names]


def get_conn_db_folder(conn: Connection) -> PathLike:
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list;")
    db_path = Path(cursor.fetchone()[2])
    return db_path.parent.absolute()


def check_if_table(conn: Connection, table_name: str) -> bool:
    query = (
        f'SELECT name FROM sqlite_master WHERE type="table" AND name="{table_name}";'
    )
    cursor = conn.cursor()
    try:
        return cursor.execute(query).fetchone() is not None
    except OperationalError:
        try:
            parquet_folder = get_conn_db_folder(conn) / "parquet"
            return (parquet_folder / f"{table_name}.parquet").exists()
        except Exception as e:
            return False


def connect_to_sql_db(db_path: Union[str, PathLike], db_name: str) -> CustomConnection:
    db_path = Path(db_path) / db_name
    return CustomConnection(db_path)


@suppress_user_warning
def read_sql_table(
    conn: Connection,
    table_name: str,
    columns: Union[str, List[str], ndarray] = None,
    index_col: str = "index",
) -> DataFrame:
    if columns is None:
        query = f'SELECT * FROM "{table_name}";'
    elif isinstance(columns, (list, ndarray)):
        query = f'SELECT {", ".join(columns)} FROM "{table_name}";'
    elif isinstance(columns, str):
        query = f'SELECT {columns} FROM "{table_name}";'
    else:
        raise ValueError("Invalid column specification")

    return pd.read_sql(query, conn, index_col=index_col)


def read_sql_tables(
    conn: Connection,
    table_names: Iterable[str],
    columns: Union[str, List[str], ndarray] = None,
    index_col: str = "index",
) -> List[DataFrame]:
    return [read_sql_table(conn, name, columns, index_col) for name in table_names]


@suppress_user_warning
def transfer_sql_table(
    src_conn: Connection,
    dst_conn: Connection,
    table_name: str,
    if_exists: str = "fail",
    index_col: str = "index",
) -> None:
    query = f"SELECT * FROM {table_name};"
    table = pd.read_sql(query, src_conn, index_col=index_col)
    table.to_sql(table_name, dst_conn, if_exists=if_exists)
