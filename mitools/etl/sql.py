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
    return db_path.parent


def check_if_table(conn: Connection, tablename: str) -> bool:
    c = conn.cursor()
    try:
        _ = conn.cursor().execute("SELECT name FROM sqlite_master")
        c.execute(f'SELECT * FROM "{tablename}"')
        return True if c.fetchone() else False
    except OperationalError:
        _ = conn.cursor().execute("SELECT name FROM sqlite_master")
        try:
            db_folder = get_conn_db_folder(conn)
            parquet_folder = os.path.join(db_folder, "parquet")
            table_file = os.path.join(parquet_folder, f"{tablename}.parquet")
            return True if os.path.exists(table_file) else False
        except Exception:
            return False


def connect_to_sql_db(db_path: Union[str, os.PathLike], db_name: str) -> Connection:
    db_path = os.path.join(db_path, db_name)
    return CustomConnection(db_path)


@suppress_user_warning
def read_sql_table(
    conn: Connection,
    tablename: str,
    columns: Optional[Union[str, list, ndarray]] = None,
    index_col: Optional[str] = "index",
) -> DataFrame:
    if columns is None:
        return pd.read_sql(f'SELECT * FROM "{tablename}"', conn, index_col=index_col)
    elif isinstance(columns, (list, ndarray)):
        return pd.read_sql(f'SELECT {", ".join(columns)} FROM "{tablename}"', conn)
    elif isinstance(columns, str):
        return pd.read_sql(f'SELECT {columns} FROM "{tablename}"', conn)


def read_sql_tables(
    conn: Connection,
    tablenames: Iterable[str],
    columns: Optional[Union[str, list, ndarray]] = None,
    index_col: Optional[str] = "index",
) -> List[DataFrame]:
    return [
        read_sql_table(conn, tablename, columns, index_col) for tablename in tablenames
    ]


@suppress_user_warning
def transfer_sql_tables(
    src_db_connection: Connection,
    dst_db_connection: Connection,
    tablename: str,
    if_exists: Optional[str] = "fail",
    index_col: Optional[str] = "index",
) -> None:
    table = pd.read_sql(
        f"SELECT * FROM {tablename};", src_db_connection, index_col=index_col
    )
    table.to_sql(tablename, dst_db_connection, if_exists=if_exists)
