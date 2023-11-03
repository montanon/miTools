import os
import pandas as pd
import warnings

from numpy import ndarray
from sqlite3 import Connection, OperationalError
from functools import wraps
from typing import Callable, Iterable, Union, Optional
    
    
class CustomConnection(Connection):
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.path = path

    @property
    def __class__(self):
        return Connection

def suppress_user_warning(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return func(*args, **kwargs)
    return wrapper

def check_if_tables(conn: Connection, tablesnames: Iterable[str]):
    checks = []
    for tablename in tablesnames:
        checks.append(check_if_table(conn, tablename))
    return checks

def get_conn_db_folder(conn: Connection):

    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list;")
    result = cursor.fetchone()
    db_path = result[2]

    db_folder = os.path.dirname(db_path)

    return db_folder

def check_if_table(conn: Connection, tablename: str):
    c = conn.cursor()
    try:
        res = conn.cursor().execute("SELECT name FROM sqlite_master")
        c.execute(f'SELECT * FROM "{tablename}"')
        return True if c.fetchone() else False
    except OperationalError:
        res = conn.cursor().execute("SELECT name FROM sqlite_master")
        try:
            db_folder = get_conn_db_folder(conn)
            parquet_folder = os.path.join(db_folder, 'parquet')
            table_file = os.path.join(parquet_folder, f"{tablename}.parquet")
            return True if os.path.exists(table_file) else False
        except Exception:
            return False
        
def connect_to_sql_db(db_path: Union[str, os.PathLike], db_name: str):
    db_path = os.path.join(db_path, db_name)
    return CustomConnection(db_path)

@suppress_user_warning
def read_sql_table(conn: Connection, tablename: str, 
                   columns: Optional[Union[str,list,ndarray]]=None, 
                   index_col: Optional[str]='index'):
    if columns is None:
        return pd.read_sql(f'SELECT * FROM "{tablename}"', conn, index_col=index_col)
    elif isinstance(columns, (list, ndarray)):
        return pd.read_sql(f'SELECT {", ".join(columns)} FROM "{tablename}"', conn)
    elif isinstance(columns, str):
        return pd.read_sql(f'SELECT {columns} FROM "{tablename}"', conn)
    
def read_sql_tables(conn: Connection, tablenames: Iterable[str],
                    columns: Optional[Union[str,list,ndarray]]=None,
                    index_col: Optional[str]='index'):
    return [read_sql_table(conn, tablename, columns, index_col) for tablename in tablenames]

@suppress_user_warning
def transfer_sql_tables(src_db_connection: Connection, dst_db_connection: Connection,
                    tablename: str, if_exists: Optional[str]='fail', 
                    index_col: Optional[str]='index'):
    table = pd.read_sql(
        f"SELECT * FROM {tablename};", src_db_connection, index_col=index_col)
    table.to_sql(tablename, dst_db_connection, if_exists=if_exists)
