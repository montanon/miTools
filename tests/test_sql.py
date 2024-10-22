import os
import sqlite3
import unittest
import warnings
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from mitools.etl import (
    Connection,
    CustomConnection,
    MainConnection,
    check_if_table,
    check_if_tables,
    connect_to_sql_db,
    get_conn_db_folder,
    read_sql_table,
    read_sql_tables,
    suppress_user_warning,
    transfer_sql_table,
)


class TestMainConnection(TestCase):
    def setUp(self):
        self.conn1 = Path("./tests/.test_assets/conn1.db")
        self.conn2 = Path("./tests/.test_assets/conn2.db")
        self.dummy_db_path = CustomConnection(self.conn1)
        self.dummy_db_path = CustomConnection(self.conn2)

    def tearDown(self):
        self.conn1.unlink()
        self.conn2.unlink()

    def test_singleton_property(self):
        conn1 = MainConnection(self.conn1)
        conn2 = MainConnection(self.conn1)
        self.assertIs(conn1, conn2)

    def test_different_instances(self):
        conn1 = MainConnection(self.conn1)
        conn2 = MainConnection(self.conn2)
        self.assertIsNot(conn1, conn2)

    def test_path_normalization(self):
        absolute_path = Path(self.conn1).absolute()
        conn1 = MainConnection(self.conn1)
        conn2 = MainConnection(absolute_path)
        self.assertIs(conn1, conn2)

    def test_initialization_check(self):
        conn1 = MainConnection(self.conn1)
        with self.assertRaises(AttributeError):
            conn1.some_random_attribute
        conn1.some_random_attribute = True
        conn2 = MainConnection(self.conn1)
        self.assertTrue(hasattr(conn2, "some_random_attribute"))


class TestCustomConnection(TestCase):
    def setUp(self):
        self.path = Path("sample_path").absolute()
        self.conn = CustomConnection(self.path)

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_initialization(self):
        # Check that the CustomConnection object is instantiated
        self.assertIsInstance(self.conn, CustomConnection)

    def test_path_attribute(self):
        # Ensure that the 'path' attribute is correctly set during instantiation
        self.assertEqual(self.conn.path, self.path)

    def test_class_property(self):
        # Ensure the __class__ property correctly returns Connection
        self.assertIs(self.conn.__class__, Connection)


class TestSuppressUserWarningDecorator(TestCase):
    def test_suppresses_user_warning(self):
        @suppress_user_warning
        def warn_user():
            warnings.warn("This is a user warning", category=UserWarning)
            return "Function Completed"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = warn_user()
            self.assertEqual(result, "Function Completed")
            self.assertEqual(len(w), 0)

    def test_does_not_suppress_other_warnings(self):
        @suppress_user_warning
        def warn_deprecated():
            warnings.warn("This is a deprecated warning", category=DeprecationWarning)
            return "Function Completed"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = warn_deprecated()
            self.assertEqual(result, "Function Completed")
            self.assertNotEqual(len(w), 0)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_function_functionality(self):
        @suppress_user_warning
        def some_function(x, y):
            return x + y

        self.assertEqual(some_function(2, 3), 5)


class TestCheckIfTable(TestCase):
    def setUp(self):
        self.path = ".test_db"
        self.conn = connect_to_sql_db("./", self.path)
        self.conn.cursor().execute("CREATE TABLE table1(id)")
        self.conn.execute('INSERT INTO table1 VALUES ("abc")')

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_table_exists_in_db(self):
        self.assertTrue(check_if_table(self.conn, "table1"))

    def test_table_nor_parquet_exists(self):
        self.assertFalse(check_if_table(self.conn, "table2"))


class TestCheckIfTables(TestCase):
    def setUp(self):
        self.path = ".test_db"
        self.conn = connect_to_sql_db("./", self.path)
        self.conn.cursor().execute("CREATE TABLE table1(id)")
        self.conn.execute('INSERT INTO table1 VALUES ("abc")')
        self.conn.cursor().execute("CREATE TABLE table2(id)")
        self.conn.execute('INSERT INTO table2 VALUES ("abc")')

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_all_tables_exist(self):
        tables = ["table1", "table2"]
        result = check_if_tables(self.conn, tables)
        self.assertEqual(result, [True, True])

    def test_some_tables_exist(self):
        tables = ["table1", "table2", "table3"]
        result = check_if_tables(self.conn, tables)
        self.assertEqual(result, [True, True, False])

    def test_no_tables_exist(self):
        tables = ["table3", "table4", "table5"]
        result = check_if_tables(self.conn, tables)
        self.assertEqual(result, [False, False, False])

    def test_empty_table_list(self):
        tables = []
        result = check_if_tables(self.conn, tables)
        self.assertEqual(result, [])


class TestGetConnDbFolder(TestCase):
    def setUp(self):
        self.db_path = Path("./test_db_sqlite")
        self.conn = sqlite3.connect(self.db_path)

    def tearDown(self):
        self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink()  # Remove the temporary database file

    def test_get_conn_db_folder(self):
        db_folder = get_conn_db_folder(self.conn)
        expected_folder = self.db_path.parent.absolute()
        self.assertEqual(db_folder, expected_folder)

    def test_in_memory_database(self):
        with sqlite3.connect(":memory:") as in_memory_conn:
            db_folder = get_conn_db_folder(in_memory_conn)
            self.assertEqual(db_folder, Path(".").absolute())


class TestConnectToSqlDb(TestCase):
    def setUp(self):
        self.path = ".test_db"
        self.conn = connect_to_sql_db("./", self.path)

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_successful_connection(self):
        conn = connect_to_sql_db("./", self.path)
        self.assertIsInstance(conn, CustomConnection)


class TestReadSqlTable(TestCase):
    def setUp(self):
        self.db_path = Path("./test_temp_db.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER
            );
            """
        )
        self.conn.executemany(
            "INSERT INTO test_table (name, age) VALUES (?, ?);",
            [("Alice", 25), ("Bob", 30), ("Charlie", 35)],
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink()  # Remove the temporary database file

    def test_read_full_table(self):
        expected = DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        ).set_index("id")

        result = read_sql_table(self.conn, "test_table", index_col="id")
        assert_frame_equal(result, expected)

    def test_read_specific_columns(self):
        expected = DataFrame({"name": ["Alice", "Bob", "Charlie"]})

        result = read_sql_table(self.conn, "test_table", columns=["name"])
        from IPython.display import display

        display(result)
        assert_frame_equal(result, expected)

    def test_read_single_column_as_string(self):
        expected = DataFrame({"age": [25, 30, 35]})
        result = read_sql_table(self.conn, "test_table", columns="age")
        assert_frame_equal(result, expected)

    def test_invalid_column_specification(self):
        with self.assertRaises(ValueError):
            read_sql_table(self.conn, "test_table", columns={"invalid": "dict"})

    def test_non_existent_table(self):
        with self.assertRaises(pd.io.sql.DatabaseError):
            read_sql_table(self.conn, "non_existent_table")


class TestReadSqlTables(TestCase):
    def setUp(self):
        self.path = ".test_db"
        self.conn = connect_to_sql_db("./", self.path)
        self.conn.cursor().execute("CREATE TABLE table1(id)")
        self.conn.execute('INSERT INTO table1 VALUES ("abc")')
        self.conn.cursor().execute("CREATE TABLE table2(id)")
        self.conn.execute('INSERT INTO table2 VALUES ("abc")')

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_read_existing_sql_tables(self):
        tablenames = ["table1", "table2"]
        results = read_sql_tables(self.conn, tablenames, index_col=None)
        self.assertEqual(
            len(results), len(tablenames)
        )  # There should be 2 results, one for each table
        for result, tablename in zip(results, tablenames):
            self.assertTrue(
                pd.read_sql(f'SELECT * FROM "{tablename}"', self.conn).equals(result)
            )


class TestTransferSqlTable(TestCase):
    def setUp(self):
        # Set up two in-memory SQLite databases
        self.src_conn = sqlite3.connect(":memory:")
        self.dst_conn = sqlite3.connect(":memory:")
        # Create a sample table in the source database
        self.src_conn.execute("""
            CREATE TABLE sample (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        self.src_conn.executemany(
            """
            INSERT INTO sample (data) VALUES (?)
        """,
            [("a",), ("b",), ("c",)],
        )

    def tearDown(self):
        # Clean up the connections
        self.src_conn.close()
        self.dst_conn.close()

    def test_transfer_sql_table(self):
        # Given
        tablename = "sample"
        # Transfer table from src to dst
        transfer_sql_table(
            self.src_conn, self.dst_conn, tablename, if_exists="replace", index_col=None
        )
        # Verify the data was transferred
        df_dst = pd.read_sql("SELECT * FROM sample", self.dst_conn)
        self.assertEqual(len(df_dst), 3)
        self.assertTrue("data" in df_dst.columns)
        self.assertTrue("index" in df_dst.columns)


if __name__ == "__main__":
    unittest.main()
