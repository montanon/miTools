import unittest
from unittest.mock import Mock, patch, MagicMock
from unittest import TestCase
from mitools.etl import *
import os
import pandas as pd
import sqlite3
import warnings


class TestCustomConnection(TestCase):

    def setUp(self):
        self.path = "sample_path"
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
        self.path = '.test_db'
        self.conn = connect_to_sql_db('./', self.path)
        self.conn.cursor().execute("CREATE TABLE table1(id)")
        self.conn.execute('INSERT INTO table1 VALUES ("abc")')

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_table_exists_in_db(self):
        self.assertTrue(check_if_table(self.conn, 'table1'))

    def test_table_nor_parquet_exists(self):
        self.assertFalse(check_if_table(self.conn, "table2"))


class TestCheckIfTables(TestCase):

    def setUp(self):
        self.path = '.test_db'
        self.conn = connect_to_sql_db('./', self.path)
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
        self.conn = Mock(spec=Connection)
        self.cursor = self.conn.cursor()

    @patch('os.path.dirname', return_value="/path/to/db_folder")
    def test_valid_path(self, mock_dirname):
        self.cursor.fetchone.return_value = [None, None, "/path/to/db.sqlite"]
        db_folder = get_conn_db_folder(self.conn)
        self.assertEqual(db_folder, "/path/to/db_folder")

    def test_none_path(self):
        self.cursor.fetchone.return_value = None  # None is returned from fetchone
        with self.assertRaises(TypeError):  # This would cause TypeError when trying to access result[2]
            get_conn_db_folder(self.conn)

    def test_invalid_result(self):
        self.cursor.fetchone.return_value = [None, None]  # Missing the third element
        with self.assertRaises(IndexError):  # This would cause IndexError
            get_conn_db_folder(self.conn)

    def test_exception_while_executing_sql(self):
        self.cursor.execute.side_effect = Exception("Some error")  # Simulating some database error
        with self.assertRaises(Exception) as context:
            get_conn_db_folder(self.conn)
        self.assertEqual(str(context.exception), "Some error")


class TestConnectToSqlDb(TestCase):

    def setUp(self):
        self.path = '.test_db'
        self.conn = connect_to_sql_db('./', self.path)

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_successful_connection(self):
        conn = connect_to_sql_db('./', self.path)
        self.assertIsInstance(conn, CustomConnection) 


class TestReadSqlTable(TestCase):

    def setUp(self):
        self.conn = Mock(spec=Connection)

    @patch('pandas.read_sql')
    def test_read_full_table(self, mock_read_sql):
        tablename = "sample_table"
        mock_read_sql.return_value = "dataframe_result"  # Mock return value for pd.read_sql
        result = read_sql_table(self.conn, tablename)
        mock_read_sql.assert_called_once_with(f'SELECT * FROM "{tablename}"', self.conn, index_col='index')
        self.assertEqual(result, "dataframe_result")

    @patch('pandas.read_sql')
    def test_read_specific_columns_list(self, mock_read_sql):
        tablename = "sample_table"
        columns = ["col1", "col2"]
        mock_read_sql.return_value = "dataframe_result"
        result = read_sql_table(self.conn, tablename, columns=columns)
        mock_read_sql.assert_called_once_with(f'SELECT col1, col2 FROM "{tablename}"', self.conn)
        self.assertEqual(result, "dataframe_result")

    @patch('pandas.read_sql')
    def test_read_specific_column_string(self, mock_read_sql):
        tablename = "sample_table"
        columns = "col1"
        mock_read_sql.return_value = "dataframe_result"
        result = read_sql_table(self.conn, tablename, columns=columns)
        mock_read_sql.assert_called_once_with(f'SELECT col1 FROM "{tablename}"', self.conn)
        self.assertEqual(result, "dataframe_result")

    def test_unexpected_columns_type(self):
        tablename = "sample_table"
        columns = 123  # Intentionally wrong type
        result = read_sql_table(self.conn, tablename, columns=columns)
        self.assertIsNone(result)


class TestReadSqlTables(TestCase):

    def setUp(self):
        self.path = '.test_db'
        self.conn = connect_to_sql_db('./', self.path)
        self.conn.cursor().execute("CREATE TABLE table1(id)")
        self.conn.execute('INSERT INTO table1 VALUES ("abc")')
        self.conn.cursor().execute("CREATE TABLE table2(id)")
        self.conn.execute('INSERT INTO table2 VALUES ("abc")')

    def tearDown(self):
        # Delete the 'sample_file' after the test is done
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_read_existing_sql_tables(self):
        tablenames = ['table1', 'table2']
        results = read_sql_tables(self.conn, tablenames, index_col=None)
        self.assertEqual(len(results), len(tablenames))  # There should be 2 results, one for each table
        for result, tablename in zip(results, tablenames):
            self.assertTrue(
               pd.read_sql(f'SELECT * FROM "{tablename}"', self.conn).equals(result)
            )


class TestTransferSqlTables(TestCase):

    def setUp(self):
        # Set up two in-memory SQLite databases
        self.src_conn = sqlite3.connect(":memory:")
        self.dst_conn = sqlite3.connect(":memory:")
        # Create a sample table in the source database
        self.src_conn.execute('''
            CREATE TABLE sample (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        ''')
        self.src_conn.executemany('''
            INSERT INTO sample (data) VALUES (?)
        ''', [('a',), ('b',), ('c',)])

    def tearDown(self):
        # Clean up the connections
        self.src_conn.close()
        self.dst_conn.close()

    def test_transfer_sql_tables(self):
        # Given
        tablename = 'sample'
        # Transfer table from src to dst
        transfer_sql_tables(self.src_conn, self.dst_conn, tablename, if_exists='replace', index_col=None)
        # Verify the data was transferred
        df_dst = pd.read_sql('SELECT * FROM sample', self.dst_conn)
        self.assertEqual(len(df_dst), 3)
        self.assertTrue('data' in df_dst.columns)
        self.assertTrue('index' in df_dst.columns)


if __name__ == "__main__":
    unittest.main()
