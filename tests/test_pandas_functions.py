import os
import unittest
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series, testing

from mitools.exceptions.custom_exceptions import ArgumentTypeError
from mitools.pandas.functions import (
    ArgumentValueError,
    prepare_date_cols,
    prepare_int_cols,
    prepare_str_cols,
    store_dataframe_by_level,
)


class TestPrepareIntCols(unittest.TestCase):

    def setUp(self):
        self.df = DataFrame({
            'col1': ['1', '2', 'three', '4.5', 'NaN'],
            'col2': [10.0, 20.1, 30.2, 'forty', '50']
        })

    def test_single_column_conversion(self):
        modified_df = prepare_int_cols(self.df.copy(), 'col1', nan_placeholder=-1)
        expected = Series([1, 2, -1, 4, -1], name='col1', dtype=int)
        testing.assert_series_equal(modified_df['col1'], expected)

    def test_multiple_column_conversion(self):
        modified_df = prepare_int_cols(self.df.copy(), ['col1', 'col2'], nan_placeholder=-1)
        expected_df = DataFrame({
            'col1': [1, 2, -1, 4, -1],
            'col2': [10, 20, 30, -1, 50]
        }, dtype=int)
        testing.assert_frame_equal(modified_df[['col1', 'col2']], expected_df)

    def test_error_parameter(self):
        # Using 'ignore' should keep original value if conversion fails
        modified_df = prepare_int_cols(self.df.copy(), 'col1', nan_placeholder=-1, errors='ignore')
        self.assertIn('three', modified_df['col1'].values)

    def test_nonexistent_column(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_int_cols(self.df.copy(), 'nonexistent_col', nan_placeholder=-1)


class TestPrepareStrCols(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame
        self.sample_data = DataFrame({
            'A': [1, 2, 3],
            'B': [4.5, 5.5, 6.5],
            'C': ['x', 'y', 'z'],
            'D': [True, False, True]
        })

    def test_convert_to_string(self):
        # Test that all columns are converted to strings
        df = prepare_str_cols(self.sample_data, ['A', 'B', 'C', 'D'])
        for dtype in df.dtypes:
            self.assertEqual(dtype, 'object')
        # Ensure that the values are indeed strings
        for val in df.values.flatten():
            self.assertIsInstance(val, str)

    def test_non_existent_column(self):
        # Test that a KeyError is raised when a non-existent column is passed
        with self.assertRaises(KeyError):
            prepare_str_cols(self.sample_data, ['E'])

    def test_empty_dataframe(self):
        # Test that the function handles an empty DataFrame
        empty_df = DataFrame()
        result_df = prepare_str_cols(empty_df, [])
        self.assertTrue(result_df.empty)

    def test_no_columns_provided(self):
        # Test that the function doesn't change the DataFrame if no columns are provided
        df_copy = self.sample_data.copy()
        result_df = prepare_str_cols(df_copy, [])
        testing.assert_frame_equal(df_copy, result_df)


class TestPrepareDateCols(unittest.TestCase):

    def setUp(self):
        self.df = DataFrame({
            'valid_date': ['2021-01-01', '2021/02/01', '01-03-2021', None],
            'invalid_date': ['2021-01-01', '2021/02/01', 'not a date', None],
            'mixed': [1, '2', '2021-01-01', None]
        })

    def test_valid_dates_conversion(self):
        valid_col = ['valid_date']
        df_converted = prepare_date_cols(self.df.copy(), valid_col)
        expected_df =  self.df.copy(deep=True)
        expected_df[valid_col] = expected_df[valid_col].apply(pd.to_datetime)
        testing.assert_frame_equal(pd.to_datetime(self.df, errors='coerce')[['valid_date']],
                                   df_converted[['valid_date']])

    def test_error_on_invalid_dates(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_cols(self.df.copy(), 'invalid_date')

    def test_error_on_mixed_data(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_cols(self.df.copy(), 'mixed')


class TestStoreDataframeByLevel(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame({
            ('A', 'a'): [1, 2, 3],
            ('B', 'b'): [4, 5, 6],
            ('C', 'c'): [7, 8, 9]
        })
        self.base_path = 'test.parquet'

    def tearDown(self):
        # Clean up any created files
        for path in Path('.').glob('test*_sub.parquet'):
            os.remove(path)

    def test_invalid_df(self):
        with self.assertRaises(Exception):
            store_dataframe_by_level('invalid', self.base_path, 0)

    def test_invalid_base_path(self):
        with self.assertRaises(Exception):
            store_dataframe_by_level(self.df, 123, 0)

    def test_invalid_level(self):
        with self.assertRaises(Exception):
            store_dataframe_by_level(self.df, self.base_path, 'invalid')

    def test_valid_inputs(self):
        store_dataframe_by_level(self.df, self.base_path, 0)
        self.assertTrue(Path('test0_sub.parquet').exists())
        self.assertTrue(Path('test1_sub.parquet').exists())
        self.assertTrue(Path('test2_sub.parquet').exists())


if __name__ == '__main__':
    unittest.main()