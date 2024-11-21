import os
import unittest
from pathlib import Path
from unittest import TestCase

import pandas as pd
from pandas import DataFrame, Series, testing

from mitools.exceptions.custom_exceptions import ArgumentTypeError
from mitools.pandas.functions import (
    ArgumentValueError,
    idxslice,
    load_level_destructured_dataframe,
    prepare_date_cols,
    prepare_int_cols,
    prepare_str_cols,
    store_dataframe_by_level,
)


class TestPrepareIntCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )

    def test_single_column_conversion(self):
        result = prepare_int_cols(self.df.copy(), cols="col1", nan_placeholder=0)
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 0],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_columns_conversion(self):
        result = prepare_int_cols(
            self.df.copy(), cols=["col1", "col2"], nan_placeholder=0
        )
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 0],
                "col2": [4, 0, 6, 0],
                "col3": [None, None, None, None],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_column_with_only_nans(self):
        result = prepare_int_cols(self.df.copy(), cols="col3", nan_placeholder=99)
        expected = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [99, 99, 99, 99],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_column_not_in_dataframe(self):
        with self.assertRaises(ArgumentValueError):
            prepare_int_cols(self.df.copy(), cols="nonexistent_col", nan_placeholder=0)

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_int_cols(self.df.copy(), cols=123, nan_placeholder=0)

    def test_invalid_iterable_column_type(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_int_cols(self.df.copy(), cols=[1, 2, 3], nan_placeholder=0)

    def test_ignore_errors(self):
        result = prepare_int_cols(
            self.df.copy(), cols="col2", nan_placeholder=0, errors="ignore"
        )
        expected = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_invalid_error_handling(self):
        with self.assertRaises(ArgumentValueError):
            prepare_int_cols(
                self.df.copy(), cols="col2", nan_placeholder=0, errors="invalid_option"
            )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        result = prepare_int_cols(empty_df, cols=[], nan_placeholder=0)
        pd.testing.assert_frame_equal(result, empty_df)

    def test_empty_columns(self):
        result = prepare_int_cols(self.df, cols=[], nan_placeholder=0)
        pd.testing.assert_frame_equal(result, self.df)

    def test_no_conversion_needed(self):
        df = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = prepare_int_cols(df.copy(), cols="col1", nan_placeholder=0)
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
        result = prepare_int_cols(
            df.copy(), cols="col1", nan_placeholder=0, errors="ignore"
        )
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
        result = prepare_int_cols(
            df.copy(), cols="col1", nan_placeholder=0, errors="raise"
        )
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_custom_nan_placeholder(self):
        result = prepare_int_cols(self.df.copy(), cols="col1", nan_placeholder=999)
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 999],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        pd.testing.assert_frame_equal(result, expected)


class TestPrepareStrCols(TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [4.5, 5.5, None], "col3": ["a", "b", "c"]}
        )

    def test_single_column_conversion(self):
        result = prepare_str_cols(self.df.copy(), cols="col1")
        self.assertTrue((result["col1"] == ["1", "2", "3"]).all())
        self.assertEqual(result["col1"].dtype, "object")

    def test_multiple_column_conversion(self):
        result = prepare_str_cols(self.df.copy(), cols=["col1", "col2"])
        self.assertTrue((result["col1"] == ["1", "2", "3"]).all())
        self.assertTrue((result["col2"] == ["4.5", "5.5", "nan"]).all())
        self.assertEqual(result["col1"].dtype, "object")
        self.assertEqual(result["col2"].dtype, "object")

    def test_no_conversion_needed(self):
        result = prepare_str_cols(self.df.copy(), cols="col3")
        self.assertTrue((result["col3"] == ["a", "b", "c"]).all())
        self.assertEqual(result["col3"].dtype, "object")

    def test_nonexistent_column(self):
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_cols(self.df.copy(), cols="nonexistent_col")
        self.assertIn("Columns ['nonexistent_col'] not found", str(context.exception))

    def test_mixed_column_input(self):
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_cols(self.df.copy(), cols=["col1", "nonexistent_col"])
        self.assertIn("Columns ['nonexistent_col'] not found", str(context.exception))

    def test_invalid_cols_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_str_cols(self.df.copy(), cols=123)
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_invalid_cols_elements(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_str_cols(self.df.copy(), cols=["col1", 123])
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_empty_cols_list(self):
        result = prepare_str_cols(self.df.copy(), cols=[])
        pd.testing.assert_frame_equal(result, self.df)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_cols(empty_df, cols="col1")
        self.assertIn("Columns ['col1'] not found in DataFrame", str(context.exception))

    def test_preserves_other_columns(self):
        result = prepare_str_cols(self.df.copy(), cols="col1")
        pd.testing.assert_frame_equal(
            result[["col2", "col3"]], self.df[["col2", "col3"]]
        )

    def test_large_dataframe(self):
        large_df = pd.DataFrame(
            {
                "col1": range(10000),
                "col2": [str(x) for x in range(10000)],
            }
        )
        result = prepare_str_cols(large_df, cols="col1")
        self.assertTrue((result["col1"] == large_df["col1"].astype(str)).all())


class TestPrepareDateCols(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "valid_date": ["2021-01-01", "2021/02/01", "01-03-2021", None],
                "invalid_date": ["2021-01-01", "2021/02/01", "not a date", None],
                "mixed": [1, "2", "2021-01-01", None],
            }
        )

    def test_valid_dates_conversion(self):
        valid_col = ["valid_date"]
        df_converted = prepare_date_cols(self.df.copy(), valid_col)
        expected_df = self.df.copy(deep=True)
        expected_df[valid_col] = expected_df[valid_col].apply(pd.to_datetime)
        testing.assert_frame_equal(
            pd.to_datetime(self.df, errors="coerce")[["valid_date"]],
            df_converted[["valid_date"]],
        )

    def test_error_on_invalid_dates(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_cols(self.df.copy(), "invalid_date")

    def test_error_on_mixed_data(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_cols(self.df.copy(), "mixed")


class TestStoreDataframeByLevel(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame(
            {("A", "a"): [1, 2, 3], ("B", "b"): [4, 5, 6], ("C", "c"): [7, 8, 9]}
        )
        self.base_path = "test.parquet"

    def tearDown(self):
        # Clean up any created files
        for path in Path(".").glob("test*_sub.parquet"):
            os.remove(path)

    def test_invalid_df(self):
        with self.assertRaises(Exception):
            store_dataframe_by_level("invalid", self.base_path, 0)

    def test_invalid_base_path(self):
        with self.assertRaises(Exception):
            store_dataframe_by_level(self.df, 123, 0)

    def test_invalid_level(self):
        with self.assertRaises(Exception):
            store_dataframe_by_level(self.df, self.base_path, "invalid")

    def test_valid_inputs(self):
        store_dataframe_by_level(self.df, self.base_path, 0)
        self.assertTrue(Path("test0_sub.parquet").exists())
        self.assertTrue(Path("test1_sub.parquet").exists())
        self.assertTrue(Path("test2_sub.parquet").exists())


class TestLoadLevelDestructuredDataframe(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame(
            {("A", "a"): [1, 2, 3], ("B", "b"): [4, 5, 6], ("C", "c"): [7, 8, 9]}
        )
        self.df.columns = pd.MultiIndex.from_tuples(self.df.columns)
        self.base_path = "test.parquet"
        store_dataframe_by_level(self.df, self.base_path, 0)

    def tearDown(self):
        # Clean up any created files
        for path in Path(".").glob("test*_sub.parquet"):
            os.remove(path)

    def test_invalid_base_path(self):
        with self.assertRaises(ValueError):
            load_level_destructured_dataframe(123, 0)

    def test_invalid_level(self):
        with self.assertRaises(ValueError):
            load_level_destructured_dataframe(self.base_path, "invalid")

    def test_no_files_found(self):
        with self.assertRaises(FileNotFoundError):
            load_level_destructured_dataframe("nonexistent.parquet", 0)

    def test_valid_inputs(self):
        df = load_level_destructured_dataframe(self.base_path, 0)
        self.assertTrue(isinstance(df, DataFrame))
        self.assertEqual(df.shape, self.df.shape)


class TestIdxSlice(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame(
            {("A", "a"): [1, 2, 3], ("B", "b"): [4, 5, 6], ("C", "c"): [7, 8, 9]}
        )
        self.df.columns = pd.MultiIndex.from_tuples(self.df.columns)
        self.df.columns.names = ["level_0", "level_1"]

    def test_invalid_axis(self):
        with self.assertRaises(ValueError):
            idxslice(self.df, 0, "A", 2)

    def test_invalid_level(self):
        with self.assertRaises(ValueError):
            idxslice(self.df, "D", "A", 1)

    def test_valid_inputs(self):
        result = idxslice(self.df, "level_0", "A", 1)
        self.assertEqual(result, pd.IndexSlice[["A"], :])
        result = idxslice(self.df, 0, "A", 1)
        self.assertEqual(result, pd.IndexSlice[["A"], :])
        result = idxslice(self.df, "level_1", "b", 1)
        self.assertEqual(result, pd.IndexSlice[:, ["b"]])
        result = idxslice(self.df, 1, "b", 1)
        self.assertEqual(result, pd.IndexSlice[:, ["b"]])


if __name__ == "__main__":
    unittest.main()
