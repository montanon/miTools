import os
import shutil
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, testing
from pandas.testing import assert_frame_equal

from mitools.exceptions.custom_exceptions import ArgumentTypeError, ArgumentValueError
from mitools.pandas.functions import (
    get_entities_data,
    get_entity_data,
    idxslice,
    load_dataframe_parquet,
    long_to_wide_dataframe,
    prepare_bool_cols,
    prepare_date_cols,
    prepare_int_cols,
    prepare_str_cols,
    reshape_group_data,
    reshape_groups_subgroups,
    store_dataframe_parquet,
    wide_to_long_dataframe,
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
        assert_frame_equal(result, expected)

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
        assert_frame_equal(result, expected)

    def test_column_with_only_nans(self):
        result = prepare_int_cols(self.df.copy(), cols="col3", nan_placeholder=99)
        expected = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [99, 99, 99, 99],
            }
        )
        assert_frame_equal(result, expected)

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
        assert_frame_equal(result, expected)

    def test_invalid_error_handling(self):
        with self.assertRaises(ArgumentValueError):
            prepare_int_cols(
                self.df.copy(), cols="col2", nan_placeholder=0, errors="invalid_option"
            )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        result = prepare_int_cols(empty_df, cols=[], nan_placeholder=0)
        assert_frame_equal(result, empty_df)

    def test_empty_columns(self):
        result = prepare_int_cols(self.df, cols=[], nan_placeholder=0)
        assert_frame_equal(result, self.df)

    def test_no_conversion_needed(self):
        df = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = prepare_int_cols(df.copy(), cols="col1", nan_placeholder=0)
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert_frame_equal(result, expected, check_dtype=False)
        result = prepare_int_cols(
            df.copy(), cols="col1", nan_placeholder=0, errors="ignore"
        )
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert_frame_equal(result, expected, check_dtype=False)
        result = prepare_int_cols(
            df.copy(), cols="col1", nan_placeholder=0, errors="raise"
        )
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert_frame_equal(result, expected, check_dtype=False)

    def test_custom_nan_placeholder(self):
        result = prepare_int_cols(self.df.copy(), cols="col1", nan_placeholder=999)
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 999],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        assert_frame_equal(result, expected)


class TestPrepareStrCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
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
        assert_frame_equal(result, self.df)

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_cols(empty_df, cols="col1")
        self.assertIn("Columns ['col1'] not found in DataFrame", str(context.exception))

    def test_preserves_other_columns(self):
        result = prepare_str_cols(self.df.copy(), cols="col1")
        assert_frame_equal(result[["col2", "col3"]], self.df[["col2", "col3"]])

    def test_large_dataframe(self):
        large_df = DataFrame(
            {
                "col1": range(10000),
                "col2": [str(x) for x in range(10000)],
            }
        )
        result = prepare_str_cols(large_df, cols="col1")
        self.assertTrue((result["col1"] == large_df["col1"].astype(str)).all())


class TestPrepareDateCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "valid_dates": ["2021-01-01", "2021-02-01", None],
                "invalid_dates": ["invalid_date", "2021-01-01", None],
                "mixed_dates": ["2021-01-01", "invalid_date", "2021-02-01"],
            }
        )

    def test_basic_conversion(self):
        result = prepare_date_cols(
            self.df.copy(), cols="valid_dates", nan_placeholder="2000-01-01"
        )
        self.assertTrue(isinstance(result["valid_dates"].iloc[0], pd.Timestamp))
        self.assertEqual(result["valid_dates"].iloc[2], pd.Timestamp("2000-01-01"))

    def test_multiple_column_conversion(self):
        result = prepare_date_cols(
            self.df.copy(),
            cols=["valid_dates", "mixed_dates"],
            nan_placeholder="2000-01-01",
        )
        self.assertTrue(isinstance(result["valid_dates"].iloc[0], pd.Timestamp))
        self.assertTrue(isinstance(result["mixed_dates"].iloc[0], pd.Timestamp))
        self.assertEqual(result["mixed_dates"].iloc[1], pd.Timestamp("2000-01-01"))

    def test_invalid_date_handling_coerce(self):
        result = prepare_date_cols(
            self.df.copy(),
            cols="invalid_dates",
            nan_placeholder="2000-01-01",
            errors="coerce",
        )
        self.assertEqual(result["invalid_dates"].iloc[0], pd.Timestamp("2000-01-01"))

    def test_invalid_date_handling_ignore(self):
        result = prepare_date_cols(
            self.df.copy(),
            cols="invalid_dates",
            nan_placeholder="2000-01-01",
            errors="ignore",
        )
        self.assertEqual(result["invalid_dates"].iloc[0], "invalid_date")
        self.assertTrue(pd.isna(result["invalid_dates"].iloc[2]))

    def test_invalid_date_handling_raise(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_date_cols(
                self.df.copy(),
                cols="invalid_dates",
                nan_placeholder="2000-01-01",
                errors="raise",
            )

    def test_custom_date_format(self):
        df = DataFrame({"custom_dates": ["01-01-2021", "02-01-2021", None]})
        result = prepare_date_cols(
            df,
            cols="custom_dates",
            nan_placeholder="2000-01-01",
            date_format="%d-%m-%Y",
        )
        self.assertEqual(result["custom_dates"].iloc[0], pd.Timestamp("2021-01-01"))
        self.assertEqual(result["custom_dates"].iloc[2], pd.Timestamp("2000-01-01"))

    def test_invalid_cols_type(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_date_cols(self.df.copy(), cols=123, nan_placeholder="2000-01-01")

    def test_nonexistent_column(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_cols(
                self.df.copy(), cols="nonexistent", nan_placeholder="2000-01-01"
            )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError):
            prepare_date_cols(empty_df, cols="any_column", nan_placeholder="2000-01-01")

    def test_missing_placeholder(self):
        df = DataFrame({"dates": ["2021-01-01", None]})
        result = prepare_date_cols(df, cols="dates", nan_placeholder=None)
        self.assertTrue(pd.isna(result["dates"].iloc[1]))

    def test_preserves_other_columns(self):
        result = prepare_date_cols(
            self.df.copy(), cols="valid_dates", nan_placeholder="2000-01-01"
        )
        assert_frame_equal(result[["invalid_dates"]], self.df[["invalid_dates"]])

    def test_large_dataframe(self):
        large_df = DataFrame({"dates": ["2021-01-01"] * 100000 + [None] * 100000})
        result = prepare_date_cols(large_df, cols="dates", nan_placeholder="2000-01-01")
        self.assertEqual(result["dates"].iloc[-1], pd.Timestamp("2000-01-01"))


class TestPrepareBoolCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "col1": [1, 0, None],
                "col2": [True, False, None],
                "col3": ["yes", "no", None],
            }
        )

    def test_single_column_conversion(self):
        result = prepare_bool_cols(self.df.copy(), cols="col1")
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue((result["col1"] == [True, False, False]).all())

    def test_multiple_column_conversion(self):
        result = prepare_bool_cols(self.df.copy(), cols=["col1", "col2"])
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue(result["col2"].dtype == bool)
        self.assertTrue((result["col1"] == [True, False, False]).all())
        self.assertTrue((result["col2"] == [True, False, False]).all())

    def test_with_nan_placeholder_true(self):
        result = prepare_bool_cols(self.df.copy(), cols=["col1"], nan_placeholder=True)
        self.assertTrue((result["col1"] == [True, False, True]).all())

    def test_with_nan_placeholder_false(self):
        result = prepare_bool_cols(self.df.copy(), cols=["col1"], nan_placeholder=False)
        self.assertTrue((result["col1"] == [True, False, False]).all())

    def test_preserves_other_columns(self):
        result = prepare_bool_cols(self.df.copy(), cols=["col1"])
        assert_frame_equal(result[["col2", "col3"]], self.df[["col2", "col3"]])

    def test_invalid_cols_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_bool_cols(self.df.copy(), cols=123)
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings.",
            str(context.exception),
        )

    def test_invalid_cols_elements(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_bool_cols(self.df.copy(), cols=["col1", 123])
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings.",
            str(context.exception),
        )

    def test_nonexistent_columns(self):
        with self.assertRaises(ArgumentValueError) as context:
            prepare_bool_cols(self.df.copy(), cols=["nonexistent"])
        self.assertIn(
            "Columns ['nonexistent'] not found in DataFrame.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError) as context:
            prepare_bool_cols(empty_df, cols="col1")
        self.assertIn(
            "Columns ['col1'] not found in DataFrame.", str(context.exception)
        )

    def test_empty_cols_list(self):
        result = prepare_bool_cols(self.df.copy(), cols=[])
        assert_frame_equal(result, self.df)

    def test_large_dataframe(self):
        large_df = DataFrame({"col1": [1, 0, None] * 100000})
        result = prepare_bool_cols(large_df, cols="col1", nan_placeholder=False)
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue((result["col1"].iloc[2] == False))

    def test_mixed_data_types(self):
        df = DataFrame(
            {
                "col1": [1, "yes", None],
                "col2": [0, "no", "yes"],
            }
        )
        result = prepare_bool_cols(
            df.copy(), cols=["col1", "col2"], nan_placeholder=True
        )
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue(result["col2"].dtype == bool)
        self.assertTrue((result["col1"] == [True, True, True]).all())
        self.assertTrue((result["col2"] == [False, True, True]).all())

    def test_preserves_column_order(self):
        result = prepare_bool_cols(self.df.copy(), cols=["col1"])
        self.assertTrue(list(result.columns) == ["col1", "col2", "col3"])

    def test_no_changes_for_all_boolean_columns(self):
        df = DataFrame({"bool_col": [True, False, True]})
        result = prepare_bool_cols(df.copy(), cols="bool_col")
        self.assertTrue((result["bool_col"] == df["bool_col"]).all())


class TestReshapeGroupData(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Y", "X"],
                "value": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-02",
                    "2021-02",
                    "2021-03",
                ],
            }
        )

    def test_valid_input(self):
        result = reshape_group_data(
            dataframe=self.df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {"X": [10.0, None, 60.0], "Y": [20.0, None, None], "Z": [30.0, None, None]},
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["A", "X", 15, "2021-01"]  # Add duplicate to test aggregation
        result = reshape_group_data(
            dataframe=df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {"X": [12.5, None, 60.0], "Y": [20.0, None, None], "Z": [30.0, None, None]},
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["subgroup"])
        with self.assertRaises(ArgumentValueError) as context:
            reshape_group_data(
                dataframe=df,
                filter_value="A",
                value_column="value",
                group_column="group",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "Columns {'subgroup'} not found in the DataFrame.", str(context.exception)
        )

    def test_no_matching_filter_value(self):
        with self.assertRaises(ArgumentValueError) as context:
            reshape_group_data(
                dataframe=self.df,
                filter_value="C",
                value_column="value",
                group_column="group",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "No data found for group 'C' in column 'group'.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            reshape_group_data(
                dataframe=empty_df,
                filter_value="A",
                value_column="value",
                group_column="group",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "No data found for group 'A' in column 'group'.", str(context.exception)
        )

    def test_preserves_column_order(self):
        result = reshape_group_data(
            dataframe=self.df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertEqual(list(result.columns), ["X", "Y", "Z"])

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = reshape_group_data(
            dataframe=large_df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result is non-empty

    def test_single_subgroup(self):
        df = self.df[self.df["subgroup"] == "X"]
        result = reshape_group_data(
            dataframe=df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {"X": [10.0, None, 60.0]}, index=["2021-01", "2021-02", "2021-03"]
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected, check_dtype=False)


class TestReshapeGroupsSubgroups(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Y", "X"],
                "value": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-02",
                    "2021-02",
                    "2021-03",
                ],
            }
        )

    def test_valid_input(self):
        result = reshape_groups_subgroups(
            dataframe=self.df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("A", "X"): [10.0, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
                ("B", "X"): [None, 40.0, None],
                ("B", "Y"): [None, 50.0, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["A", "X", 15, "2021-01"]  # Add duplicate to test aggregation
        result = reshape_groups_subgroups(
            dataframe=df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                ("A", "X"): [12.5, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
                ("B", "X"): [None, 40.0, None],
                ("B", "Y"): [None, 50.0, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["subgroup"])
        with self.assertRaises(ArgumentValueError) as context:
            reshape_groups_subgroups(
                dataframe=df,
                group_column="group",
                value_column="value",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "Columns {'subgroup'} not found in the DataFrame.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError):
            reshape_groups_subgroups(
                dataframe=empty_df,
                group_column="group",
                value_column="value",
                subgroup_column="subgroup",
                time_column="time",
            )

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = reshape_groups_subgroups(
            dataframe=large_df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.shape[1] > 0)  # Ensure columns exist
        self.assertTrue(result.shape[0] > 0)  # Ensure rows exist

    def test_single_group(self):
        single_group_df = DataFrame(
            {
                "group": ["A", "A", "A", "A", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Z"],
                "value": [10, 20, 30, 60, None],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-03",
                    "2021-02",
                ],
            }
        )
        result = reshape_groups_subgroups(
            dataframe=single_group_df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("A", "X"): [10.0, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.index.name = "A"
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_column_type_mismatch(self):
        df = self.df.copy()
        df["time"] = pd.to_datetime(df["time"])  # Convert time to datetime
        result = reshape_groups_subgroups(
            dataframe=df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.index.dtype == "datetime64[ns]")

    def test_preserves_column_order(self):
        result = reshape_groups_subgroups(
            dataframe=self.df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertEqual(
            result.columns.get_level_values(1).tolist(), ["X", "Y", "Z", "X", "Y"]
        )


class TestGetEntityData(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "country": ["USA", "USA", "USA", "CAN", "CAN", "USA"],
                "indicator1": [100, 200, 300, 400, 500, 600],
                "indicator2": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q3",
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q4",
                ],
            }
        )

    def test_valid_input(self):
        result = get_entity_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        expected = DataFrame(
            {
                "indicator1": [100, 200, 300, 600],
                "indicator2": [10, 20, 30, 60],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "USA"
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["USA", 150, 15, "2021-Q1"]  # Add duplicate for aggregation
        result = get_entity_data(
            dataframe=df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                "indicator1": [125.0, 200.0, 300.0, 600.0],
                "indicator2": [12.5, 20.0, 30.0, 60.0],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "USA"
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["indicator1"])
        with self.assertRaises(ArgumentValueError) as context:
            get_entity_data(
                dataframe=df,
                data_columns=["indicator1", "indicator2"],
                entity="USA",
                entity_column="country",
                time_column="time",
            )
        self.assertIn(
            "Columns {'indicator1'} not found in the DataFrame.", str(context.exception)
        )

    def test_no_matching_entity(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_entity_data(
                dataframe=self.df,
                data_columns=["indicator1", "indicator2"],
                entity="MEX",
                entity_column="country",
                time_column="time",
            )
        self.assertIn(
            "No data found for entity 'MEX' in column 'country'.",
            str(context.exception),
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            get_entity_data(
                dataframe=empty_df,
                data_columns=["indicator1", "indicator2"],
                entity="USA",
                entity_column="country",
                time_column="time",
            )
        self.assertIn("No data found for entity 'USA'", str(context.exception))

    def test_column_type_mismatch(self):
        df = self.df.copy()
        df["time"] = pd.to_datetime(df["time"])  # Convert time to datetime
        result = get_entity_data(
            dataframe=df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        self.assertTrue(result.index.dtype == "datetime64[ns]")

    def test_reindexing_with_missing_times(self):
        result = get_entity_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity="CAN",
            entity_column="country",
            time_column="time",
        )
        expected = DataFrame(
            {
                "indicator1": [400, 500, None, None],
                "indicator2": [40, 50, None, None],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "CAN"
        assert_frame_equal(result, expected)

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = get_entity_data(
            dataframe=large_df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result has rows
        self.assertTrue(result.shape[1] == 2)  # Ensure the result has two columns

    def test_sorts_column_order(self):
        result = get_entity_data(
            dataframe=self.df,
            data_columns=["indicator2", "indicator1"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        self.assertEqual(list(result.columns), ["indicator1", "indicator2"])


class TestGetEntitiesData(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "country": ["USA", "USA", "USA", "CAN", "CAN", "USA"],
                "indicator1": [100, 200, 300, 400, 500, 600],
                "indicator2": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q3",
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q4",
                ],
            }
        )

    def test_valid_input(self):
        result = get_entities_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("CAN", "indicator1"): [400, 500, None, None],
                ("CAN", "indicator2"): [40, 50, None, None],
                ("USA", "indicator1"): [100, 200, 300, 600],
                ("USA", "indicator2"): [10, 20, 30, 60],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "time"
        expected.columns.names = ["country", "indicator"]
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["USA", 150, 15, "2021-Q1"]  # Add duplicate for aggregation
        result = get_entities_data(
            dataframe=df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                ("CAN", "indicator1"): [400.0, 500.0, None, None],
                ("CAN", "indicator2"): [40.0, 50.0, None, None],
                ("USA", "indicator1"): [125.0, 200.0, 300.0, 600.0],
                ("USA", "indicator2"): [12.5, 20.0, 30.0, 60.0],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "time"
        expected.columns.names = ["country", "indicator"]
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["indicator1"])
        with self.assertRaises(ArgumentValueError) as context:
            get_entities_data(
                dataframe=df,
                data_columns=["indicator1", "indicator2"],
                entity_column="country",
                time_column="time",
            )
        self.assertIn(
            "Columns {'indicator1'} not found in the DataFrame.", str(context.exception)
        )

    def test_specific_entities(self):
        result = get_entities_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
            entities=["USA"],
        )
        expected = DataFrame(
            {
                ("USA", "indicator1"): [100, 200, 300, 600],
                ("USA", "indicator2"): [10, 20, 30, 60],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "time"
        expected.columns.names = ["country", "indicator"]
        assert_frame_equal(result, expected)

    def test_no_matching_entities(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_entities_data(
                dataframe=self.df,
                data_columns=["indicator1", "indicator2"],
                entity_column="country",
                time_column="time",
                entities=["MEX"],
            )
        self.assertIn("Error processing entity 'MEX'", str(context.exception))

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            get_entities_data(
                dataframe=empty_df,
                data_columns=["indicator1", "indicator2"],
                entity_column="country",
                time_column="time",
            )

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = get_entities_data(
            dataframe=large_df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result has rows
        self.assertTrue(result.shape[1] == 4)  # Ensure the result has four columns

    def test_sorts_column_order(self):
        result = get_entities_data(
            dataframe=self.df,
            data_columns=["indicator2", "indicator1"],
            entity_column="country",
            time_column="time",
        )
        self.assertEqual(
            list(result.columns.get_level_values(1)), ["indicator1", "indicator2"] * 2
        )


class TestWideToLongDataFrame(TestCase):
    def setUp(self):
        self.data = DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3, 2],
                "category": ["A", "B", "A", "B", "A", "B", "C"],
                "category2": ["A", "D", "A", "B", "A", "B", "C"],
                "year": [2020, 2020, 2021, 2021, 2022, 2022, 2021],
                "value": [10, 20, 30, 40, 50, 60, 5],
                "value2": [100, 200, 300, 400, 500, 600, 30],
            }
        )

    def test_basic_transformation(self):
        result = wide_to_long_dataframe(
            dataframe=self.data, index="id", columns="category", values="value"
        )
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "A": [10, 30, 50],
                "B": [20, 40, 60],
                "C": [None, 5, None],
            }
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_multiple_transformation(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index=["id", "year"],
            columns=["category", "category2"],
            values=["value", "value2"],
            filter_columns={"category": ["A"], "category2": ["A"]},
        )
        self.assertTrue(result.columns.names == [None, "category", "category2"])
        self.assertTrue(result.index.names == ["id", "year"])
        self.assertTrue(result.shape == (3, 2))

    def test_multi_column_transformation(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index=["id", "year"],
            columns="category",
            values="value",
        )
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "year": [2020, 2021, 2022],
                "A": [10, 30, 50],
                "B": [20, 40, 60],
                "C": [None, 5, None],
            }
        ).set_index(["id", "year"])
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_filter_index(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            filter_index={"id": [1, 2]},
        )
        expected = DataFrame(
            {"id": [1, 2], "A": [10, 30], "B": [20, 40], "C": [None, 5]}
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_filter_columns(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            filter_columns={"category": ["A"]},
        )
        expected = DataFrame({"id": [1, 2, 3], "A": [10, 30, 50]}).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected)

    def test_aggfunc_sum(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            agg_func="sum",
        )
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "A": [10, 30, 50],
                "B": [20, 40, 60],
                "C": [None, 5, None],
            }
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_fill_value(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            fill_value=0,
        )
        from IPython.display import display

        display(result)
        expected = DataFrame(
            {"id": [1, 2, 3], "A": [10, 30, 50], "B": [20, 40, 60], "C": [0, 5, 0]}
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected)

    def test_missing_columns_error(self):
        with self.assertRaises(ArgumentValueError):
            wide_to_long_dataframe(
                dataframe=self.data,
                index="nonexistent",
                columns="category",
                values="value",
            )

    def test_invalid_filter_index_error(self):
        with self.assertRaises(ArgumentValueError):
            wide_to_long_dataframe(
                dataframe=self.data,
                index="id",
                columns="category",
                values="value",
                filter_index={"nonexistent": [1]},
            )

    def test_invalid_filter_columns_error(self):
        with self.assertRaises(ArgumentValueError):
            wide_to_long_dataframe(
                dataframe=self.data,
                index="id",
                columns="category",
                values="value",
                filter_columns={"nonexistent": ["A"]},
            )


class TestLongToWideDataFrame(TestCase):
    def setUp(self):
        self.data = DataFrame(
            {
                "id": [1, 2, 3],
                "2020_A": [10, 20, 30],
                "2020_B": [40, 50, 60],
                "2021_A": [70, 80, 90],
                "2021_B": [100, 110, 120],
            }
        )

    def test_basic_transformation(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            var_name="year_category",
            value_name="value",
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "year_category": ["2020_A", "2020_B", "2021_A", "2021_B"] * 3,
                    "value": [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
                }
            )
            .sort_values(by=["year_category", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_default_parameters(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "variable": ["2020_A", "2020_B", "2021_A", "2021_B"] * 3,
                    "value": [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_no_value_vars(self):
        data = DataFrame(
            {"id": [1, 2, 3], "2020_A": [10, 20, 30], "2020_B": [40, 50, 60]}
        )
        result = long_to_wide_dataframe(dataframe=data, id_vars="id")
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 2, 2, 3, 3],
                    "variable": ["2020_A", "2020_B"] * 3,
                    "value": [10, 40, 20, 50, 30, 60],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_filter_id_vars(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            filter_id_vars={"id": [1]},
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1],
                    "variable": ["2020_A", "2020_B", "2021_A", "2021_B"],
                    "value": [10, 40, 70, 100],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_filter_value_vars(self):
        result = long_to_wide_dataframe(
            dataframe=self.data, id_vars="id", value_vars=["2020_A", "2021_A"]
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 2, 2, 3, 3],
                    "variable": ["2020_A", "2021_A"] * 3,
                    "value": [10, 70, 20, 80, 30, 90],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_missing_columns(self):
        with self.assertRaises(ArgumentValueError):
            long_to_wide_dataframe(
                dataframe=self.data,
                id_vars="id",
                value_vars=["2020_A", "missing_column"],
            )

    def test_empty_dataframe(self):
        empty_data = DataFrame(columns=["id", "2020_A", "2020_B"])
        expected = DataFrame(columns=["id", "variable", "value"])
        result = long_to_wide_dataframe(dataframe=empty_data, id_vars="id")
        assert_frame_equal(result, expected)

    def test_custom_var_value_names(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            var_name="attribute",
            value_name="measurement",
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "attribute": ["2020_A", "2020_B", "2021_A", "2021_B"] * 3,
                    "measurement": [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
                }
            )
            .sort_values(by=["attribute", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_invalid_id_vars(self):
        with self.assertRaises(ArgumentValueError):
            long_to_wide_dataframe(dataframe=self.data, id_vars="missing_id_var")

    def test_filter_with_scalar(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            filter_id_vars={"id": 1},
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1],
                    "variable": ["2020_A", "2020_B", "2021_A", "2021_B"],
                    "value": [10, 40, 70, 100],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)


class TestParquetStorage(TestCase):
    def setUp(self):
        self.test_dir = Path("test_parquet_storage")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_store_and_load_simple_dataframe(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        store_dataframe_parquet(df, self.test_dir, "simple_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "simple_df")
        assert_frame_equal(df, loaded_df)

    def test_store_and_load_multiindex_dataframe(self):
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["group", "number"]
        )
        columns = pd.MultiIndex.from_tuples(
            [("X", "x"), ("X", "y"), ("Y", "z")], names=["level1", "level2"]
        )
        df = DataFrame(np.random.rand(4, 3), index=index, columns=columns)
        store_dataframe_parquet(df, self.test_dir, "multiindex_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "multiindex_df")
        assert_frame_equal(df, loaded_df)

    def test_store_and_load_with_default_index(self):
        df = DataFrame(np.random.rand(5, 5), columns=list("ABCDE"))
        store_dataframe_parquet(df, self.test_dir, "default_index_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "default_index_df")
        assert_frame_equal(df, loaded_df)

    def test_store_and_load_with_non_default_index(self):
        df = DataFrame({"A": [10, 20, 30], "B": [40, 50, 60]}, index=[100, 200, 300])
        store_dataframe_parquet(df, self.test_dir, "custom_index_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "custom_index_df")
        assert_frame_equal(df, loaded_df)

    def test_overwrite_protection(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        store_dataframe_parquet(df, self.test_dir, "protected_df", overwrite=True)
        with self.assertRaises(ArgumentValueError):
            store_dataframe_parquet(df, self.test_dir, "protected_df", overwrite=False)

    def test_missing_directory(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        with self.assertRaises(ArgumentValueError):
            store_dataframe_parquet(df, "non_existent_dir", "missing_dir_df")

    def test_missing_parquet_file(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        with self.assertRaises(ArgumentValueError):
            load_dataframe_parquet(df, self.test_dir, "nonexistent_df")

    def test_missing_index_file(self):
        df = DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=pd.MultiIndex.from_tuples(
                [("A", 1), ("B", 2), ("C", 3)], names=["letter", "number"]
            ),
        )
        store_dataframe_parquet(df, self.test_dir, "missing_index_test", overwrite=True)
        (
            self.test_dir / "missing_index_test_index.parquet"
        ).unlink()  # Remove the index file
        loaded_df = load_dataframe_parquet(df, self.test_dir, "missing_index_test")
        self.assertTrue(
            loaded_df.index.equals(pd.RangeIndex(start=0, stop=3))
        )  # Default index applied

    def test_missing_columns_file(self):
        df = DataFrame(
            np.random.rand(3, 3),
            columns=pd.MultiIndex.from_tuples(
                [("X", "x"), ("Y", "y"), ("Z", "z")], names=["level1", "level2"]
            ),
        )
        store_dataframe_parquet(
            df, self.test_dir, "missing_columns_test", overwrite=True
        )
        (
            self.test_dir / "missing_columns_test_columns.parquet"
        ).unlink()  # Remove the columns file
        loaded_df = load_dataframe_parquet(df, self.test_dir, "missing_columns_test")
        self.assertTrue(
            loaded_df.columns.equals(pd.RangeIndex(start=0, stop=3))
        )  # Default columns applied

    def test_store_and_load_empty_dataframe(self):
        df = DataFrame()
        store_dataframe_parquet(df, self.test_dir, "empty_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "empty_df")
        assert_frame_equal(df, loaded_df)


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
