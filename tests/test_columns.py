import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from numpy import log
from pandas import DataFrame, Index, IndexSlice, MultiIndex
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal

# Assuming the provided function is imported or defined here
from mitools.economic_complexity.columns import (
    GROWTH_COLUMN_NAME,
    GROWTH_PCT_COLUMN_NAME,
    add_columns,
    divide_columns,
    growth_columns,
    multiply_columns,
    select_columns,
    select_index,
    shift_columns,
    transform_columns,
)
from mitools.exceptions.custom_exceptions import (
    ArgumentKeyError,
    ArgumentTypeError,
    ArgumentValueError,
)


class TestSelectIndex(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}).T
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        self.df_multi = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=columns, index=columns
        ).T

    def test_single_column_single_level(self):
        result = select_index(self.df_single, "A")
        self.assertListEqual(list(result.index), ["A"])
        self.assertEqual(result.shape, (1, 3))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_multiple_columns_single_level(self):
        result = select_index(self.df_single, ["A", "C"])
        self.assertListEqual(list(result.index), ["A", "C"])
        self.assertEqual(result.shape, (2, 3))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_invalid_column_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_single, "D")
        self.assertIn("Invalid index", str(context.exception))

    def test_mix_columns_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_single, ["A", "D"])
        self.assertIn("Invalid index", str(context.exception))

    def test_empty_column_selection_single_level(self):
        result = select_index(self.df_single, [])
        self.assertEqual(result.shape, (0, 3))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_select_all_columns_single_level(self):
        result = select_index(self.df_single, list(self.df_single.index))
        self.assertListEqual(list(result.index), list(self.df_single.index))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_multiindex_column_selection(self):
        result = select_index(self.df_multi, [("A", "X"), ("B", "Z")])
        self.assertListEqual(list(result.index), [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (2, 3))

    def test_multiindex_with_level_positional(self):
        result = select_index(self.df_multi, ["X", "Y"], level=1)
        self.assertListEqual(list(result.index), [("A", "X"), ("A", "Y")])

    def test_multiindex_with_level_name(self):
        result = select_index(self.df_multi, ["X", "Y"], level="lower")
        self.assertListEqual(list(result.index), [("A", "X"), ("A", "Y")])

    def test_invalid_level_name(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_multi, ["X"], level="invalid")
        self.assertIn("Invalid level name", str(context.exception))

    def test_invalid_level_index(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_multi, ["X"], level=2)
        self.assertIn("Invalid level index", str(context.exception))

    def test_tuple_column_mismatch(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_multi, [("A", "X", "extra")])
        self.assertIn("Invalid index", str(context.exception))

    def test_level_in_single_level_dataframe(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_single, ["A"], level=0)
        self.assertIn("level can only be specified", str(context.exception))

    def test_empty_dataframe_single_level(self):
        empty_df = DataFrame(index=["A", "B", "C"])
        result = select_index(empty_df, ["A", "B"])
        self.assertEqual(result.shape, (2, 0))

    def test_empty_dataframe_multiindex(self):
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        index = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        empty_multi_df = DataFrame(index=index)
        result = select_index(empty_multi_df, [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (2, 0))

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            select_index(self.df_single, {})
        self.assertIn(
            "Provided 'index' must be a string, tuple, int, or list.",
            str(context.exception),
        )


class TestSelectColumns(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        self.df_multi = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=columns, index=columns
        )

    def test_single_column_single_level(self):
        result = select_columns(self.df_single, "A")
        self.assertListEqual(list(result.columns), ["A"])
        self.assertEqual(result.shape, (3, 1))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_multiple_columns_single_level(self):
        result = select_columns(self.df_single, ["A", "C"])
        self.assertListEqual(list(result.columns), ["A", "C"])
        self.assertEqual(result.shape, (3, 2))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_invalid_column_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_single, "D")
        self.assertIn("Invalid columns", str(context.exception))

    def test_mix_columns_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_single, ["A", "D"])
        self.assertIn("Invalid columns", str(context.exception))

    def test_empty_column_selection_single_level(self):
        result = select_columns(self.df_single, [])
        self.assertEqual(result.shape, (3, 0))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_select_all_columns_single_level(self):
        result = select_columns(self.df_single, list(self.df_single.columns))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_multiindex_column_selection(self):
        result = select_columns(self.df_multi, [("A", "X"), ("B", "Z")])
        self.assertListEqual(list(result.columns), [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (3, 2))

    def test_multiindex_with_level_positional(self):
        result = select_columns(self.df_multi, ["X", "Y"], level=1)
        self.assertListEqual(list(result.columns), [("A", "X"), ("A", "Y")])

    def test_multiindex_with_level_name(self):
        result = select_columns(self.df_multi, ["X", "Y"], level="lower")
        self.assertListEqual(list(result.columns), [("A", "X"), ("A", "Y")])

    def test_invalid_level_name(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_multi, ["X"], level="invalid")
        self.assertIn("Invalid level name", str(context.exception))

    def test_invalid_level_index(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_multi, ["X"], level=2)
        self.assertIn("Invalid level index", str(context.exception))

    def test_tuple_column_mismatch(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_multi, [("A", "X", "extra")])
        self.assertIn("Invalid columns", str(context.exception))

    def test_level_in_single_level_dataframe(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_single, ["A"], level=0)
        self.assertIn("level can only be specified", str(context.exception))

    def test_empty_dataframe_single_level(self):
        empty_df = DataFrame(columns=["A", "B", "C"])
        result = select_columns(empty_df, ["A", "B"])
        self.assertEqual(result.shape, (0, 2))

    def test_empty_dataframe_multiindex(self):
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        empty_multi_df = DataFrame(columns=columns)
        result = select_columns(empty_multi_df, [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (0, 2))

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            select_columns(self.df_single, {})
        self.assertIn(
            "Provided 'columns' must be a string, tuple, int, or list.",
            str(context.exception),
        )


class TestTransformColumns(TestCase):
    def setUp(self):
        # Setup a DataFrame with MultiIndex columns for testing
        self.multiidx_df = pd.DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)

        # Single-index DataFrame
        self.singleidx_df = pd.DataFrame(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7, 8, 9],
            }
        )

    def test_transform_multiidx_log(self):
        result = transform_columns(self.multiidx_df, log, ["one", "three"], level=-1)
        expected_columns = [("A", "one_log"), ("B", "three_log")]
        self.assertListEqual(list(result.columns), expected_columns)
        for col in result.columns:
            original_col = (col[0], col[1].replace("_log", ""))
            self.assertTrue(is_numeric_dtype(result[col]))
            self.assertTrue(
                all(result[col] == log(self.multiidx_df[original_col].replace(0, 1e-6)))
            )

    def test_transform_singleidx_log(self):
        result = transform_columns(self.singleidx_df, log, ["one", "three"])
        expected_columns = ["one_log", "three_log"]
        self.assertListEqual(list(result.columns), expected_columns)
        for col in result.columns:
            original_col = col.replace("_log", "")
            self.assertTrue(is_numeric_dtype(result[col]))
            self.assertTrue(
                all(
                    result[col] == log(self.singleidx_df[original_col].replace(0, 1e-6))
                )
            )

    def test_transform_with_custom_rename(self):
        result = transform_columns(self.singleidx_df, log, ["one"], rename="custom")
        expected_columns = ["one_custom"]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_transform_with_invalid_column(self):
        with self.assertRaises(ArgumentValueError):
            transform_columns(self.multiidx_df, log, ["four"])

    def test_transform_with_invalid_function(self):
        def invalid_function(x):
            return x + "invalid"

        with self.assertRaises(ArgumentValueError):
            transform_columns(self.multiidx_df, invalid_function, ["one"])

    def test_transform_multiidx_with_level_positional(self):
        result = transform_columns(self.multiidx_df, log, ["one", "three"], level=1)
        expected_columns = [("A", "one_log"), ("B", "three_log")]
        self.assertListEqual(list(result.columns), expected_columns)
        for col in result.columns:
            original_col = (col[0], col[1].replace("_log", ""))
            self.assertTrue(is_numeric_dtype(result[col]))
            self.assertTrue(
                all(result[col] == log(self.multiidx_df[original_col].replace(0, 1e-6)))
            )

    def test_transform_with_non_callable_transformation(self):
        with self.assertRaises(ArgumentTypeError):
            transform_columns(self.singleidx_df, "not_callable", ["one"])

    def test_transform_with_tuple_column_not_matching_multiindex(self):
        with self.assertRaises(ValueError):
            transform_columns(self.multiidx_df, log, [("A", "invalid")])


class TestVariationColumns(TestCase):
    def setUp(self):
        self.multiidx_df = pd.DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)
        self.singleidx_df = pd.DataFrame(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7, 8, 9],
            }
        )

    def test_variation_singleidx_absolute(self):
        result = growth_columns(self.singleidx_df, ["one", "three"], t=1)
        expected_columns = ["one_growth_1", "three_growth_1"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {"one_growth_1": [None, 1, 1], "three_growth_1": [None, 1, 1]}
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_singleidx_pct(self):
        result = growth_columns(self.singleidx_df, ["one", "three"], t=1, pct=True)
        expected_columns = ["one_growth%_1", "three_growth%_1"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {
                "one_growth%_1": [None, 50.0, 33.333333],
                "three_growth%_1": [None, 12.5, 11.111111],  # Rounded for clarity
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_multiidx_absolute(self):
        result = growth_columns(self.multiidx_df, ["one", "three"], t=1, level=-1)
        expected_columns = [("A", "one_growth_1"), ("B", "three_growth_1")]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {
                ("A", "one_growth_1"): [None, 1, 1],
                ("B", "three_growth_1"): [None, 1, 1],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_multiidx_pct(self):
        result = growth_columns(
            self.multiidx_df, ["one", "three"], t=1, pct=True, level=-1
        )
        expected_columns = [
            ("A", "one_growth%_1"),
            ("B", "three_growth%_1"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {
                ("A", "one_growth%_1"): [None, 50.0, 33.333333],
                ("B", "three_growth%_1"): [
                    None,
                    12.5,
                    11.111111,
                ],  # Rounded for clarity
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_with_invalid_t(self):
        with self.assertRaises(ArgumentTypeError):
            growth_columns(self.singleidx_df, ["one"], t="invalid")

    def test_variation_with_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            growth_columns(self.singleidx_df, ["invalid_column"], t=1)

    def test_variation_with_custom_rename(self):
        result = growth_columns(self.singleidx_df, ["one"], t=1, rename="custom_name")
        expected_columns = ["one_custom_name"]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_variation_multiidx_with_positional_level(self):
        result = growth_columns(self.multiidx_df, ["one", "three"], t=1, level=1)
        expected_columns = [("A", "one_growth_1"), ("B", "three_growth_1")]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_variation_non_numeric_data(self):
        df_non_numeric = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        with self.assertRaises(ArgumentValueError):
            growth_columns(df_non_numeric, ["A"], t=1)


class TestShiftColumns(TestCase):
    def setUp(self):
        self.multiidx_df = pd.DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)
        self.singleidx_df = pd.DataFrame(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7, 8, 9],
            }
        )

    def test_shift_singleidx(self):
        result = shift_columns(self.singleidx_df, ["one", "three"], t=1)
        expected_columns = ["one_shifted_1", "three_shifted_1"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {"one_shifted_1": [None, 1, 2], "three_shifted_1": [None, 7, 8]}
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_multiidx(self):
        result = shift_columns(self.multiidx_df, ["one", "three"], t=1, level=-1)
        expected_columns = [("A", "one_shifted_1"), ("B", "three_shifted_1")]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {
                ("A", "one_shifted_1"): [None, 1, 2],
                ("B", "three_shifted_1"): [None, 7, 8],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_singleidx_with_custom_rename(self):
        result = shift_columns(self.singleidx_df, ["one"], t=1, rename="custom_name")
        expected_columns = ["one_custom_name"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame({"one_custom_name": [None, 1, 2]})
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_multiidx_with_positional_level(self):
        result = shift_columns(self.multiidx_df, ["one", "three"], t=1, level=1)
        expected_columns = [("A", "one_shifted_1"), ("B", "three_shifted_1")]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = pd.DataFrame(
            {
                ("A", "one_shifted_1"): [None, 1, 2],
                ("B", "three_shifted_1"): [None, 7, 8],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_with_invalid_t(self):
        with self.assertRaises(ArgumentTypeError):
            shift_columns(self.singleidx_df, ["one"], t="invalid")

    def test_shift_with_invalid_column(self):
        with self.assertRaises(ArgumentValueError):
            shift_columns(self.singleidx_df, ["invalid_column"], t=1)


class TestAddColumns(unittest.TestCase):
    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame(
            {
                ("Country1", "Indicator1"): [1, 2, 3, 4, 5],
                ("Country1", "Indicator2"): [6, 7, 8, 9, 10],
                ("Country2", "Indicator1"): [11, 12, 13, 14, 15],
                ("Country2", "Indicator2"): [16, 17, 18, 19, 20],
            }
        )
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_addition(self):
        new_name = "Sum_Indicator"
        added_df = add_columns(self.dataframe, "Indicator1", "Indicator2", new_name)
        # Verify the addition is correct
        for country in self.dataframe.columns.levels[0]:
            self.assertTrue(
                all(
                    added_df[(country, new_name)]
                    == self.dataframe[(country, "Indicator1")]
                    + self.dataframe[(country, "Indicator2")]
                )
            )

    def test_column_names(self):
        new_name = "Sum_Indicator"
        added_df = add_columns(self.dataframe, "Indicator1", "Indicator2", new_name)
        # Check if the new column names are correctly assigned
        expected_columns = [
            (country, new_name) for country in self.dataframe.columns.levels[0]
        ]
        self.assertEqual(added_df.columns.tolist(), expected_columns)

    def test_nonexistent_columns(self):
        # Attempt to add non-existent columns
        with self.assertRaises(KeyError):
            add_columns(
                self.dataframe, "NonexistentColumn1", "NonexistentColumn2", "Result"
            )


class TestDivideColumns(unittest.TestCase):
    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame(
            {
                ("Country1", "Indicator1"): [2, 4, 6, 8, 10],
                ("Country1", "Indicator2"): [1, 2, 3, 4, 5],
                ("Country2", "Indicator1"): [20, 40, 60, 80, 100],
                ("Country2", "Indicator2"): [10, 20, 30, 40, 50],
            }
        )
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_division(self):
        new_name = "Ratio_Indicator"
        divided_df = divide_columns(
            self.dataframe, "Indicator1", "Indicator2", new_name
        )
        # Verify the division is correct
        for country in self.dataframe.columns.levels[0]:
            expected_result = (
                self.dataframe[(country, "Indicator1")]
                / self.dataframe[(country, "Indicator2")]
            )
            self.assertTrue(
                np.allclose(divided_df[(country, new_name)], expected_result)
            )

    def test_column_names(self):
        new_name = "Ratio_Indicator"
        divided_df = divide_columns(
            self.dataframe, "Indicator1", "Indicator2", new_name
        )
        # Check if the new column names are correctly assigned
        expected_columns = [
            (country, new_name) for country in self.dataframe.columns.levels[0]
        ]
        self.assertEqual(divided_df.columns.tolist(), expected_columns)

    def test_nonexistent_columns(self):
        # Attempt to divide non-existent columns
        with self.assertRaises(KeyError):
            divide_columns(
                self.dataframe, "NonexistentColumn1", "NonexistentColumn2", "Result"
            )

    def test_division_by_zero(self):
        # Add a row with zero to test division by zero
        zero_row = pd.DataFrame({("Country1", "Indicator2"): [0]}, index=[5])
        zero_df = pd.concat([self.dataframe, zero_row])
        new_name = "Ratio_Indicator"
        divided_df = divide_columns(zero_df, "Indicator1", "Indicator2", new_name)
        # Check if division by zero results in infinity)
        self.assertTrue(np.isnan(divided_df.loc[5, ("Country1", new_name)]))


class TestMultiplyColumns(unittest.TestCase):
    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame(
            {
                ("Country1", "Indicator1"): [1, 2, 3, 4, 5],
                ("Country1", "Indicator2"): [6, 7, 8, 9, 10],
                ("Country2", "Indicator1"): [11, 12, 13, 14, 15],
                ("Country2", "Indicator2"): [16, 17, 18, 19, 20],
            }
        )
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_multiplication(self):
        new_name = "Product_Indicator"
        multiplied_df = multiply_columns(
            self.dataframe, "Indicator1", "Indicator2", new_name
        )

        # Verify the multiplication is correct
        for country in self.dataframe.columns.levels[0]:
            expected_result = (
                self.dataframe[(country, "Indicator1")]
                * self.dataframe[(country, "Indicator2")]
            )
            self.assertTrue(
                np.allclose(multiplied_df[(country, new_name)], expected_result)
            )

    def test_column_names(self):
        new_name = "Product_Indicator"
        multiplied_df = multiply_columns(
            self.dataframe, "Indicator1", "Indicator2", new_name
        )

        # Check if the new column names are correctly assigned
        expected_columns = [
            (country, new_name) for country in self.dataframe.columns.levels[0]
        ]
        self.assertEqual(multiplied_df.columns.tolist(), expected_columns)

    def test_nonexistent_columns(self):
        # Attempt to multiply non-existent columns
        with self.assertRaises(KeyError):
            multiply_columns(
                self.dataframe, "NonexistentColumn1", "NonexistentColumn2", "Result"
            )


if __name__ == "__main__":
    unittest.main()
