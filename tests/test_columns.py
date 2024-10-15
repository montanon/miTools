import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, IndexSlice, MultiIndex, testing
from pandas.api.types import is_numeric_dtype

# Assuming the provided function is imported or defined here
from mitools.economic_complexity.columns import (
    GROWTH_COLUMN_NAME,
    GROWTH_PCT_COLUMN_NAME,
    add_columns,
    divide_columns,
    multiply_columns,
    select_columns,
    shift_columns,
    transform_columns,
    variation_columns,
)
from mitools.exceptions.custom_exceptions import ArgumentKeyError, ArgumentTypeError


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
        with self.assertRaises(ValueError) as context:
            select_columns(self.df_single, "D")
        self.assertIn("Invalid columns", str(context.exception))

    def test_mix_columns_single_level(self):
        with self.assertRaises(ValueError) as context:
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
        with self.assertRaises(ValueError) as context:
            select_columns(self.df_multi, ["X"], level="invalid")
        self.assertIn("Invalid level name", str(context.exception))

    def test_invalid_level_index(self):
        with self.assertRaises(ValueError) as context:
            select_columns(self.df_multi, ["X"], level=2)
        self.assertIn("Invalid level index", str(context.exception))

    def test_tuple_column_mismatch(self):
        with self.assertRaises(ValueError) as context:
            select_columns(self.df_multi, [("A", "X", "extra")])
        self.assertIn("Invalid columns", str(context.exception))

    def test_level_in_single_level_dataframe(self):
        with self.assertRaises(ValueError) as context:
            select_columns(self.df_single, ["A"], level=0)
        self.assertIn("level can only be specified", str(context.exception))

    def test_empty_dataframe_single_level(self):
        empty_df = pd.DataFrame(columns=["A", "B", "C"])
        result = select_columns(empty_df, ["A", "B"])
        self.assertEqual(result.shape, (0, 2))

    def test_empty_dataframe_multiindex(self):
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        empty_multi_df = pd.DataFrame(columns=columns)
        result = select_columns(empty_multi_df, [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (0, 2))

    def test_invalid_column_type(self):
        with self.assertRaises(TypeError) as context:
            select_columns(self.df_single, {})
        self.assertIn(
            "Provided 'columns' must be a string, tuple, int, or list.",
            str(context.exception),
        )


class TestTransformColumns(unittest.TestCase):
    def setUp(self):
        # Setup a DataFrame with MultiIndex columns for testing
        self.multiidx_df = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)
        self.singleidx_df = DataFrame(
            {
                ("one"): [1, 2, 3],
                ("two"): [4, 5, 6],
                ("three"): [7, 8, 9],
            }
        )

    def test_with_multiidx(self):
        # Testing with a log transformation
        from numpy import log

        result = transform_columns(self.multiidx_df, ["one", "three"], log)
        # Check if the transformation was applied correctly
        for col in result.columns:
            if col[1].endswith("_log"):
                self.assertTrue(is_numeric_dtype(result[col]))
                # Check if the transformed columns have correct values
                original_col = (col[0], col[1].replace("_log", ""))
                self.assertTrue(
                    all(
                        result[col]
                        == log(self.multiidx_df[original_col].replace(0, 1e-6))
                    )
                )

    def test_with_singleidx(self):
        # Testing with a log transformation
        from numpy import log

        result = transform_columns(self.singleidx_df, ["one", "three"], log)
        # Check if the transformation was applied correctly
        for col in result.columns:
            if col.endswith("_log"):
                self.assertTrue(is_numeric_dtype(result[col]))
                # Check if the transformed columns have correct values
                original_col = col.replace("_log", "")
                self.assertTrue(
                    all(
                        result[col]
                        == log(self.singleidx_df[original_col].replace(0, 1e-6))
                    )
                )

    def test_transform_with_invalid_column(self):
        # Testing with a column that doesn't exist
        from numpy import log

        with self.assertRaises(ArgumentKeyError):
            transform_columns(self.multiidx_df, ["four"], log)

    def test_transform_with_invalid_function(self):
        # Testing with an invalid transformation
        def invalid_function(x):
            return x + "invalid"

        with self.assertRaises(ArgumentTypeError):
            transform_columns(self.multiidx_df, ["one"], invalid_function)


class TestVariationColumns(unittest.TestCase):
    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        index = Index(range(1995, 2021), name="Year")
        iterables = [["CountryA", "CountryB"], ["Indicator1", "Indicator2"]]
        columns = MultiIndex.from_product(iterables, names=["Country", "Indicator"])
        self.data = DataFrame(
            [[float(i + j) for j in range(4)] for i in range(26)],
            index=index,
            columns=columns,
        )

    def test_absolute_variation(self):
        # Test the function with absolute variation
        t = 1
        result = variation_columns(self.data, ["Indicator1"], t, pct=False)
        expected_change_name = GROWTH_COLUMN_NAME.format(t)
        self.assertTrue(
            "Indicator1" + expected_change_name in result.columns.get_level_values(-1)
        )
        # Test the correctness of calculations
        old_value = ("CountryA", "Indicator1")
        new_value = ("CountryA", f"Indicator1{expected_change_name}")
        original_values = self.data.loc[:, [old_value]]
        shifted_values = original_values.shift(t)
        expected_values = original_values - shifted_values
        new_tuples = [
            (new_value if old == old_value else old) for old in expected_values.columns
        ]
        expected_values.columns = MultiIndex.from_tuples(
            new_tuples, names=expected_values.columns.names
        )
        testing.assert_frame_equal(result[[new_value]], expected_values)

    def test_percentage_variation(self):
        # Test the function with percentage variation
        t = 1
        result = variation_columns(self.data, ["Indicator1"], t, pct=True).copy(
            deep=True
        )
        expected_change_name = GROWTH_PCT_COLUMN_NAME.format(t)
        self.assertTrue(
            f"Indicator1{expected_change_name}" in result.columns.get_level_values(1)
        )
        # Test the correctness of calculations
        old_values = [("CountryA", "Indicator1"), ("CountryB", "Indicator1")]
        new_values = [
            ("CountryA", f"Indicator1{expected_change_name}"),
            ("CountryB", f"Indicator1{expected_change_name}"),
        ]
        original_values = self.data.loc[:, IndexSlice[:, "Indicator1"]]
        shifted_values = original_values.shift(t)
        variation_cols = original_values - shifted_values
        expected_values = (variation_cols / original_values) * 100.0
        expected_values.columns = MultiIndex.from_tuples(
            new_values, names=expected_values.columns.names
        )
        testing.assert_frame_equal(result, expected_values)


class TestShiftColumns(unittest.TestCase):
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

    def test_positive_shift(self):
        t = 1
        columns_to_shift = ["Indicator1"]
        shifted_df = shift_columns(self.dataframe, columns_to_shift, t)
        # Test if the columns are shifted down by t
        for col in shifted_df.columns:
            if col[1].startswith("Indicator1_shifted_by"):
                self.assertTrue(
                    shifted_df[col].equals(
                        self.dataframe[col[0], "Indicator1"].shift(-t)
                    )
                )

    def test_negative_shift(self):
        t = -1
        columns_to_shift = ["Indicator2"]
        shifted_df = shift_columns(self.dataframe, columns_to_shift, t)
        # Test if the columns are shifted up by t
        for col in shifted_df.columns:
            if col[1].startswith("Indicator2_shifted_by"):
                self.assertTrue(
                    shifted_df[col].equals(
                        self.dataframe[col[0], "Indicator2"].shift(-t)
                    )
                )

    def test_shift_with_nonexistent_column(self):
        t = 1
        columns_to_shift = ["NonexistentColumn"]
        with self.assertRaises(KeyError):
            shift_columns(self.dataframe, columns_to_shift, t)

    def test_zero_shift(self):
        t = 0
        columns_to_shift = ["Indicator1"]
        shifted_df = shift_columns(self.dataframe, columns_to_shift, t)
        # Test if the columns are not shifted (remain the same)
        for col in shifted_df.columns:
            if col[1].startswith("Indicator1_shifted_by"):
                self.assertTrue(
                    shifted_df[col].equals(self.dataframe[col[0], "Indicator1"])
                )


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
