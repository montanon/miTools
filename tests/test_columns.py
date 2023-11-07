import unittest
from typing import Callable, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, testing
from pandas.api.types import is_numeric_dtype

# Assuming the provided function is imported or defined here
from mitools.economic_complexity.columns import *
from mitools.exceptions.custom_exceptions import (ArgumentKeyError,
                                                  ArgumentTypeError)


class TestTransformColumns(unittest.TestCase):

    def setUp(self):
        # Setup a DataFrame with MultiIndex columns for testing
        self.data = DataFrame({
            ('A', 'one'): [1, 2, 3],
            ('A', 'two'): [4, 5, 6],
            ('B', 'three'): [7, 8, 9]
        })
        self.data.columns = MultiIndex.from_tuples(self.data.columns)

    def test_transform_with_log(self):
        # Testing with a log transformation
        from numpy import log
        result = transform_columns(self.data, ['one', 'three'], log)
        # Check if the transformation was applied correctly
        for col in result.columns:
            if col[1].endswith('_log'):
                self.assertTrue(is_numeric_dtype(result[col]))
                # Check if the transformed columns have correct values
                original_col = (col[0], col[1].replace('_log', ''))
                self.assertTrue(all(result[col] == log(self.data[original_col].replace(0, 1e-6))))

    def test_transform_with_invalid_column(self):
        # Testing with a column that doesn't exist
        from numpy import log
        with self.assertRaises(ArgumentKeyError):
            transform_columns(self.data, ['four'], log)

    def test_transform_with_invalid_function(self):
        # Testing with an invalid transformation
        def invalid_function(x):
            return x + "invalid"
        with self.assertRaises(IndexError):
            transform_columns(self.data, ['one'], invalid_function)


class TestVariationColumns(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        index = Index(range(1995, 2021), name='Year')
        iterables = [['CountryA', 'CountryB'], ['Indicator1', 'Indicator2']]
        columns = MultiIndex.from_product(iterables, names=['Country', 'Indicator'])
        self.data = DataFrame(
            [[float(i + j) for j in range(4)] for i in range(26)],
            index=index,
            columns=columns
        )

    def test_absolute_variation(self):
        # Test the function with absolute variation
        t = 1
        result = variation_columns(self.data, ['Indicator1'], t, pct=False)
        expected_change_name = GROWTH_COLUMN_NAME.format(t)
        self.assertTrue('Indicator1' + expected_change_name in result.columns.get_level_values(-1))
        # Test the correctness of calculations
        old_value = ('CountryA', 'Indicator1')
        new_value = ('CountryA', f'Indicator1{expected_change_name}')
        original_values = self.data.loc[:, [old_value]]
        shifted_values = original_values.shift(t)
        expected_values = original_values - shifted_values
        new_tuples = [(new_value if old == old_value else old) for old in expected_values.columns]
        expected_values.columns = MultiIndex.from_tuples(new_tuples, names=expected_values.columns.names)
        testing.assert_frame_equal(result[[new_value]], expected_values)

    def test_percentage_variation(self):
        # Test the function with percentage variation
        t = 1
        result = variation_columns(self.data, ['Indicator1'], t, pct=True).copy(deep=True)
        expected_change_name = GROWTH_PCT_COLUMN_NAME.format(t)
        self.assertTrue(f"Indicator1{expected_change_name}" in result.columns.get_level_values(1))
        # Test the correctness of calculations
        old_values = [('CountryA', 'Indicator1'), ('CountryB', 'Indicator1')]
        new_values = [('CountryA', f'Indicator1{expected_change_name}'),
                       ('CountryB', f'Indicator1{expected_change_name}')]
        original_values = self.data.loc[:, IndexSlice[:, 'Indicator1']]
        shifted_values = original_values.shift(t)
        variation_cols = original_values - shifted_values
        expected_values = (variation_cols / original_values) * 100.0
        expected_values.columns = MultiIndex.from_tuples(new_values, names=expected_values.columns.names)
        testing.assert_frame_equal(result, expected_values)


class TestShiftColumns(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame({
            ('Country1', 'Indicator1'): [1, 2, 3, 4, 5],
            ('Country1', 'Indicator2'): [6, 7, 8, 9, 10],
            ('Country2', 'Indicator1'): [11, 12, 13, 14, 15],
            ('Country2', 'Indicator2'): [16, 17, 18, 19, 20]
        })
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_positive_shift(self):
        t = 1
        columns_to_shift = ['Indicator1']
        shifted_df = shift_columns(self.dataframe, columns_to_shift, t)
        # Test if the columns are shifted down by t
        for col in shifted_df.columns:
            if col[1].startswith('Indicator1_shifted_by'):
                self.assertTrue(shifted_df[col].equals(self.dataframe[col[0], 'Indicator1'].shift(-t)))

    def test_negative_shift(self):
        t = -1
        columns_to_shift = ['Indicator2']
        shifted_df = shift_columns(self.dataframe, columns_to_shift, t)
        # Test if the columns are shifted up by t
        for col in shifted_df.columns:
            if col[1].startswith('Indicator2_shifted_by'):
                self.assertTrue(shifted_df[col].equals(self.dataframe[col[0], 'Indicator2'].shift(-t)))

    def test_shift_with_nonexistent_column(self):
        t = 1
        columns_to_shift = ['NonexistentColumn']
        with self.assertRaises(KeyError):
            shift_columns(self.dataframe, columns_to_shift, t)

    def test_zero_shift(self):
        t = 0
        columns_to_shift = ['Indicator1']
        shifted_df = shift_columns(self.dataframe, columns_to_shift, t)
        # Test if the columns are not shifted (remain the same)
        for col in shifted_df.columns:
            if col[1].startswith('Indicator1_shifted_by'):
                self.assertTrue(shifted_df[col].equals(self.dataframe[col[0], 'Indicator1']))


class TestAddColumns(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame({
            ('Country1', 'Indicator1'): [1, 2, 3, 4, 5],
            ('Country1', 'Indicator2'): [6, 7, 8, 9, 10],
            ('Country2', 'Indicator1'): [11, 12, 13, 14, 15],
            ('Country2', 'Indicator2'): [16, 17, 18, 19, 20]
        })
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_addition(self):
        new_name = 'Sum_Indicator'
        added_df = add_columns(self.dataframe, 'Indicator1', 'Indicator2', new_name)
        # Verify the addition is correct
        for country in self.dataframe.columns.levels[0]:
            self.assertTrue(all(added_df[(country, new_name)] == 
                                self.dataframe[(country, 'Indicator1')] + self.dataframe[(country, 'Indicator2')]))

    def test_column_names(self):
        new_name = 'Sum_Indicator'
        added_df = add_columns(self.dataframe, 'Indicator1', 'Indicator2', new_name)
        # Check if the new column names are correctly assigned
        expected_columns = [(country, new_name) for country in self.dataframe.columns.levels[0]]
        self.assertEqual(added_df.columns.tolist(), expected_columns)

    def test_nonexistent_columns(self):
        # Attempt to add non-existent columns
        with self.assertRaises(KeyError):
            add_columns(self.dataframe, 'NonexistentColumn1', 'NonexistentColumn2', 'Result')


class TestDivideColumns(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame({
            ('Country1', 'Indicator1'): [2, 4, 6, 8, 10],
            ('Country1', 'Indicator2'): [1, 2, 3, 4, 5],
            ('Country2', 'Indicator1'): [20, 40, 60, 80, 100],
            ('Country2', 'Indicator2'): [10, 20, 30, 40, 50]
        })
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_division(self):
        new_name = 'Ratio_Indicator'
        divided_df = divide_columns(self.dataframe, 'Indicator1', 'Indicator2', new_name)
        # Verify the division is correct
        for country in self.dataframe.columns.levels[0]:
            expected_result = self.dataframe[(country, 'Indicator1')] / self.dataframe[(country, 'Indicator2')]
            self.assertTrue(np.allclose(divided_df[(country, new_name)], expected_result))

    def test_column_names(self):
        new_name = 'Ratio_Indicator'
        divided_df = divide_columns(self.dataframe, 'Indicator1', 'Indicator2', new_name)
        # Check if the new column names are correctly assigned
        expected_columns = [(country, new_name) for country in self.dataframe.columns.levels[0]]
        self.assertEqual(divided_df.columns.tolist(), expected_columns)

    def test_nonexistent_columns(self):
        # Attempt to divide non-existent columns
        with self.assertRaises(KeyError):
            divide_columns(self.dataframe, 'NonexistentColumn1', 'NonexistentColumn2', 'Result')

    def test_division_by_zero(self):
        # Add a row with zero to test division by zero
        zero_row = pd.DataFrame({('Country1', 'Indicator2'): [0]}, index=[5])
        zero_df = pd.concat([self.dataframe, zero_row])
        new_name = 'Ratio_Indicator'
        divided_df = divide_columns(zero_df, 'Indicator1', 'Indicator2', new_name)
        # Check if division by zero results in infinity)
        self.assertTrue(np.isnan(divided_df.loc[5, ('Country1', new_name)]))


class TestMultiplyColumns(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with MultiIndex columns for testing
        self.dataframe = DataFrame({
            ('Country1', 'Indicator1'): [1, 2, 3, 4, 5],
            ('Country1', 'Indicator2'): [6, 7, 8, 9, 10],
            ('Country2', 'Indicator1'): [11, 12, 13, 14, 15],
            ('Country2', 'Indicator2'): [16, 17, 18, 19, 20]
        })
        self.dataframe.columns = MultiIndex.from_tuples(self.dataframe.columns)

    def test_multiplication(self):
        new_name = 'Product_Indicator'
        multiplied_df = multiply_columns(self.dataframe, 'Indicator1', 'Indicator2', new_name)

        # Verify the multiplication is correct
        for country in self.dataframe.columns.levels[0]:
            expected_result = self.dataframe[(country, 'Indicator1')] * self.dataframe[(country, 'Indicator2')]
            self.assertTrue(np.allclose(multiplied_df[(country, new_name)], expected_result))

    def test_column_names(self):
        new_name = 'Product_Indicator'
        multiplied_df = multiply_columns(self.dataframe, 'Indicator1', 'Indicator2', new_name)

        # Check if the new column names are correctly assigned
        expected_columns = [(country, new_name) for country in self.dataframe.columns.levels[0]]
        self.assertEqual(multiplied_df.columns.tolist(), expected_columns)

    def test_nonexistent_columns(self):
        # Attempt to multiply non-existent columns
        with self.assertRaises(KeyError):
            multiply_columns(self.dataframe, 'NonexistentColumn1', 'NonexistentColumn2', 'Result')



if __name__ == '__main__':
    unittest.main()
