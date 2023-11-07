import unittest
from typing import Callable, List

from pandas import DataFrame, Index, MultiIndex, testing
from pandas.api.types import is_numeric_dtype

# Assuming the provided function is imported or defined here
from mitools.economic_complexity.columns import *
from mitools.exceptions.custom_exceptions import ArgumentKeyError, ArgumentTypeError


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
            [[i + j for j in range(6)] for i in range(26)],
            index=index,
            columns=columns
        )

    def test_absolute_variation(self):
        # Test the function with absolute variation
        t = 1
        result = variation_columns(self.data, ['Indicator1'], t, pct=False)
        expected_change_name = GROWTH_COLUMN_NAME.format(t)
        self.assertTrue(expected_change_name in result.columns.get_level_values(1))

        # Test the correctness of calculations
        original_values = self.data.loc[:, ('CountryA', 'Indicator1')]
        shifted_values = original_values.shift(t)
        expected_values = original_values - shifted_values
        testing.assert_series_equal(result[('CountryA', f'Indicator1{expected_change_name}')], expected_values)

    def test_percentage_variation(self):
        # Test the function with percentage variation
        t = 1
        result = variation_columns(self.data, ['Indicator1'], t, pct=True)
        expected_change_name = GROWTH_PCT_COLUMN_NAME.format(t)
        self.assertTrue(expected_change_name in result.columns.get_level_values(1))

        # Test the correctness of calculations
        original_values = self.data.loc[:, ('CountryA', 'Indicator1')]
        shifted_values = original_values.shift(t)
        expected_values = ((original_values - shifted_values) / shifted_values) * 100
        testing.assert_series_equal(result[('CountryA', f'Indicator1{expected_change_name}')], expected_values)



if __name__ == '__main__':
    unittest.main()
