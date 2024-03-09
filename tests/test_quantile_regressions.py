import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
from pandas import DataFrame

from mitools.regressions import (
    QuantileRegStrs,
    create_regression_file_paths,
    get_group_data,
    prepare_x_values,
)


class TestQuantileRegStrs(unittest.TestCase):

    def test_constants_presence_type_and_exclusivity(self):
        # List of all expected constants in QuantileRegStrs
        expected_constants = set([
            'UNNAMED', 'COEF', 'T_VALUE', 'P_VALUE', 'VALUE',
            'QUANTILE', 'INDEPENDENT_VARS', 'REGRESSION_TYPE',
            'REGRESSION_DEGREE', 'DEPENDENT_VAR', 'VARIABLE_TYPE',
            'EXOG_VAR', 'CONTROL_VAR', 'ID', 'QUADRATIC_REG',
            'LINEAR_REG', 'QUADRATIC_VAR_SUFFIX', 'INDEPENDENT_VARS_PATTERN',
            'STATS', 'INTERCEPT', 'ANNOTATION', 'PARQUET_SUFFIX',
            'EXCEL_SUFFIX', 'MAIN_PLOT', 'PLOTS_SUFFIX'
        ])

        actual_constants = set([attr for attr in dir(QuantileRegStrs) if not attr.startswith("__")])

        # Check for unexpected attributes
        self.assertFalse(actual_constants - expected_constants, 
                         f"Unexpected attributes found: {actual_constants - expected_constants}")

        # Check for missing attributes
        self.assertFalse(expected_constants - actual_constants, 
                         f"Expected attributes not found: {expected_constants - actual_constants}")

        # Check types for the expected attributes
        for const in expected_constants:
            self.assertIsInstance(getattr(QuantileRegStrs, const), str)

    def test_immutability(self):
        with self.assertRaises(FrozenInstanceError):
            setattr(QuantileRegStrs(), 'UNNAMED', 'New Value')

class TestCreateRegressionFilePaths(unittest.TestCase):
    def setUp(self):
        # This list will hold the paths of all files created during the tests
        self.created_files = []

    def tearDown(self):
        # Delete all created files
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()

    def test_file_paths_creation(self):
        eci_type_folder = Path("/tmp/eci_type_folder")  # Use a temporary directory for testing
        regression_id = "123"
        main_plot, regression_plot = create_regression_file_paths(eci_type_folder, regression_id)
        # Add the created file paths to the list for cleanup
        self.created_files.extend([main_plot, regression_plot])
        expected_main_plot = eci_type_folder / f"{QuantileRegStrs.MAIN_PLOT}.png"
        expected_regression_plot = eci_type_folder / f"{regression_id}_{QuantileRegStrs.PLOTS_SUFFIX}.png"
        self.assertEqual(main_plot, expected_main_plot)
        self.assertEqual(regression_plot, expected_regression_plot)

    def test_input_types(self):
        eci_type_folder_str = "/tmp/eci_type_folder"  # Use a temporary directory for testing
        eci_type_folder_path = Path(eci_type_folder_str)
        regression_id = "123"
        main_plot_path, regression_plot_path = create_regression_file_paths(eci_type_folder_path, regression_id)
        main_plot_str, regression_plot_str = create_regression_file_paths(eci_type_folder_str, regression_id)
        # Add the created file paths to the list for cleanup
        self.created_files.extend([main_plot_path, regression_plot_path, main_plot_str, regression_plot_str])
        self.assertEqual(main_plot_path, main_plot_str)
        self.assertEqual(regression_plot_path, regression_plot_str)

class TestGetGroupData(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = DataFrame({
            'value': [1, 2, 3, 4],
            'group': ['A', 'B', 'A', 'B']
        }).set_index('group')

    def test_filtering_by_group(self):
        # Test filtering for a specific group
        result = get_group_data(self.data, 'A', 'group', 'All')
        self.assertEqual(len(result), 2)  # Expecting 2 rows for group 'A'
        self.assertTrue((result.index == 'A').all())  # All rows should belong to group 'A'

    def test_returning_all_data(self):
        # Test returning all data when group equals all_groups
        result = get_group_data(self.data, 'All', 'group', 'All')
        self.assertEqual(len(result), len(self.data))  # Should return the entire DataFrame

    def test_empty_dataframe(self):
        # Test with an empty DataFrame
        empty_df = DataFrame(columns=['value', 'group']).set_index('group')
        result = get_group_data(empty_df, 'A', 'group', 'All')
        self.assertTrue(result.empty)  # Result should also be an empty DataFrame

    def test_group_not_present(self):
        # Test with a group value not present in the DataFrame
        result = get_group_data(self.data, 'C', 'group', 'All')
        self.assertTrue(result.empty)  # Expecting an empty DataFrame

class TestPrepareXValues(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        self.group_data = DataFrame({
            'x1': np.random.rand(10),
            'x2': np.random.rand(10),
            'x3' + QuantileRegStrs.QUADRATIC_VAR_SUFFIX: np.random.rand(10),  # Quadratic variable
        })

    def test_correct_columns(self):
        # Test that the returned DataFrame has correct columns (excluding quadratic variables)
        independent_vars = ['x1', 'x2', 'x3' + QuantileRegStrs.QUADRATIC_VAR_SUFFIX]
        result = prepare_x_values(self.group_data, independent_vars)
        expected_columns = ['x1', 'x2']  # 'x3' with the quadratic suffix should be excluded
        self.assertListEqual(list(result.columns), expected_columns)

    def test_linear_spacing(self):
        # Test that the values in each column are linearly spaced
        independent_vars = ['x1', 'x2']
        result = prepare_x_values(self.group_data, independent_vars)
        for var in independent_vars:
            generated_values = result[var].to_numpy()
            self.assertTrue(np.allclose(np.diff(generated_values), np.diff(generated_values)[0]), f"Column {var} values are not linearly spaced")

    def test_empty_dataframe(self):
        # Test with an empty DataFrame
        empty_df = DataFrame()
        independent_vars = ['x1', 'x2']
        with self.assertRaises(KeyError):
            prepare_x_values(empty_df, independent_vars)

    def test_vars_not_in_dataframe(self):
        # Test with independent variables not present in the DataFrame
        independent_vars = ['x4', 'x5']  # Variables not in self.group_data
        with self.assertRaises(KeyError):
            prepare_x_values(self.group_data, independent_vars)


if __name__ == '__main__':
    unittest.main()
