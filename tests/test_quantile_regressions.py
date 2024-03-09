import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from mitools.regressions import QuantileRegStrs, create_regression_file_paths


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

if __name__ == '__main__':
    unittest.main()
