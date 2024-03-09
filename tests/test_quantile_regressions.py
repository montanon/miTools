import unittest
from dataclasses import FrozenInstanceError

from mitools.regressions import QuantileRegStrs


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

if __name__ == '__main__':
    unittest.main()
