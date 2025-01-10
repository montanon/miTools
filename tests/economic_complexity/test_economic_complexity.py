import time
import unittest
from unittest import TestCase

import numpy as np
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal

from mitools.economic_complexity import (
    calculate_economic_complexity,
    calculate_exports_matrix_rca,
    calculate_proximity_matrix,
    calculate_relatedness_matrix,
    exports_data_to_matrix,
    mask_matrix,
)
from mitools.exceptions.custom_exceptions import ArgumentValueError


class TestExportsDataToMatrix(TestCase):
    def setUp(self):
        self.dataframe = DataFrame(
            {
                "origin": ["USA", "USA", "CAN", "CAN"],
                "product_code": ["A", "B", "A", "C"],
                "export_value": [100, 200, 300, 400],
            }
        )
        self.products_codes = DataFrame({"product_code": ["A", "B", "C"]})

    def test_valid_data(self):
        result = exports_data_to_matrix(
            self.dataframe,
            origin_col="origin",
            products_cols=["product_code"],
            value_col="export_value",
            products_codes=self.products_codes,
        )
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.loc["USA", "A"], 100)
        self.assertEqual(result.loc["USA", "B"], 200)
        self.assertEqual(result.loc["CAN", "A"], 300)
        self.assertEqual(result.loc["CAN", "C"], 400)

    def test_missing_product_codes(self):
        products_codes = DataFrame({"product_code": ["A", "B", "C", "D"]})
        result = exports_data_to_matrix(
            self.dataframe,
            origin_col="origin",
            products_cols=["product_code"],
            value_col="export_value",
            products_codes=products_codes,
        )
        self.assertEqual(result.loc["USA", "D"], 0)
        self.assertEqual(result.loc["CAN", "D"], 0)

    def test_total_value_consistency(self):
        result = exports_data_to_matrix(
            self.dataframe,
            origin_col="origin",
            products_cols=["product_code"],
            value_col="export_value",
            products_codes=self.products_codes,
        )
        self.assertEqual(self.dataframe["export_value"].sum(), result.values.sum())

    def test_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            exports_data_to_matrix(
                self.dataframe.drop(columns=["export_value"]),
                origin_col="origin",
                products_cols=["product_code"],
                value_col="export_value",
                products_codes=self.products_codes,
            )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=["origin", "product_code", "export_value"])
        with self.assertRaises(ArgumentValueError):
            exports_data_to_matrix(
                empty_df,
                origin_col="origin",
                products_cols=["product_code"],
                value_col="export_value",
                products_codes=self.products_codes,
            )


class TestCalculateExportsMatrixRCA(TestCase):
    def setUp(self):
        self.exports_matrix = DataFrame(
            {
                "Product A": [100, 200, 300],
                "Product B": [400, 500, 600],
                "Product C": [700, 800, 900],
            },
            index=["USA", "CAN", "MEX"],
        )

    def test_valid_rca_calculation(self):
        result = calculate_exports_matrix_rca(self.exports_matrix)
        total_exports = self.exports_matrix.sum().sum()
        origin_totals = self.exports_matrix.sum(axis=1)
        product_totals = self.exports_matrix.sum(axis=0)
        expected_rca = self.exports_matrix * total_exports
        expected_rca = expected_rca.div(product_totals, axis=1)
        expected_rca = expected_rca.div(origin_totals, axis=0)
        expected_rca = expected_rca.fillna(0.0)
        assert_frame_equal(result, expected_rca)

    def test_rca_with_zeros(self):
        exports_matrix_with_zeros = self.exports_matrix.copy()
        exports_matrix_with_zeros.loc["USA", "Product A"] = 0
        result = calculate_exports_matrix_rca(exports_matrix_with_zeros)
        self.assertFalse(result.isna().any().any())

    def test_empty_exports_matrix(self):
        empty_matrix = DataFrame()
        result = calculate_exports_matrix_rca(empty_matrix)
        self.assertTrue(result.empty)

    def test_single_origin_and_product(self):
        single_matrix = DataFrame({"Product A": [100]}, index=["USA"])
        result = calculate_exports_matrix_rca(single_matrix)
        expected = DataFrame({"Product A": [1.0]}, index=["USA"])
        assert_frame_equal(result, expected)

    def test_nan_values_handling(self):
        matrix_with_nan = self.exports_matrix.copy()
        matrix_with_nan.loc["USA", "Product B"] = None
        result = calculate_exports_matrix_rca(matrix_with_nan)
        self.assertEqual(result.loc["USA", "Product B"], 0.0)

    def test_non_numeric_values(self):
        matrix_with_non_numeric = self.exports_matrix.copy()
        matrix_with_non_numeric.loc["USA", "Product B"] = "invalid"
        with self.assertRaises(ArgumentValueError):
            calculate_exports_matrix_rca(matrix_with_non_numeric)


class TestMaskMatrix(TestCase):
    def setUp(self):
        self.matrix = DataFrame(
            {"A": [0.2, 0.5, 0.8], "B": [0.3, 0.6, 0.9], "C": [0.1, 0.4, 0.7]}
        )

    def test_mask_with_valid_threshold(self):
        result = mask_matrix(self.matrix, threshold=0.5)
        expected = DataFrame(
            {"A": [0.0, 1.0, 1.0], "B": [0.0, 1.0, 1.0], "C": [0.0, 0.0, 1.0]}
        )
        assert_frame_equal(result, expected)

    def test_mask_with_zero_threshold(self):
        result = mask_matrix(self.matrix, threshold=0.0)
        expected = DataFrame(
            {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0], "C": [1.0, 1.0, 1.0]}
        )
        assert_frame_equal(result, expected)

    def test_mask_with_high_threshold(self):
        result = mask_matrix(self.matrix, threshold=1.0)
        expected = DataFrame(
            {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0], "C": [0.0, 0.0, 0.0]}
        )
        assert_frame_equal(result, expected)

    def test_mask_with_negative_threshold(self):
        result = mask_matrix(self.matrix, threshold=-1.0)
        expected = DataFrame(
            {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0], "C": [1.0, 1.0, 1.0]}
        )
        assert_frame_equal(result, expected)

    def test_mask_on_empty_matrix(self):
        empty_matrix = DataFrame()
        result = mask_matrix(empty_matrix, threshold=0.5)
        assert_frame_equal(result, empty_matrix)

    def test_mask_with_nan_values(self):
        matrix_with_nan = self.matrix.copy()
        matrix_with_nan.loc[0, "A"] = None
        result = mask_matrix(matrix_with_nan, threshold=0.5)
        expected = DataFrame(
            {"A": [None, 1.0, 1.0], "B": [0.0, 1.0, 1.0], "C": [0.0, 0.0, 1.0]}
        )
        assert_frame_equal(result, expected)

    def test_mask_with_non_numeric_values(self):
        matrix_with_non_numeric = self.matrix.copy()
        matrix_with_non_numeric.loc[0, "A"] = "invalid"
        with self.assertRaises(ArgumentValueError):
            mask_matrix(matrix_with_non_numeric, threshold=0.5)

    def test_mask_with_invalid_threshold(self):
        with self.assertRaises(ArgumentValueError):
            mask_matrix(self.matrix, threshold="invalid_threshold")


class TestCalculateProximityMatrix(TestCase):
    def setUp(self):
        self.dataframe = DataFrame(
            {"Product A": [1, 0, 0], "Product B": [0, 1, 1], "Product C": [1, 1, 0]},
            index=["USA", "CAN", "MEX"],
        )

    def test_valid_proximity_matrix(self):
        result = calculate_proximity_matrix(self.dataframe, symmetric=True)
        expected = DataFrame(
            {
                "Product A": [0.0, 0.0, 0.5],
                "Product B": [0.0, 0.0, 0.5],
                "Product C": [0.5, 0.5, 0.0],
            },
            index=["Product A", "Product B", "Product C"],
        )
        assert_frame_equal(result, expected)

    def test_asymmetric_proximity_matrix(self):
        result = calculate_proximity_matrix(self.dataframe, symmetric=False)
        expected = DataFrame(
            {
                "Product A": [0.0, 0.0, 1.0],
                "Product B": [0.0, 0.0, 0.5],
                "Product C": [0.5, 0.5, 0.0],
            },
            index=["Product A", "Product B", "Product C"],
        )
        assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        result = calculate_proximity_matrix(empty_df)
        self.assertTrue(result.empty)

    def test_single_product(self):
        single_product_df = DataFrame(
            {"Product A": [1, 0, 1]}, index=["USA", "CAN", "MEX"]
        )
        result = calculate_proximity_matrix(single_product_df)
        expected = DataFrame({"Product A": [0.0]}, index=["Product A"])
        assert_frame_equal(result, expected)

    def test_nan_handling(self):
        dataframe_with_nan = self.dataframe.copy()
        dataframe_with_nan.loc["USA", "Product B"] = np.nan
        with self.assertRaises(ArgumentValueError):
            calculate_proximity_matrix(dataframe_with_nan)


class TestCalculateRelatednessMatrix(TestCase):
    def setUp(self):
        self.proximity_matrix = DataFrame(
            {
                "Product A": [1.0, 0.5, 0.2],
                "Product B": [0.5, 1.0, 0.3],
                "Product C": [0.2, 0.3, 1.0],
            },
            index=["Product A", "Product B", "Product C"],
        )
        self.rca_matrix = DataFrame(
            {"Product A": [1, 0, 1], "Product B": [0, 1, 1], "Product C": [1, 1, 0]},
            index=["USA", "CAN", "MEX"],
        )

    def test_valid_relatedness_matrix(self):
        result = calculate_relatedness_matrix(self.proximity_matrix, self.rca_matrix)
        expected_data = {
            ("Product A", "USA"): 0.705882,
            ("Product A", "CAN"): 0.411765,
            ("Product A", "MEX"): 0.882353,
            ("Product B", "USA"): 0.444444,
            ("Product B", "CAN"): 0.722222,
            ("Product B", "MEX"): 0.833333,
            ("Product C", "USA"): 0.800000,
            ("Product C", "CAN"): 0.866667,
            ("Product C", "MEX"): 0.333333,
        }
        expected = Series(expected_data, name="relatedness").to_frame()
        assert_frame_equal(result, expected)

    def test_relatedness_col(self):
        result = calculate_relatedness_matrix(
            self.proximity_matrix, self.rca_matrix, relatedness_col="w"
        )
        self.assertListEqual(list(result.columns), ["w"])

    def test_empty_matrices(self):
        empty_proximity = DataFrame()
        empty_rca = DataFrame()
        result = calculate_relatedness_matrix(empty_proximity, empty_rca)
        self.assertTrue(result.empty)

    def test_mismatched_indices(self):
        mismatched_rca = DataFrame(
            {"Product X": [1, 0, 1], "Product Y": [0, 1, 1], "Product Z": [1, 1, 0]},
            index=["USA", "CAN", "MEX"],
        )
        with self.assertRaises(ArgumentValueError):
            calculate_relatedness_matrix(self.proximity_matrix, mismatched_rca)

    def test_nan_handling(self):
        proximity_with_nan = self.proximity_matrix.copy()
        proximity_with_nan.loc["Product A", "Product B"] = None
        with self.assertRaises(ArgumentValueError):
            calculate_relatedness_matrix(proximity_with_nan, self.rca_matrix)


class TestCalculateEconomicComplexity(TestCase):
    def setUp(self):
        self.rca_matrix = DataFrame(
            {"Product A": [10, 5, 1], "Product B": [0, 5, 3], "Product C": [15, 5, 0]},
            index=["USA", "CAN", "MEX"],
        )

    def test_valid_economic_complexity(self):
        eci_df, pci_df = calculate_economic_complexity(self.rca_matrix)
        expected_eci = DataFrame(
            {"ECI": [1.176965, 0.090536, -1.267500]}, index=["USA", "CAN", "MEX"]
        )
        expected_pci = DataFrame(
            {"PCI": [0.889001, 0.508001, -1.397001]},
            index=["Product C", "Product A", "Product B"],
        )
        assert_frame_equal(eci_df, expected_eci)
        assert_frame_equal(pci_df, expected_pci)

    def test_standardization(self):
        eci_df, pci_df = calculate_economic_complexity(
            self.rca_matrix, standardize=False
        )
        self.assertNotAlmostEqual(eci_df["ECI"].mean(), 0.0, places=2)
        self.assertNotAlmostEqual(eci_df["ECI"].std(), 1.0, places=2)
        self.assertNotAlmostEqual(pci_df["PCI"].mean(), 0.0, places=2)
        self.assertNotAlmostEqual(pci_df["PCI"].std(), 1.0, places=2)

    def test_empty_rca_matrix(self):
        empty_rca = DataFrame()
        with self.assertRaises(ArgumentValueError):
            calculate_economic_complexity(empty_rca)

    def test_nan_handling(self):
        rca_with_nan = self.rca_matrix.copy()
        rca_with_nan.loc["USA", "Product B"] = np.nan
        with self.assertRaises(ArgumentValueError):
            calculate_economic_complexity(rca_with_nan)

    def test_single_country_and_product(self):
        single_rca = DataFrame({"Product A": [10]}, index=["USA"])
        with self.assertRaises(ArgumentValueError):
            calculate_economic_complexity(single_rca)

    def test_double_country_and_product(self):
        single_rca = DataFrame(
            {"Product A": [10, 5], "Product B": [2, 3]}, index=["USA", "CHN"]
        )
        eci_df, pci_df = calculate_economic_complexity(single_rca)
        expected_eci = DataFrame({"ECI": [1.0, -1.0]}, index=["USA", "CHN"])
        expected_pci = DataFrame(
            {"PCI": [1.0, -1.0]},
            index=["Product A", "Product B"],
        )
        assert_frame_equal(eci_df, expected_eci)
        assert_frame_equal(pci_df, expected_pci)

    def test_fast_option(self):
        _N = 4
        large_rca = DataFrame(np.random.rand(100, 100))
        fast_times = []
        for _ in range(_N):
            start_time = time.time()
            calculate_economic_complexity(large_rca, fast=True)
            fast_times.append(time.time() - start_time)
        fast_time = np.mean(fast_times[1:])
        standard_times = []
        for _ in range(_N):
            start_time = time.time()
            calculate_economic_complexity(large_rca, fast=False)
            standard_times.append(time.time() - start_time)
        standard_time = np.mean(standard_times)
        self.assertLess(
            fast_time,
            standard_time / 2,
            f"Fast version ({fast_time:.4f}s) is not significantly faster than standard version ({standard_time:.4f}s)",
        )
        eci_fast, pci_fast = calculate_economic_complexity(self.rca_matrix, fast=True)
        eci_standard, pci_standard = calculate_economic_complexity(
            self.rca_matrix, fast=False
        )
        assert_frame_equal(eci_fast, eci_standard)
        assert_frame_equal(pci_fast, pci_standard)
        print(f"Uncompiled Fast version time: {fast_times[0]:.4f}s")
        print(f"Fast version time: {fast_time:.4f}s")
        print(f"Standard version time: {standard_time:.4f}s")


if __name__ == "__main__":
    unittest.main()
