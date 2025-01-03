import unittest
from typing import Dict, List
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series

from mitools.exceptions import (
    ArgumentTypeError,
    ArgumentValueError,
    ColumnValidationError,
)
from mitools.utils import (
    sankey_plot_validation,
    validate_args_types,
    validate_dataframe_structure,
)


class TestValidateTypesDecorator(TestCase):
    def test_correct_types_positional_arguments(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        self.assertTrue(test_func(10, "hello"))

    def test_correct_types_keyword_arguments(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        self.assertTrue(test_func(x=10, y="hello"))

    def test_incorrect_type_positional_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        with self.assertRaises(ArgumentTypeError) as context:
            test_func(10, 20)  # y should be a str, not an int
        self.assertIn("Argument 'y' must be of type str", str(context.exception))

    def test_incorrect_type_keyword_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        with self.assertRaises(ArgumentTypeError) as context:
            test_func(x=10, y=20)  # y should be a str, not an int
        self.assertIn("Argument 'y' must be of type str", str(context.exception))

    def test_missing_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x):
            return True

        with self.assertRaises(ArgumentValueError):
            test_func(10)  # Missing argument 'y'

    def test_extra_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y, z):
            return True

        self.assertTrue(test_func(10, "hello", "extra argument"))

    def test_multiple_arguments_different_types(self):
        @validate_args_types(a=int, b=float, c=str)
        def test_func(a, b, c):
            return True

        self.assertTrue(test_func(5, 3.14, "test"))

    def test_multiple_incorrect_arguments(self):
        @validate_args_types(a=int, b=float, c=str)
        def test_func(a, b, c):
            return True

        with self.assertRaises(ArgumentTypeError) as context:
            test_func(5, "not a float", 10)  # b is incorrect
        self.assertIn("Argument 'b' must be of type float", str(context.exception))

    def test_unexpected_argument_name(self):
        @validate_args_types(a=int, b=str)
        def test_func(x, y):
            return True

        with self.assertRaises(ArgumentValueError) as context:
            test_func(5, "hello")
        self.assertIn(
            "Argument 'a' not found in function signature", str(context.exception)
        )

    def test_with_default_values(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y="default"):
            return True

        self.assertTrue(test_func(5))  # y should use the default value, no TypeError

    def test_type_check_on_default_value(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y="default"):
            return True

        with self.assertRaises(TypeError):
            test_func(
                5, y=10
            )  # y should be a str, not an int, even with default values present

    def test_no_type_validation_when_not_specified(self):
        @validate_args_types(x=int)
        def test_func(x, y):
            return True

        self.assertTrue(
            test_func(5, "anything")
        )  # y has no specified type, so any type is allowed


def custom_validation(
    dataframe: DataFrame,
    required_columns: List[str] = None,
    column_types: Dict[str, str] = None,
) -> bool:
    if required_columns:
        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            raise ArgumentValueError(
                f"DataFrame is missing required columns: {missing_columns}"
            )
    if column_types:
        for col, expected_type in column_types.items():
            if col in dataframe.columns and not pd.api.types.is_dtype_equal(
                dataframe[col].dtype, expected_type
            ):
                raise ArgumentTypeError(
                    f"Column '{col}' must be of type {expected_type}. Found {dataframe[col].dtype} instead."
                )


@validate_dataframe_structure(
    dataframe_name="data",
    validation=custom_validation,
    required_columns=["column1", "column2"],
    column_types={"column1": "int64", "column2": "float64"},
)
def process_data(data: DataFrame) -> str:
    return "Data processed successfully"


class TestValidateDataFrameStructureDecorator(TestCase):
    def setUp(self):
        self.correct_df = DataFrame(
            {
                "column1": Series([1, 2, 3], dtype="int64"),
                "column2": Series([1.0, 2.0, 3.0], dtype="float64"),
            }
        )
        self.missing_column_df = DataFrame(
            {"column1": Series([1, 2, 3], dtype="int64")}
        )
        self.wrong_type_df = DataFrame(
            {
                "column1": Series([1.0, 2.0, 3.0], dtype="float64"),  # Should be int64
                "column2": Series([1.0, 2.0, 3.0], dtype="float64"),
            }
        )
        self.extra_columns_df = DataFrame(
            {
                "column1": Series([1, 2, 3], dtype="int64"),
                "column2": Series([1.0, 2.0, 3.0], dtype="float64"),
                "extra_column": Series(["a", "b", "c"], dtype="object"),
            }
        )
        self.non_dataframe_input = "Not a DataFrame"

    def test_correct_dataframe_structure(self):
        result = process_data(data=self.correct_df)
        self.assertEqual(result, "Data processed successfully")

    def test_missing_required_column(self):
        with self.assertRaises(ArgumentValueError) as context:
            process_data(data=self.missing_column_df)
        self.assertIn(
            "DataFrame is missing required columns: ['column2']", str(context.exception)
        )

    def test_incorrect_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            process_data(data=self.wrong_type_df)
        self.assertIn(
            "Column 'column1' must be of type int64. Found float64 instead.",
            str(context.exception),
        )

    def test_extra_columns(self):
        result = process_data(data=self.extra_columns_df)
        self.assertEqual(result, "Data processed successfully")

    def test_non_dataframe_input(self):
        with self.assertRaises(ArgumentTypeError) as context:
            process_data(data=self.non_dataframe_input)
        self.assertIn("must be a DataFrame.", str(context.exception))

    def test_empty_dataframe(self):
        empty_correct_df = DataFrame(
            {
                "column1": Series([], dtype="int64"),
                "column2": Series([], dtype="float64"),
            }
        )
        result = process_data(data=empty_correct_df)
        self.assertEqual(result, "Data processed successfully")

    def test_partial_column_types(self):
        @validate_dataframe_structure(
            dataframe_name="data",
            validation=custom_validation,
            required_columns=["column1"],
            column_types={"column1": "int64"},
        )
        def partial_type_check(data: DataFrame) -> str:
            return "Data processed with partial column type check"

        result = partial_type_check(data=self.correct_df)
        self.assertEqual(result, "Data processed with partial column type check")

    def test_no_validation_criteria(self):
        @validate_dataframe_structure(
            dataframe_name="data",
            validation=custom_validation,
        )
        def no_criteria_check(data: DataFrame) -> str:
            return "Data processed with no criteria"

        result = no_criteria_check(data=self.correct_df)
        self.assertEqual(result, "Data processed with no criteria")

    def test_missing_required_column_with_partial_check(self):
        @validate_dataframe_structure(
            dataframe_name="data",
            validation=custom_validation,
            required_columns=["column1", "column2"],
        )
        def partial_column_check(data: DataFrame) -> str:
            return "Data processed with partial column check"

        with self.assertRaises(ArgumentValueError) as context:
            partial_column_check(data=self.missing_column_df)
        self.assertIn(
            "DataFrame is missing required columns: ['column2']", str(context.exception)
        )


class TestValidateDataFrameColumns(TestCase):
    def setUp(self):
        correct_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "2_3-Gram", "Gram"),
                ("(2000, 2020)", "2_3-Gram", "Count"),
                ("(2010, 2020)", "1_2-Gram", "Gram"),
                ("(2010, 2020)", "1_2-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.correct_df = pd.DataFrame(
            [
                ["example1", 10, "example2", 15],
                [None, 20, "example3", np.nan],
            ],
            columns=correct_index,
        )
        invalid_level_0_index = MultiIndex.from_tuples(
            [
                ("2000-2020", "2_3-Gram", "Gram"),
                ("2000-2020", "2_3-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.invalid_level_0_df = pd.DataFrame(
            [["example", 5], [None, 10]], columns=invalid_level_0_index
        )
        invalid_level_1_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "3_2-Gram", "Gram"),
                ("(2000, 2020)", "3_2-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.invalid_level_1_df = pd.DataFrame(
            [["example", 5], [None, 10]], columns=invalid_level_1_index
        )
        invalid_level_2_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "2_3-Gram", "Frequency"),
                ("(2000, 2020)", "2_3-Gram", "Sum"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.invalid_level_2_df = pd.DataFrame(
            [["example", 5], [None, 10]], columns=invalid_level_2_index
        )
        self.invalid_gram_column_df = self.correct_df.copy()
        self.invalid_gram_column_df[("(2000, 2020)", "2_3-Gram", "Gram")] = [123, None]
        self.invalid_count_column_df = self.correct_df.copy()
        self.invalid_count_column_df[("(2000, 2020)", "2_3-Gram", "Count")] = [
            "not a number",
            np.nan,
        ]

    def test_correct_dataframe(self):
        self.assertTrue(sankey_plot_validation(self.correct_df))

    def test_invalid_level_0_format(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_level_0_df)
        self.assertIn("Level 0 column", str(context.exception))

    def test_invalid_level_1_format(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_level_1_df)
        self.assertIn("Level 1 column", str(context.exception))

    def test_invalid_level_2_names(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_level_2_df)
        self.assertIn("Level 2 column", str(context.exception))

    def test_invalid_gram_column_values(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_gram_column_df)
        self.assertIn(
            "Level 2 'Gram' columns must contain strings or NaN", str(context.exception)
        )

    def test_invalid_count_column_values(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_count_column_df)
        self.assertIn(
            "Level 2 'Count' columns must contain numeric values or NaN",
            str(context.exception),
        )

    def test_empty_dataframe(self):
        empty_correct_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "2_3-Gram", "Gram"),
                ("(2000, 2020)", "2_3-Gram", "Count"),
                ("(2010, 2020)", "1_2-Gram", "Gram"),
                ("(2010, 2020)", "1_2-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        empty_correct_df = pd.DataFrame(columns=empty_correct_index)
        self.assertTrue(sankey_plot_validation(empty_correct_df))


if __name__ == "__main__":
    unittest.main()
