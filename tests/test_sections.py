import os
import re
import unittest

import pandas as pd

from mitools.exceptions.custom_exceptions import ArgumentKeyError
from mitools.notebooks import (FULL_TEXT_COLUMN, create_full_text_column, etl,
                               filter_text_rows_by_pattern,
                               merge_csvs_into_dataframe, read_and_concat_csvs,
                               rename_columns)


class TestReadAndConcatCSVs(unittest.TestCase):
    def setUp(self):
        # Setup code to create test CSV files and directories if needed
        self.test_folder = "test_data"
        os.makedirs(self.test_folder, exist_ok=True)
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        df1.to_csv(os.path.join(self.test_folder, "file1.csv"))
        df2.to_csv(os.path.join(self.test_folder, "file2.csv"))

    def tearDown(self):
        # Teardown code to remove test files and directories after tests
        for file in os.listdir(self.test_folder):
            os.remove(os.path.join(self.test_folder, file))
        os.rmdir(self.test_folder)

    def test_valid_csv_files(self):
        result = read_and_concat_csvs(self.test_folder)
        self.assertEqual(result.shape[0], 4)  # Assuming axis=0

    def test_empty_directory(self):
        empty_folder = "empty_test_data"
        os.makedirs(empty_folder, exist_ok=True)
        with self.assertRaises(ValueError):  # Assuming it raises ValueError on empty input
            read_and_concat_csvs(empty_folder)
        os.rmdir(empty_folder)

    def test_non_csv_files(self):
        # Create a non-CSV file in the test folder
        with open(os.path.join(self.test_folder, "not_a_csv.txt"), 'w') as file:
            file.write("This is not a CSV file.")
        result = read_and_concat_csvs(self.test_folder)
        self.assertEqual(result.shape[0], 4)  # Assuming axis=0 and only CSV files are read

    def test_axis_parameter(self):
        result = read_and_concat_csvs(self.test_folder, axis=1)
        self.assertEqual(result.shape[1], 4)  # Assuming axis=1


class TestCreateFullTextColumn(unittest.TestCase):

    def test_valid_dataframe(self):
        dataframe = pd.DataFrame({
            'A': ['Hello', 'This'],
            'B': ['World', 'is'],
            'C': ['!', 'a test']
        })
        text_columns = ['A', 'B', 'C']
        result = create_full_text_column(dataframe, text_columns)
        expected = pd.Series(['Hello World !', 'This is a test'], name=FULL_TEXT_COLUMN)
        pd.testing.assert_series_equal(result[FULL_TEXT_COLUMN], expected)

    def test_non_string_columns(self):
        dataframe = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4],
            'C': [5, 6]
        })
        text_columns = ['A', 'B', 'C']
        result = create_full_text_column(dataframe, text_columns)
        expected = pd.Series(['1 3 5', '2 4 6'], name=FULL_TEXT_COLUMN)
        pd.testing.assert_series_equal(result[FULL_TEXT_COLUMN], expected)

    def test_missing_columns(self):
        dataframe = pd.DataFrame({
            'A': ['Hello', 'This'],
            'B': ['World', 'is']
        })
        text_columns = ['A', 'B', 'C']  # 'C' does not exist in dataframe
        with self.assertRaises(ArgumentKeyError):
            create_full_text_column(dataframe, text_columns)

    def test_empty_dataframe_with_columns(self):
        text_columns = ['A', 'B', 'C']
        dataframe = pd.DataFrame(columns=text_columns)
        result = create_full_text_column(dataframe, text_columns)
        self.assertTrue(result.empty)

    def test_empty_dataframe(self):
        text_columns = ['A', 'B', 'C']
        dataframe = pd.DataFrame()
        with self.assertRaises(ArgumentKeyError):
            create_full_text_column(dataframe, text_columns)


    def test_special_characters(self):
        dataframe = pd.DataFrame({
            'A': ['Hello', 'This'],
            'B': ['World!', 'is:'],
            'C': ['Here', 'a test']
        })
        text_columns = ['A', 'B', 'C']
        result = create_full_text_column(dataframe, text_columns)
        expected = pd.Series(['Hello World! Here', 'This is: a test'], name=FULL_TEXT_COLUMN)
        pd.testing.assert_series_equal(result[FULL_TEXT_COLUMN], expected)


class TestRenameColumns(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

    def test_valid_rename(self):
        columns_map = {'A': 'X', 'B': 'Y', 'C': 'Z'}
        result_df = rename_columns(self.df, columns_map)
        self.assertListEqual(list(result_df.columns), ['X', 'Y', 'Z'])

    def test_invalid_keys(self):
        columns_map = {'A': 'X', 'D': 'Y'}
        with self.assertRaises(ArgumentKeyError):
            rename_columns(self.df, columns_map)

    def test_non_unique_values(self):
        columns_map = {'A': 'X', 'B': 'X'}
        with self.assertRaises(ArgumentKeyError):
            rename_columns(self.df, columns_map)

    def test_preexisting_column_names(self):
        columns_map = {'A': 'B', 'C': 'D'}
        with self.assertRaises(ArgumentKeyError):
            rename_columns(self.df, columns_map)

    def test_inverse_map(self):
        columns_map = {'X': 'A', 'Y': 'B', 'Z': 'C'}
        result_df = rename_columns(self.df, columns_map, inverse_map=True)
        self.assertListEqual(list(result_df.columns), ['X', 'Y', 'Z'])

    def test_inverse_map_with_none(self):
        columns_map = {'X': 'A', 'Y': None, 'Z': 'C'}
        result_df = rename_columns(self.df, columns_map, inverse_map=True)
        self.assertListEqual(list(result_df.columns), ['X', 'B', 'Z'])


class TestFilterTextRowsByPattern(unittest.TestCase):

    def setUp(self):
        self.dataframe = pd.DataFrame({
            'text_column': ['Hello World', 'hello world', 'HELLO WORLD', 'goodbye']
        })

    def test_valid_pattern(self):
        pattern = 'hello'
        result = filter_text_rows_by_pattern(self.dataframe, 'text_column', pattern)
        self.assertEqual(len(result), 3)  # 'Hello World' and 'hello world'

    def test_case_sensitive(self):
        pattern = 'hello'
        result = filter_text_rows_by_pattern(self.dataframe, 'text_column', pattern, case=True)
        self.assertEqual(len(result), 1)  # Only 'hello world'

    def test_non_existent_column(self):
        pattern = 'hello'
        with self.assertRaises(ArgumentKeyError):
            filter_text_rows_by_pattern(self.dataframe, 'nonexistent_column', pattern)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        pattern = 'hello'
        with self.assertRaises(ArgumentKeyError):
            filter_text_rows_by_pattern(empty_df, 'text_column', pattern)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['text_column'])
        pattern = 'hello'
        result = filter_text_rows_by_pattern(empty_df, 'text_column', pattern)
        pd.testing.assert_frame_equal(empty_df, result)

    def test_special_regex_pattern(self):
        pattern = '^H[a-z]+o\sWorld$'
        result = filter_text_rows_by_pattern(self.dataframe, 'text_column', pattern, case=True)
        self.assertEqual(len(result), 1)  # 'Hello World'

    def test_special_regex_pattern(self):
        pattern = '^h[a-z]+o\sworld$'
        result = filter_text_rows_by_pattern(self.dataframe, 'text_column', pattern, case=False)
        self.assertEqual(len(result), 3)  # 'Hello World'


class TestMergeCSVsIntoDataFrame(unittest.TestCase):
    
    def setUp(self):
        # Setup code to create test CSV files and directories if needed
        self.test_folder = "test_csv_data"
        os.makedirs(self.test_folder, exist_ok=True)
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [2, 5], 'B': [4, 8]})
        df1.to_csv(os.path.join(self.test_folder, "file1.csv"))
        df2.to_csv(os.path.join(self.test_folder, "file2.csv"))

    def tearDown(self):
        # Teardown code to remove test files and directories after tests
        for file in os.listdir(self.test_folder):
            os.remove(os.path.join(self.test_folder, file))
        os.rmdir(self.test_folder)

    def test_multiple_csv_files(self):
        result = merge_csvs_into_dataframe(self.test_folder)
        self.assertEqual(len(result), 3)  # Expect 3 unique rows

    def test_single_csv_file(self):
        single_file_folder = "single_csv_test"
        os.makedirs(single_file_folder, exist_ok=True)
        df = pd.DataFrame({'C': [1, 2], 'D': [3, 4]})
        df.to_csv(os.path.join(single_file_folder, "file.csv"))
        result = merge_csvs_into_dataframe(single_file_folder)
        self.assertEqual(len(result), 2)  # Two rows from the single file
        for file in os.listdir(single_file_folder):
            os.remove(os.path.join(single_file_folder, file))
        os.rmdir(single_file_folder)

    def test_empty_folder(self):
        empty_folder = "empty_csv_test"
        os.makedirs(empty_folder, exist_ok=True)
        with self.assertRaises(ValueError):  # Assuming it raises ValueError on empty input
            merge_csvs_into_dataframe(empty_folder)
        os.rmdir(empty_folder)

    def test_non_csv_files(self):
        # Create a non-CSV file in the test folder
        with open(os.path.join(self.test_folder, "not_a_csv.txt"), 'w') as file:
            file.write("This is not a CSV file.")
        result = merge_csvs_into_dataframe(self.test_folder)
        self.assertEqual(len(result), 3)  # Expect 3 unique rows from CSVs only


class TestETL(unittest.TestCase):
    def setUp(self):
        # Setup for ETL test
        self.test_csv_folder = "test_csv_folder"
        os.makedirs(self.test_csv_folder, exist_ok=True)
        df1 = pd.DataFrame({'A': ['Hello', 'Bye'], 'B': ['World', 'Moon']}).reset_index()
        df2 = pd.DataFrame({'A': ['Good', 'Bad'], 'B': ['Day', 'Night']}).reset_index()
        df1.to_csv(os.path.join(self.test_csv_folder, "file1.csv"), index=False)
        df2.to_csv(os.path.join(self.test_csv_folder, "file2.csv"), index=False)

        self.df_path = "test_output.parquet"
        self.columns_map = {'A': 'Greeting', 'B': 'Object'}
        self.text_columns = ['Greeting', 'Object']
        self.pattern = 'Hello'
        self.filter_col = 'Greeting'
        self.recalculate = True

    def tearDown(self):
        # Clean up test files and folders
        for file in os.listdir(self.test_csv_folder):
            os.remove(os.path.join(self.test_csv_folder, file))
        os.rmdir(self.test_csv_folder)
        if os.path.exists(self.df_path):
            os.remove(self.df_path)

    def test_full_etl_process(self):
        result = etl(self.df_path, self.test_csv_folder, self.columns_map, 
                     self.text_columns, self.pattern, self.filter_col, self.recalculate)
        print(result)
        expected_rows = 1  # Assuming only 'Hello World' matches the pattern
        self.assertEqual(len(result), expected_rows)
        self.assertTrue(all(col in result.columns for col in self.text_columns))

if __name__ == '__main__':
    unittest.main()
