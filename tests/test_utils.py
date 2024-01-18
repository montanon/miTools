import unittest
from typing import List, Optional

import numpy as np
import pandas as pd

from mitools.utils import (
    BitArray,
    add_significance,
    auto_adjust_columns_width,
    build_dir_tree,
    can_convert_to,
    check_symmetrical_matrix,
    clean_str,
    dict_from_kwargs,
    display_env_variables,
    find_str_line_number_in_text,
    fuzz_ratio,
    fuzz_string_in_string,
    get_numbers_from_str,
    invert_dict,
    iprint,
    iterable_chunks,
    lcs_similarity,
    load_pkl_object,
    pretty_dict_str,
    read_text_file,
    remove_chars,
    remove_dataframe_duplicates,
    remove_multiple_spaces,
    replace_prefix,
    split_strings,
    store_pkl_object,
    str_is_number,
    stretch_string,
    unpack_list_of_lists,
)


class TestRemoveChars(unittest.TestCase):

    def test_basic_removal(self):
        self.assertEqual(remove_chars("Hello, World!", "lo"), "He, Wrd!")

    def test_no_chars_to_remove(self):
        self.assertEqual(remove_chars("Hello, World!", ""), "Hello, World!")

    def test_remove_all_characters(self):
        self.assertEqual(remove_chars("Hello, World!", "Helo, Wrd!"), "")

    def test_string_with_no_matching_characters(self):
        self.assertEqual(remove_chars("Hello, World!", "abc"), "Hello, World!")

    def test_empty_string_input(self):
        self.assertEqual(remove_chars("", "abc"), "")

    def test_special_characters(self):
        self.assertEqual(remove_chars("H@#llo, W$rld!", "@#$"), "Hllo, Wrld!")


class TestStretchString(unittest.TestCase):
    def test_normal_case(self):
        self.assertEqual(stretch_string("This is a sample string for testing purposes", 10),
                         "This is a\nsample\nstring for\ntesting\npurposes")

    def test_no_spaces(self):
        self.assertEqual(stretch_string("LongStringWithNoSpaces", 5),
                         "LongS\ntring\nWithN\noSpac\nes")

    def test_edge_cases(self):
        self.assertEqual(stretch_string("", 10), "")
        self.assertEqual(stretch_string("Short", 10), "Short")
        self.assertEqual(stretch_string("ExactlyTen", 10), "ExactlyTen")

    def test_whitespace_handling(self):
        self.assertEqual(stretch_string("  This   string has  weird spacing ", 10),
                         "This\nstring has\nweird\nspacing")

    def test_long_word(self):
        self.assertEqual(stretch_string("Supercalifragilisticexpialidocious", 10),
                         "Supercalif\nragilistic\nexpialidoc\nious")

class TestAutoAdjustColumnsWidth(unittest.TestCase):
    @patch('openpyxl.utils.get_column_letter', return_value='A')
    def test_auto_adjust_columns_width(self, mock_get_column_letter):
        # Create a mock worksheet with some columns
        wb = Workbook()
        ws = wb.active
        ws.append(['Hello', 'World'])
        ws.append(['Longer string here', 'Another string'])
        # Call the function to test
        auto_adjust_columns_width(ws)
        # Check that the width of the columns has been adjusted
        self.assertEqual(ws.column_dimensions['A'].width, 15)
        self.assertEqual(ws.column_dimensions['B'].width, 13)

class TestDisplayEnvVariables(unittest.TestCase):

    def setUp(self):
        self.env_vars = [
            ('small_int', 1),
            ('large_list', list(range(10000))),
            ('string', 'hello world'),
            ('large_dict', {i: i for i in range(1000)})
        ]

    def test_no_large_variables(self):
        threshold_mb = sys.getsizeof(self.env_vars[1][1]) / (1024**2) + 1
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertTrue(df.empty)

    def test_large_variables(self):
        threshold_mb = 0
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertFalse(df.empty)
        self.assertTrue(all(df['Size (MB)'] > threshold_mb))

    def test_edge_cases(self):
        # Empty env_vars
        df_empty = display_env_variables([], 0)
        self.assertTrue(df_empty.empty)
        # Extremely high threshold
        df_high_threshold = display_env_variables(self.env_vars, 1000000)
        self.assertTrue(df_high_threshold.empty)
        
    def test_different_data_types(self):
        threshold_mb = 0
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertIn('large_list', df['Variable'].values)
        self.assertIn('large_dict', df['Variable'].values)

class TestPrettyDictStr(unittest.TestCase):
    def test_pretty_dict_str(self):
        dictionary = {"key2": "value2", "key1": "value1"}
        pretty_str = pretty_dict_str(dictionary)
        expected_str = json.dumps(dictionary, indent=4, sort_keys=True)
        self.assertEqual(pretty_str, expected_str)

    def test_pretty_dict_str_empty(self):
        dictionary = {}
        pretty_str = pretty_dict_str(dictionary)
        expected_str = json.dumps(dictionary, indent=4, sort_keys=True)
        self.assertEqual(pretty_str, expected_str)

class TestBuildDirTree(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.sub_dir = self.test_dir / 'sub_dir'
        self.sub_dir.mkdir()
        (self.sub_dir / 'file.txt').touch()
        self.tree = Tree()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_build_dir_tree(self):
        tree = build_dir_tree(self.test_dir, self.tree)
        self.assertEqual(len(tree.all_nodes()), len(self.tree.all_nodes()))
        self.assertTrue(any(node.tag == 'sub_dir' for node in tree.all_nodes()))
        self.assertTrue(any(node.tag == 'file.txt' for node in tree.all_nodes()))

class TestCleanStr(unittest.TestCase):
    def test_clean_str_no_pattern(self):
        string = "Hello, World!"
        result = clean_str(string, None)
        self.assertEqual(result, string)

    def test_clean_str_with_pattern(self):
        string = "Hello, World!"
        pattern = ","
        result = clean_str(string, pattern)
        self.assertEqual(result, "Hello World!")

    def test_clean_str_with_pattern_and_sub_char(self):
        string = "Hello, World!"
        pattern = ","
        sub_char = ";"
        result = clean_str(string, pattern, sub_char)
        self.assertEqual(result, "Hello; World!")


if __name__ == '__main__':
    unittest.main()