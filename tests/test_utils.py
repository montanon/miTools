import sys
import unittest
from typing import List, Optional
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd

from mitools.utils import (BitArray, dict_from_kwargs, display_env_variables,
                           find_str_line_number_in_text, get_numbers_from_str,
                           iterable_chunks, lcs_similarity, read_text_file,
                           remove_chars, remove_multiple_spaces, str_is_number,
                           stretch_string)


class TestIterableChunks(unittest.TestCase):
    def test_list_input(self):
        iterable = [1, 2, 3, 4, 5, 6]
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, [[1, 2], [3, 4], [5, 6]])

    def test_string_input(self):
        iterable = "123456"
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, ["12", "34", "56"])

    def test_tuple_input(self):
        iterable = (1, 2, 3, 4, 5, 6)
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, [(1, 2), (3, 4), (5, 6)])

    def test_bytes_input(self):
        iterable = b"123456"
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, [b"12", b"34", b"56"])

    def test_invalid_input(self):
        iterable = set([1, 2, 3, 4, 5, 6])
        chunk_size = 2
        with self.assertRaises(TypeError):
            list(iterable_chunks(iterable, chunk_size))

class TestStrIsNumber(unittest.TestCase):
    def test_integer_string(self):
        self.assertTrue(str_is_number('123'))

    def test_float_string(self):
        self.assertTrue(str_is_number('123.456'))

    def test_negative_integer_string(self):
        self.assertTrue(str_is_number('-123'))

    def test_negative_float_string(self):
        self.assertTrue(str_is_number('-123.456'))

    def test_non_numeric_string(self):
        self.assertFalse(str_is_number('abc'))

    def test_empty_string(self):
        self.assertFalse(str_is_number(''))

class TestGetNumbersFromStr(unittest.TestCase):
    def test_integer_string(self):
        string = 'abc 123 def 456'
        self.assertEqual(get_numbers_from_str(string), [123.0, 456.0])

    def test_float_string(self):
        string = 'abc 123.456 def 789.012'
        self.assertEqual(get_numbers_from_str(string), [123.456, 789.012])

    def test_negative_number_string(self):
        string = 'abc -123 def -456'
        self.assertEqual(get_numbers_from_str(string), [-123.0, -456.0])

    def test_non_numeric_string(self):
        string = 'abc def'
        self.assertEqual(get_numbers_from_str(string), [])

    def test_empty_string(self):
        string = ''
        self.assertEqual(get_numbers_from_str(string), [])

    def test_indexed_return(self):
        string = 'abc 123 def 456'
        self.assertEqual(get_numbers_from_str(string, 1), 456.0)

class TestRemoveMultipleSpaces(unittest.TestCase):
    def test_multiple_spaces(self):
        string = 'abc   def   ghi'
        self.assertEqual(remove_multiple_spaces(string), 'abc def ghi')

    def test_tabs(self):
        string = 'abc\t\t\tdef\t\t\tghi'
        self.assertEqual(remove_multiple_spaces(string), 'abc def ghi')

    def test_newlines(self):
        string = 'abc\n\ndef\n\nghi'
        self.assertEqual(remove_multiple_spaces(string), 'abc def ghi')

    def test_mixed_whitespace(self):
        string = 'abc \t \n def \t \n ghi'
        self.assertEqual(remove_multiple_spaces(string), 'abc def ghi')

    def test_no_extra_spaces(self):
        string = 'abc def ghi'
        self.assertEqual(remove_multiple_spaces(string), 'abc def ghi')

class TestReadTextFile(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data='abc\ndef\nghi')
    def test_read_text_file(self, mock_file):
        text_path = 'dummy_path'
        result = read_text_file(text_path)
        self.assertEqual(result, 'abc\ndef\nghi')
        mock_file.assert_called_once_with(text_path, 'r')

class TestDictFromKwargs(unittest.TestCase):
    def test_no_arguments(self):
        self.assertEqual(dict_from_kwargs(), {})

    def test_one_argument(self):
        self.assertEqual(dict_from_kwargs(a=1), {'a': 1})

    def test_multiple_arguments(self):
        self.assertEqual(dict_from_kwargs(a=1, b=2, c=3), {'a': 1, 'b': 2, 'c': 3})

class TestLcsSimilarity(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(lcs_similarity('abc', 'abc'), 1.0)

    def test_different_strings(self):
        self.assertEqual(lcs_similarity('abc', 'def'), 0.0)

    def test_common_subsequence(self):
        self.assertEqual(lcs_similarity('abc', 'adc'), 2/3)

    def test_empty_string(self):
        self.assertEqual(lcs_similarity('abc', ''), 0.0)
        self.assertEqual(lcs_similarity('', 'abc'), 0.0)
        self.assertEqual(lcs_similarity('', ''), 0.0)


class TestFindStrLineNumberInText(unittest.TestCase):
    def test_substring_at_start(self):
        text = 'abc\ndef\nghi'
        substring = 'abc'
        self.assertEqual(find_str_line_number_in_text(text, substring), 0)

    def test_substring_in_middle(self):
        text = 'abc\ndef\nghi'
        substring = 'def'
        self.assertEqual(find_str_line_number_in_text(text, substring), 1)

    def test_substring_at_end(self):
        text = 'abc\ndef\nghi'
        substring = 'ghi'
        self.assertEqual(find_str_line_number_in_text(text, substring), 2)

    def test_substring_not_found(self):
        text = 'abc\ndef\nghi'
        substring = 'jkl'
        self.assertEqual(find_str_line_number_in_text(text, substring), None)


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

class TestBitArray(unittest.TestCase):

    def test_zeros_initialization(self):
        size = 16
        bit_array = BitArray.zeros(size)
        for i in range(size):
            self.assertEqual(bit_array.get_index(i), 0)

    def test_set_and_get_bits(self):
        bit_array = BitArray.zeros(16)
        indices_to_set = [1, 3, 15]
        for index in indices_to_set:
            bit_array[index] = 1
            self.assertEqual(bit_array.get_index(index), 1)

    def test_out_of_range_index(self):
        bit_array = BitArray.zeros(10)
        with self.assertRaises(IndexError):
            bit_array.get_index(10)
        with self.assertRaises(IndexError):
            bit_array[10] = 1

    def test_length(self):
        size = 20
        bit_array = BitArray.zeros(size)
        self.assertEqual(len(bit_array), size)

    def test_repr(self):
        bit_array = BitArray.zeros(8)
        expected_repr = "BitArray([0, 0, 0, 0, 0, 0, 0, 0])"
        self.assertEqual(repr(bit_array), expected_repr)

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

if __name__ == '__main__':
    unittest.main()
