import sys
import unittest
from typing import List, Optional

import numpy as np
import pandas as pd

from mitools.utils import (BitArray, display_env_variables, remove_chars,
                           stretch_string)


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

    def test_different_data_types(self):
        threshold_mb = 0
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertIn('large_list', df['Variable'].values)
        self.assertIn('large_dict', df['Variable'].values)

    def test_edge_cases(self):
        # Empty env_vars
        df_empty = display_env_variables([], 0)
        self.assertTrue(df_empty.empty)

        # Extremely high threshold
        df_high_threshold = display_env_variables(self.env_vars, 1000000)
        self.assertTrue(df_high_threshold.empty)


if __name__ == '__main__':
    unittest.main()