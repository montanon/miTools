import unittest
from typing import List, Optional

import numpy as np
import pandas as pd

from mitools.utils import BitArray, remove_chars


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

if __name__ == '__main__':
    unittest.main()