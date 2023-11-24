import unittest
from typing import List, Optional

import numpy as np
import pandas as pd

from mitools.utils import remove_chars, stretch_string


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


if __name__ == '__main__':
    unittest.main()