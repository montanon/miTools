import re
import unittest
from typing import Sequence
from unittest import TestCase

import nltk

from mitools.nlp.nlp_typing import BaseString
from mitools.nlp.tokenizers import (
    BaseTokenizer,
    BlanklineTokenizer,
    RegexpTokenizer,
    SentenceTokenizer,
    WhiteSpaceTokenizer,
    WordPunctTokenizer,
    WordTokenizer,
)


class TestBaseTokenizer(TestCase):
    class MockTokenizer(BaseTokenizer):
        def tokenize(self, text: BaseString) -> Sequence[BaseString]:
            return text.split()

    def setUp(self):
        self.tokenizer = self.MockTokenizer()

    def test_tokenize(self):
        self.assertEqual(self.tokenizer.tokenize("Hello world"), ["Hello", "world"])

    def test_itokenize(self):
        result = list(self.tokenizer.itokenize("Hello world"))
        self.assertEqual(result, ["Hello", "world"])

    def test_span_tokenize(self):
        with self.assertRaises(NotImplementedError):
            list(self.tokenizer.span_tokenize("Hello world"))

    def test_tokenize_sents(self):
        texts = ["Hello world", "This is a test"]
        result = self.tokenizer.tokenize_sents(texts)
        self.assertEqual(result, [["Hello", "world"], ["This", "is", "a", "test"]])


class TestWordTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_singleton(self):
        t2 = WordTokenizer()
        self.assertIs(self.tokenizer, t2)

    def test_tokenize_with_punctuation(self):
        text = "Hello, world! This is a test."
        tokens = self.tokenizer.tokenize(text, include_punctuation=True)
        self.assertIn(",", tokens)
        self.assertIn("!", tokens)
        self.assertIn(".", tokens)

    def test_tokenize_without_punctuation(self):
        text = "Hello, world! This is a test."
        tokens = self.tokenizer.tokenize(text, include_punctuation=False)
        self.assertNotIn(",", tokens)
        self.assertNotIn("!", tokens)
        self.assertNotIn(".", tokens)
        self.assertEqual(tokens, ["Hello", "world", "This", "is", "a", "test"])


class TestSentenceTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = SentenceTokenizer()

    def test_singleton(self):
        t2 = SentenceTokenizer()
        self.assertIs(self.tokenizer, t2)

    def test_tokenize_sentences(self):
        text = "Hello world. This is a test. Another sentence!"
        sents = self.tokenizer.tokenize(text)
        self.assertEqual(len(sents), 3)
        self.assertEqual(sents[0], "Hello world.")
        self.assertEqual(sents[1], "This is a test.")
        self.assertEqual(sents[2], "Another sentence!")

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])


class TestRegexpTokenizer(TestCase):
    def test_singleton_varied_parameters(self):
        t1 = RegexpTokenizer(pattern=r"\w+")
        t2 = RegexpTokenizer(pattern=r"\w+", gaps=True)
        t3 = RegexpTokenizer(pattern=r"\w+", gaps=True, discard_empty=False)
        t4 = RegexpTokenizer(pattern=r"\s+", gaps=True)
        self.assertIsNot(t1, t2)
        self.assertIsNot(t2, t3)
        self.assertIsNot(t3, t4)
        t5 = RegexpTokenizer(pattern=r"\w+")
        self.assertIs(t1, t5)

    def test_tokenize_with_pattern(self):
        tokenizer = RegexpTokenizer(pattern=r"\w+")
        text = "Hello, world! 123"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Hello", "world", "123"])

    def test_tokenize_gaps(self):
        tokenizer = RegexpTokenizer(pattern=r"\s+", gaps=True)
        text = "Hello   world   test"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Hello", "world", "test"])

    def test_discard_empty_false(self):
        tokenizer = RegexpTokenizer(pattern=r":", gaps=True, discard_empty=False)
        text = "word::test"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["word", "", "test"])

    def test_flags(self):
        tokenizer = RegexpTokenizer(
            pattern=r".+", gaps=False, discard_empty=False, flags=re.DOTALL
        )
        text = "Line1\nLine2\nLine3"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Line1\nLine2\nLine3"])


class TestWhiteSpaceTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = WhiteSpaceTokenizer()

    def test_singleton(self):
        t2 = WhiteSpaceTokenizer()
        self.assertIs(self.tokenizer, t2)

    def test_tokenize(self):
        text = "Hello   world \n test"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["Hello", "world", "test"])

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])


class TestWordPunctTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = WordPunctTokenizer()

    def test_singleton(self):
        t2 = WordPunctTokenizer()
        self.assertIs(self.tokenizer, t2)

    def test_tokenize(self):
        text = "Hello, world! This is a test."
        tokens = self.tokenizer.tokenize(text)
        self.assertIn("Hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("!", tokens)
        self.assertIn(".", tokens)


class TestBlanklineTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = BlanklineTokenizer()

    def test_singleton(self):
        t2 = BlanklineTokenizer()
        self.assertIs(self.tokenizer, t2)

    def test_tokenize(self):
        text = "Hello world\n\nThis is a test\n\nAnother paragraph\n"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(
            tokens, ["Hello world", "This is a test", "Another paragraph\n"]
        )

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])


if __name__ == "__main__":
    unittest.main()
