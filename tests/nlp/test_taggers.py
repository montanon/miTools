import re
import unittest
from typing import Sequence, Tuple
from unittest import TestCase

import nltk

from mitools.nlp.taggers import BaseTagger, NLTKTagger, PatternTagger
from mitools.nlp.tokenizers import BaseTokenizer

nltk.download("punkt", download_dir="/Users/sebastian/nltk_data")
nltk.data.path.append("/Users/sebastian/nltk_data")


class MockTagger(BaseTagger):
    def tag_tokens(self, text: str, tokenize: bool = True) -> Sequence[Tuple[str, str]]:
        if tokenize:
            words = text.split()
        else:
            words = [text]
        return [(w, "MOCK") for w in words]


class TestBaseTagger(TestCase):
    def test_abstract(self):
        with self.assertRaises(TypeError):
            BaseTagger()

    def test_mock_tagger(self):
        tagger = MockTagger()
        result = tagger.tag_tokens("This is a test")
        self.assertEqual(
            result, [("This", "MOCK"), ("is", "MOCK"), ("a", "MOCK"), ("test", "MOCK")]
        )
        result_no_tokenize = tagger.tag_tokens("This is a test", tokenize=False)
        self.assertEqual(result_no_tokenize, [("This is a test", "MOCK")])


class TestPatternTagger(TestCase):
    def setUp(self):
        self.tagger = PatternTagger()

    def test_tag_with_tokenize(self):
        text = "This is a simple test."
        result = self.tagger.tag_tokens(text, tokenize=True)
        self.assertIsInstance(result, (list, tuple))
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in result))
        words, tags = [t[0] for t in result], [t[1] for t in result]
        self.assertEqual(words, ["This", "is", "a", "simple", "test", "."])
        self.assertEqual(tags, ["DT", "VBZ", "DT", "JJ", "NN", "."])

    def test_tag_without_tokenize(self):
        text = "This is a simple test."
        result = self.tagger.tag_tokens(text, tokenize=False)
        self.assertIsInstance(result, (list, tuple))
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in result))
        words, tags = [t[0] for t in result], [t[1] for t in result]
        self.assertEqual(words, ["This", "is", "a", "simple", "test."])
        self.assertEqual(tags, ["DT", "VBZ", "DT", "JJ", "NN"])

    def test_non_string_input(self):
        class Mockstr:
            def __init__(self, raw):
                self.raw = raw

        mock_text = Mockstr("Another test")
        result = self.tagger.tag_tokens(mock_text, tokenize=True)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in result))


class TestNLTKTagger(TestCase):
    def setUp(self):
        self.tagger = NLTKTagger()

    def test_singleton(self):
        t2 = NLTKTagger()
        self.assertIs(self.tagger, t2)

    def test_tag_default_tokenizer(self):
        text = "This is a simple test."
        result = self.tagger.tag_tokens(text)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in result))
        words = [t[0] for t in result]
        for w in ["This", "is", "a", "simple", "test"]:
            self.assertIn(w, words)

    def test_tag_with_custom_tokenizer(self):
        class CustomTokenizer(BaseTokenizer):
            def tokenize(self, text: str):
                return text.replace("-", " ").split()

        tokenizer = CustomTokenizer()
        text = "A hyphen-separated-word"
        result = self.tagger.tag_tokens(text, tokenizer=tokenizer)
        words = [t[0] for t in result]
        self.assertEqual(words, ["A", "hyphen", "separated", "word"])

    def test_empty_input(self):
        result = self.tagger.tag_tokens("")
        self.assertEqual(result, [])

    def test_non_string_input(self):
        class Mockstr:
            def __init__(self, raw):
                self.raw = raw

        text = Mockstr("Testing non-string input")
        result = self.tagger.tag_tokens(text)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in result))


if __name__ == "__main__":
    unittest.main()
