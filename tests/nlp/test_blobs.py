import unittest
from typing import Literal
from unittest import TestCase
from unittest.mock import patch

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

from mitools.nlp.blobs import Word, WordList
from mitools.nlp.nlp_typing import PosTag


def singularize(word, language):
    return word.rstrip("s") if language == "en" else word


def pluralize(word, language):
    return word + "s" if language == "en" else word


def suggest(word):
    return [(word, 1.0)]  # Mocked to return the same word


def translate_tag(pos, from_tagset, to_tagset):
    return wordnet.NOUN  # Mocked to return NOUN for simplicity


class TestWord(TestCase):
    def setUp(self):
        self.word = Word("dogs", pos_tag=wordnet.NOUN)

    def test_initialization(self):
        self.assertEqual(str(self.word), "dogs")
        self.assertEqual(self.word.string, "dogs")
        self.assertEqual(self.word.pos_tag, wordnet.NOUN)

    def test_singularize(self):
        singular_word = self.word.singularize(language="en")
        self.assertEqual(singular_word.string, "dog")

    def test_pluralize(self):
        plural_word = self.word.pluralize(language="en")
        self.assertEqual(plural_word.string, "dogss")  # Original input has "dogs"

    def test_spellcheck(self):
        with self.assertRaises(NotImplementedError):
            self.word.spellcheck()

    def test_correct(self):
        with self.assertRaises(NotImplementedError):
            self.word.correct()

    def test_cached_property_lemma(self):
        with patch.object(
            WordNetLemmatizer, "lemmatize", return_value="dog"
        ) as mock_lemmatize:
            self.assertEqual(self.word.lemma, "dog")
            mock_lemmatize.assert_called_once_with("dogs", wordnet.NOUN)

    def test_lemmatize(self):
        with patch.object(
            WordNetLemmatizer, "lemmatize", return_value="dog"
        ) as mock_lemmatize:
            lemma = self.word.lemmatize(pos=wordnet.VERB)
            self.assertEqual(lemma, "dog")
            mock_lemmatize.assert_called_once_with("dogs", wordnet.VERB)

    def test_stem(self):
        with patch.object(PorterStemmer, "stem", return_value="dog") as mock_stem:
            stem = self.word.stem()
            self.assertEqual(stem, "dog")
            mock_stem.assert_called_once_with("dogs")

    def test_cached_property_synsets(self):
        with patch.object(
            wordnet, "synsets", return_value=["mock_synset"]
        ) as mock_synsets:
            synsets = self.word.synsets
            self.assertEqual(synsets, ["mock_synset"])
            mock_synsets.assert_called_once_with("dogs", None)

    def test_cached_property_definitions(self):
        with patch.object(
            wordnet,
            "synsets",
            return_value=[
                type("MockSynset", (), {"definition": lambda x: "mock definition"})()
            ],
        ) as mock_synsets:
            definitions = self.word.definitions
            self.assertEqual(definitions, ["mock definition"])
            mock_synsets.assert_called_once_with("dogs", None)

    def test_get_synsets(self):
        with patch.object(
            wordnet, "synsets", return_value=["mock_synset"]
        ) as mock_synsets:
            synsets = self.word.get_synsets(pos_tag=wordnet.NOUN)
            self.assertEqual(synsets, ["mock_synset"])
            mock_synsets.assert_called_once_with("dogs", wordnet.NOUN)

    def test_define(self):
        with patch.object(
            wordnet,
            "synsets",
            return_value=[
                type("MockSynset", (), {"definition": lambda x: "mock definition"})()
            ],
        ) as mock_synsets:
            definitions = self.word.define(pos_tag=wordnet.NOUN)
            self.assertEqual(definitions, ["mock definition"])
            mock_synsets.assert_called_once_with("dogs", wordnet.NOUN)


if __name__ == "__main__":
    unittest.main()
