import unittest
from unittest import TestCase
from unittest.mock import patch

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

from mitools.nlp.blobs import Word, WordList


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


class TestWordList(TestCase):
    def setUp(self):
        self.words = WordList(["dogs", "cats", "birds"])

    def test_initialization(self):
        self.assertEqual(len(self.words), 3)
        self.assertIsInstance(self.words[0], Word)
        self.assertEqual(self.words[0].string, "dogs")

    def test_str_and_repr(self):
        self.assertEqual(str(self.words), "['dogs', 'cats', 'birds']")
        self.assertEqual(repr(self.words), "WordList(['dogs', 'cats', 'birds'])")

    def test_getitem_single(self):
        word = self.words[1]
        self.assertIsInstance(word, Word)
        self.assertEqual(word.string, "cats")

    def test_getitem_slice(self):
        sliced = self.words[:2]
        self.assertIsInstance(sliced, WordList)
        self.assertEqual(len(sliced), 2)
        self.assertEqual(sliced[0].string, "dogs")

    def test_setitem_with_string(self):
        self.words[0] = "foxes"
        self.assertIsInstance(self.words[0], Word)
        self.assertEqual(self.words[0].string, "foxes")

    def test_setitem_with_word(self):
        new_word = Word("wolves")
        self.words[1] = new_word
        self.assertIsInstance(self.words[1], Word)
        self.assertEqual(self.words[1].string, "wolves")

    def test_count_case_insensitive(self):
        word_list = WordList(["Dogs", "DOGS", "cats", "birds"])
        self.assertEqual(word_list.count("dogs"), 2)

    def test_count_case_sensitive(self):
        word_list = WordList(["Dogs", "DOGS", "cats", "birds"])
        self.assertEqual(word_list.count("dogs", case_sensitive=True), 0)

    def test_append_with_string(self):
        self.words.append("elephants")
        self.assertEqual(self.words[-1].string, "elephants")

    def test_append_with_word(self):
        self.words.append(Word("foxes"))
        self.assertEqual(self.words[-1].string, "foxes")

    def test_extend_with_strings(self):
        self.words.extend(["elephants", "rabbits"])
        self.assertEqual(self.words[-1].string, "rabbits")
        self.assertEqual(len(self.words), 5)

    def test_extend_with_words(self):
        self.words.extend([Word("lions"), Word("tigers")])
        self.assertEqual(self.words[-1].string, "tigers")
        self.assertEqual(len(self.words), 5)

    def test_upper(self):
        upper_words = self.words.upper()
        self.assertIsInstance(upper_words, WordList)
        self.assertEqual(upper_words[0].string, "DOGS")

    def test_lower(self):
        lower_words = self.words.lower()
        self.assertIsInstance(lower_words, WordList)
        self.assertEqual(lower_words[0].string, "dogs")

    def test_singularize(self):
        with patch.object(Word, "singularize", side_effect=lambda: Word("dog")):
            singular_words = self.words.singularize()
            self.assertIsInstance(singular_words, WordList)
            self.assertEqual(singular_words[0].string, "dog")

    def test_pluralize(self):
        with patch.object(Word, "pluralize", side_effect=lambda: Word("dogs")):
            plural_words = self.words.pluralize()
            self.assertIsInstance(plural_words, WordList)
            self.assertEqual(plural_words[0].string, "dogs")

    def test_lemmatize(self):
        with patch.object(Word, "lemmatize", side_effect=lambda: Word("dog")):
            lemmatized_words = self.words.lemmatize()
            self.assertIsInstance(lemmatized_words, WordList)
            self.assertEqual(lemmatized_words[0].string, "dog")

    def test_stem(self):
        with patch.object(Word, "stem", side_effect=lambda: Word("dog")):
            stemmed_words = self.words.stem()
            self.assertIsInstance(stemmed_words, WordList)
            self.assertEqual(stemmed_words[0].string, "dog")

    def test_title(self):
        with patch.object(Word, "title", side_effect=lambda: Word("Dogs")):
            titled_words = self.words.title()
            self.assertIsInstance(titled_words, WordList)
            self.assertEqual(titled_words[0].string, "Dogs")


if __name__ == "__main__":
    unittest.main()
