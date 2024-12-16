import json
import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

from mitools.exceptions import ArgumentTypeError
from mitools.nlp.blobs import BaseBlob, Sentence, TextBlob, Word, WordList
from mitools.nlp.tokenizers import WordTokenizer


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


class TestBaseBlob(TestCase):
    def setUp(self):
        self.blob = BaseBlob("The quick brown fox jumps over the lazy dog.")

    def test_initialization(self):
        self.assertEqual(self.blob.raw, "The quick brown fox jumps over the lazy dog.")
        self.assertEqual(
            self.blob.string, "The quick brown fox jumps over the lazy dog."
        )
        self.assertEqual(
            self.blob.stripped, "the quick brown fox jumps over the lazy dog"
        )
        self.assertIsInstance(self.blob.tokenizer, WordTokenizer)

    def test_invalid_initialization(self):
        with self.assertRaises(ArgumentTypeError):
            BaseBlob(12345)  # Non-string input

    def test_words_property(self):
        with patch(
            "mitools.nlp.utils.word_tokenize",
            return_value=["The", "quick", "brown", "fox"],
        ):
            words = self.blob.words
            self.assertIsInstance(words, WordList)
            self.assertEqual(words[0].string, "The")

    def test_tokens_property(self):
        with patch.object(
            self.blob.tokenizer,
            "tokenize",
            return_value=["The", "quick", "brown", "fox"],
        ):
            tokens = self.blob.tokens
            self.assertIsInstance(tokens, WordList)
            self.assertEqual(tokens[0].string, "The")

    def test_tokenize_method(self):
        with patch.object(
            self.blob.tokenizer, "tokenize", return_value=["quick", "brown", "fox"]
        ):
            tokens = self.blob.tokenize()
            self.assertIsInstance(tokens, WordList)
            self.assertEqual(tokens[0].string, "quick")

    def test_parse_method(self):
        mock_parser = MagicMock()
        mock_parser.parse.return_value = "parsed string"
        result = self.blob.parse(parser=mock_parser)
        self.assertEqual(result, "parsed string")
        mock_parser.parse.assert_called_once_with(self.blob.raw)

    def test_classify_no_classifier(self):
        with self.assertRaises(NameError):
            self.blob.classify()

    def test_sentiment_property(self):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "positive sentiment"
        self.blob.analyzer = mock_analyzer
        sentiment = self.blob.sentiment
        self.assertEqual(sentiment, "positive sentiment")
        mock_analyzer.analyze.assert_called_once_with(self.blob.raw)

    def test_sentiment_assessments_property(self):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = "detailed sentiment"
        self.blob.analyzer = mock_analyzer
        assessments = self.blob.sentiment_assessments
        self.assertEqual(assessments, "detailed sentiment")
        mock_analyzer.analyze.assert_called_once_with(
            self.blob.raw, keep_assessments=True
        )

    def test_ngram_generation(self):
        ngrams = self.blob.ngrams(2)
        self.assertEqual(len(ngrams), 8)
        self.assertIsInstance(ngrams[0], WordList)
        self.assertEqual(ngrams[0][0].string, "The")
        ngrams = self.blob.ngrams(1)
        self.assertEqual(len(ngrams), 9)
        self.assertIsInstance(ngrams[0], WordList)
        self.assertEqual(ngrams[0][0].string, "The")

    def test_ngrams_invalid_n(self):
        ngrams = self.blob.ngrams(-1)
        self.assertEqual(ngrams, [])

    def test_noun_phrases_property(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ["quick brown", "lazy dog"]
        self.blob.np_extractor = mock_extractor
        noun_phrases = self.blob.noun_phrases
        self.assertIsInstance(noun_phrases, WordList)
        self.assertEqual(noun_phrases[0].string, "quick brown")

    def test_pos_tags_property(self):
        mock_tagger = MagicMock()
        mock_tagger.tag_tokens.return_value = [("quick", "JJ"), ("brown", "NN")]
        self.blob.pos_tagger = mock_tagger
        pos_tags = self.blob.pos_tags
        self.assertEqual(pos_tags[0][0].string, "quick")
        self.assertEqual(pos_tags[0][1], "JJ")

    def test_word_counts_property(self):
        word_counts = self.blob.word_counts
        self.assertEqual(word_counts["quick"], 1)
        self.assertEqual(word_counts["the"], 2)

    def test_np_counts_property(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ["quick brown", "lazy dog", "lazy dog"]
        self.blob.np_extractor = mock_extractor
        np_counts = self.blob.np_counts
        self.assertEqual(np_counts["lazy dog"], 2)

    def test_correct_method(self):
        with self.assertRaises(NotImplementedError):
            self.blob.correct()

    def test_comparable_key(self):
        self.assertEqual(
            self.blob.comparable_key(), "The quick brown fox jumps over the lazy dog."
        )

    def test_string_key(self):
        self.assertEqual(
            self.blob.string_key(), "The quick brown fox jumps over the lazy dog."
        )

    def test_addition_with_string(self):
        new_blob = self.blob + " New sentence."
        self.assertIsInstance(new_blob, BaseBlob)
        self.assertEqual(
            new_blob.raw, "The quick brown fox jumps over the lazy dog. New sentence."
        )

    def test_addition_with_blob(self):
        other_blob = BaseBlob(" Another blob.")
        new_blob = self.blob + other_blob
        self.assertIsInstance(new_blob, BaseBlob)
        self.assertEqual(
            new_blob.raw, "The quick brown fox jumps over the lazy dog. Another blob."
        )

    def test_addition_invalid_type(self):
        with self.assertRaises(TypeError):
            _ = self.blob + 123

    def test_split_method(self):
        split_result = self.blob.split()
        self.assertIsInstance(split_result, WordList)
        self.assertEqual(split_result[0].string, "The")


class TestTextBlob(TestCase):
    def setUp(self):
        self.text = "The quick brown fox. Jumps over the lazy dog."
        self.blob = TextBlob(self.text)

    def test_initialization(self):
        self.assertEqual(self.blob.raw, self.text)
        self.assertIsInstance(self.blob, TextBlob)

    def test_sentences_property(self):
        sentences = self.blob.sentences
        self.assertEqual(len(sentences), 2)
        self.assertIsInstance(sentences[0], Sentence)
        self.assertEqual(sentences[0].raw, "The quick brown fox.")

    def test_words_property(self):
        words = self.blob.words
        self.assertIsInstance(words, WordList)
        self.assertEqual(len(words), 9)
        self.assertEqual(words[0].string, "The")

    def test_raw_sentences_property(self):
        raw_sentences = self.blob.raw_sentences
        self.assertEqual(len(raw_sentences), 2)
        self.assertEqual(raw_sentences[0], "The quick brown fox.")

    def test_serialized_property(self):
        serialized = self.blob.serialized
        self.assertEqual(len(serialized), 2)
        self.assertEqual(serialized[0]["raw"], "The quick brown fox.")

    def test_to_json(self):
        mock_serialized = [
            {
                "end_index": 20,
                "noun_phrases": ["quick brown fox"],
                "polarity": 0.7793781757354736,
                "raw": "The quick brown fox.",
                "start_index": 0,
                "stripped": "the quick brown fox",
                "subjectivity": None,
            },
            {
                "end_index": 45,
                "noun_phrases": ["jumps", "lazy dog"],
                "polarity": -0.8721502423286438,
                "raw": "Jumps over the lazy dog.",
                "start_index": 21,
                "stripped": "jumps over the lazy dog",
                "subjectivity": None,
            },
        ]
        json_output = self.blob.to_json()
        self.assertEqual(json.loads(json_output), mock_serialized)

    def test_json_property(self):
        mock_serialized = [
            {
                "end_index": 20,
                "noun_phrases": ["quick brown fox"],
                "polarity": 0.7793781757354736,
                "raw": "The quick brown fox.",
                "start_index": 0,
                "stripped": "the quick brown fox",
                "subjectivity": None,
            },
            {
                "end_index": 45,
                "noun_phrases": ["jumps", "lazy dog"],
                "polarity": -0.8721502423286438,
                "raw": "Jumps over the lazy dog.",
                "start_index": 21,
                "stripped": "jumps over the lazy dog",
                "subjectivity": None,
            },
        ]
        self.assertEqual(json.loads(self.blob.json), mock_serialized)


class TestSentence(TestCase):
    def setUp(self):
        self.sentence_text = "The quick brown fox."
        self.start_index = 0
        self.end_index = len(self.sentence_text)
        self.sentence = Sentence(
            self.sentence_text,
            start_index=self.start_index,
            end_index=self.end_index,
        )

    def test_initialization(self):
        self.assertEqual(self.sentence.raw, self.sentence_text)
        self.assertEqual(self.sentence.start_index, self.start_index)
        self.assertEqual(self.sentence.end_index, self.end_index)
        self.assertIsInstance(self.sentence, Sentence)

    def test_dict_property(self):
        result = self.sentence.dict
        self.assertEqual(result["raw"], self.sentence_text)
        self.assertEqual(result["start_index"], self.start_index)
        self.assertEqual(result["end_index"], self.end_index)
        self.assertEqual(result["noun_phrases"][0].string, "quick brown fox")
        self.assertEqual(result["polarity"], 0.7793781757354736)
        self.assertEqual(result["subjectivity"], None)


if __name__ == "__main__":
    unittest.main()
