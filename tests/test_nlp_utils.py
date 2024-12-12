import unittest
from unittest import TestCase

from mitools.nlp.nlp_typing import BaseString
from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.nlp.utils import sentence_tokenize, word_tokenize
from mitools.utils.helper_functions import strip_punctuation


class TestWordTokenize(TestCase):
    def test_basic_functionality(self):
        text = "Hello world!"
        tokens = list(word_tokenize(text))
        self.assertEqual(
            tokens,
            ["Hello", "world", "!"],
            "Should tokenize a basic sentence correctly.",
        )

    def test_multiple_sentences(self):
        text = "Hello world! How are you today?"
        tokens = list(word_tokenize(text))
        self.assertEqual(
            tokens,
            ["Hello", "world", "!", "How", "are", "you", "today", "?"],
            "Should tokenize multiple sentences correctly.",
        )

    def test_include_punctuation_true(self):
        text = "Hello, world!"
        tokens = list(word_tokenize(text, include_punctuation=True))
        self.assertIn(
            ",", tokens, "Should include punctuation when include_punctuation=True."
        )
        self.assertIn(
            "!", tokens, "Should include punctuation when include_punctuation=True."
        )
        self.assertEqual(
            tokens,
            ["Hello", ",", "world", "!"],
            "Should include punctuation when include_punctuation=True.",
        )

    def test_include_punctuation_false(self):
        text = "Hello, world!"
        tokens = list(word_tokenize(text, include_punctuation=False))
        self.assertEqual(
            tokens,
            ["Hello", "world"],
            "Should remove punctuation when include_punctuation=False.",
        )

    def test_empty_input(self):
        text = ""
        tokens = list(word_tokenize(text))
        self.assertEqual(tokens, [], "Should return an empty list for empty input.")

    def test_whitespace_only_input(self):
        text = "   \n\n   "
        tokens = list(word_tokenize(text))
        self.assertEqual(
            tokens, [], "Should return an empty list for whitespace-only input."
        )

    def test_special_characters(self):
        text = "Café costs $5.00, okay?"
        tokens_with_punct = list(word_tokenize(text, include_punctuation=True))
        tokens_no_punct = list(word_tokenize(text, include_punctuation=False))
        self.assertIn(
            "$",
            tokens_with_punct,
            "Should include special character '$' with punctuation.",
        )
        self.assertIn(
            ",", tokens_with_punct, "Should include punctuation ',' with punctuation."
        )
        self.assertIn("?", tokens_with_punct, "Should include '?' with punctuation.")
        self.assertNotIn(
            ",", tokens_no_punct, "Should remove punctuation without punctuation flag."
        )
        self.assertNotIn(
            "?", tokens_no_punct, "Should remove punctuation without punctuation flag."
        )
        self.assertNotIn(
            "$", tokens_no_punct, "Should remove punctuation without punctuation flag."
        )
        self.assertIn("Café", tokens_no_punct, "Should keep accented letters.")
        self.assertIn(
            "5.00",
            tokens_no_punct,
            "Should keep numeric tokens with periods inside them.",
        )
        self.assertEqual(
            tokens_no_punct,
            ["Café", "costs", "5.00", "okay"],
            "Should remove punctuation without punctuation flag.",
        )
        self.assertEqual(
            tokens_with_punct,
            ["Café", "costs", "$", "5.00", ",", "okay", "?"],
            "Should include punctuation when include_punctuation=True.",
        )

    def test_custom_word_tokenizer(self):
        class MockWordTokenizer(WordTokenizer):
            def tokenize(self, text, include_punctuation=True):
                tokens = text.split()
                if not include_punctuation:
                    tokens = [
                        strip_punctuation(t) for t in tokens if strip_punctuation(t)
                    ]
                return tokens

        text = "Hello world!"
        mock_word_tokenizer = MockWordTokenizer()
        tokens = list(word_tokenize(text, word_tokenizer=mock_word_tokenizer))
        self.assertEqual(
            tokens, ["Hello", "world!"], "Should use provided custom word tokenizer."
        )

    def test_custom_sentence_tokenizer(self):
        class MockSentenceTokenizer(SentenceTokenizer):
            def tokenize(self, text):
                return text.split("||")

        text = "Hello world!||How are you?"
        mock_sentence_tokenizer = MockSentenceTokenizer()
        tokens = list(word_tokenize(text, sentence_tokenizer=mock_sentence_tokenizer))
        self.assertEqual(
            tokens,
            ["Hello", "world", "!", "How", "are", "you", "?"],
            "Should use provided custom sentence tokenizer.",
        )

    def test_non_english_text(self):
        text = "¡Hola! ¿Cómo estás?"
        tokens = list(word_tokenize(text, language="spanish"))
        self.assertIn("¡Hola", tokens, "Should handle non-English words correctly.")
        self.assertIn(
            "¿Cómo", tokens, "Should include non-English punctuation by default."
        )
        self.assertEqual(
            tokens,
            ["¡Hola", "!", "¿Cómo", "estás", "?"],
            "Should handle non-English words correctly.",
        )

    def test_numeric_tokens(self):
        text = "The price is 1000 dollars."
        tokens_with_punct = list(word_tokenize(text))
        tokens_no_punct = list(word_tokenize(text, include_punctuation=False))
        self.assertIn("1000", tokens_with_punct, "Should include numeric tokens.")
        self.assertNotIn(
            ".",
            tokens_no_punct,
            "Should remove trailing punctuation without punctuation flag.",
        )

    def test_punctuation_only_strings(self):
        text = "!!!"
        tokens_with_punct = list(word_tokenize(text, include_punctuation=True))
        tokens_no_punct = list(word_tokenize(text, include_punctuation=False))
        self.assertEqual(
            tokens_with_punct,
            ["!", "!", "!"],
            "Should keep punctuation if include_punctuation=True.",
        )
        self.assertEqual(
            tokens_no_punct,
            [],
            "Should remove punctuation-only tokens if include_punctuation=False.",
        )

    def test_return_type(self):
        text = "Hello world!"
        result = word_tokenize(text)
        self.assertTrue(
            hasattr(result, "__iter__"),
            "word_tokenize should return an iterator or chain.",
        )
        tokens = list(result)
        self.assertEqual(
            tokens, ["Hello", "world", "!"], "Iterator should yield expected tokens."
        )

    def test_unusual_spacing(self):
        text = "Hello     world!   "
        tokens = list(word_tokenize(text))
        self.assertEqual(
            tokens, ["Hello", "world", "!"], "Should handle multiple spaces gracefully."
        )


class TestSentenceTokenize(TestCase):
    def test_single_sentence(self):
        text = "Hello world!"
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences, ["Hello world!"], "Should return a single sentence."
        )

    def test_multiple_sentences(self):
        text = "Hello world! How are you today? I am fine."
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["Hello world!", "How are you today?", "I am fine."],
            "Should correctly split multiple sentences.",
        )

    def test_empty_string(self):
        text = ""
        sentences = list(sentence_tokenize(text))
        self.assertEqual(sentences, [], "Empty string should return no sentences.")

    def test_whitespace_only(self):
        text = "    \n\n    "
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences, [], "Whitespace-only string should return no sentences."
        )

    def test_no_punctuation(self):
        text = "Hello world I am here"
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["Hello world I am here"],
            "Text without punctuation should be treated as a single sentence.",
        )

    def test_unusual_spacing(self):
        text = "Hello world!   How are you?   "
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["Hello world!", "How are you?"],
            "Should handle extra spaces gracefully.",
        )

    def test_special_punctuation(self):
        text = "Hello world!!! How are you??? I'm fine..."
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["Hello world!!!", "How are you???", "I'm fine..."],
            "Should handle multiple punctuation marks.",
        )

    def test_non_english_text(self):
        text = "¡Hola! ¿Cómo estás? Estoy bien."
        sentences = list(sentence_tokenize(text, language="spanish"))
        self.assertEqual(
            sentences,
            ["¡Hola!", "¿Cómo estás?", "Estoy bien."],
            "Should handle non-English sentences reasonably well.",
        )

    def test_numeric_and_symbols(self):
        text = "Version 2.0 released. Check out www.example.com!"
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["Version 2.0 released.", "Check out www.example.com!"],
            "Should split on sentence punctuation even with numeric and symbolic text.",
        )

    def test_custom_sentence_tokenizer(self):
        class MockSentenceTokenizer(SentenceTokenizer):
            def tokenize(self, text):
                return text.split("||")

        text = "Hello world!||How are you?||I am fine."
        mock_sentence_tokenizer = MockSentenceTokenizer()
        sentences = list(
            sentence_tokenize(text, sentence_tokenizer=mock_sentence_tokenizer)
        )
        self.assertEqual(
            sentences,
            ["Hello world!", "How are you?", "I am fine."],
            "Should use provided custom sentence tokenizer logic.",
        )

    def test_return_type(self):
        text = "Hello world!"
        result = sentence_tokenize(text)
        self.assertTrue(
            hasattr(result, "__iter__"), "sentence_tokenize should return an iterator."
        )
        sentences = list(result)
        self.assertEqual(
            sentences, ["Hello world!"], "Iterator should yield the expected sentences."
        )

    def test_punctuation_only_string(self):
        text = "!!! ??? ..."
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["!", "!!", "?", "??", "..."],
            "Punctuation-only string should generally be one sentence (depending on tokenizer rules).",
        )

    def test_sentence_with_emojis(self):
        text = "Hello world! 😊 How are you? 🙃"
        sentences = list(sentence_tokenize(text))
        self.assertEqual(
            sentences,
            ["Hello world!", "😊 How are you?", "🙃"],
            "Should handle emojis within sentences correctly.",
        )


if __name__ == "__main__":
    unittest.main()
