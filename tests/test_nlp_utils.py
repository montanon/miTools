import unittest
from unittest import TestCase

from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.nlp.utils import word_tokenize
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


if __name__ == "__main__":
    unittest.main()
