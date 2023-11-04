import unittest
from unittest import TestCase
from mitools.nlp import *


class TestNltkTagsToWordnetTags(TestCase):

    def test_simple_conversion(self):
        nltk_tags = [("I", "PRP"), ("run", "VBP"), ("fast", "RB")]
        expected_tags = [("I", wordnet.NOUN), ("run", wordnet.VERB), ("fast", wordnet.ADV)]
        self.assertEqual(nltk_tags_to_wordnet_tags(nltk_tags), expected_tags)

    def test_empty_input(self):
        nltk_tags = []
        expected_tags = []
        self.assertEqual(nltk_tags_to_wordnet_tags(nltk_tags), expected_tags)

    def test_adjective_conversion(self):
        nltk_tags = [("beautiful", "JJ")]
        expected_tags = [("beautiful", wordnet.ADJ)]
        self.assertEqual(nltk_tags_to_wordnet_tags(nltk_tags), expected_tags)

    def test_unknown_conversion(self):
        nltk_tags = [("!", ".")]
        expected_tags = [("!", 'n')]
        self.assertEqual(nltk_tags_to_wordnet_tags(nltk_tags), expected_tags)


class TestTagTokens(unittest.TestCase):

    def test_tag_tokens_simple(self):
        tokens = ["A", 'Man', "run", "fast"]
        expected_tags = [
            ("a", wordnet.NOUN), 
            ("man", wordnet.NOUN),
            ("run", wordnet.VERB),
            ("fast", wordnet.ADV)
            ]
        self.assertEqual(tag_tokens(tokens), expected_tags)  # Convert map to list for comparison

    def test_tag_tokens_empty(self):
        tokens = []
        expected_tags = []
        self.assertEqual(tag_tokens(tokens), expected_tags)

    def test_tag_tokens_mixed_case(self):
        tokens = ["A", 'Woman', "RUNs", "FaSt"]
        expected_tags = [
            ("a", wordnet.NOUN), 
            ("woman", wordnet.NOUN), 
            ("runs", wordnet.VERB), 
            ("fast", wordnet.ADV)]
        self.assertEqual(tag_tokens(tokens), expected_tags)

    # Additional test case for adjective:
    def test_tag_tokens_adjective(self):
        tokens = ["beautiful", 'woman']
        expected_tags = [
            ("beautiful", wordnet.ADJ),
            ("woman", wordnet.NOUN)]
        self.assertEqual(tag_tokens(tokens), expected_tags)

    # Test case for unknown POS:
    def test_tag_tokens_unknown(self):
        tokens = ["!"]
        expected_tags = [("!", wordnet.NOUN)]
        self.assertEqual(tag_tokens(tokens), expected_tags)

if __name__ == '__main__':
    unittest.main()
