import os
import unittest
from typing import Dict, List, Tuple
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd
from pandas import DataFrame

from mitools.nlp import (
    RegexpTokenizer,
    StopwordsManager,
    find_countries_in_dataframe,
    find_country_in_token,
    gen_clusters_ngrams_sankey_colors,
    gen_clusters_ngrams_sankey_links_colors,
    gen_clusters_ngrams_sankey_nodes_colors,
    gen_clusters_ngrams_sankey_positions,
    get_bow_of_tokens,
    get_cluster_ngrams,
    get_clustered_dataframe_tokens,
    get_clusters_ngrams,
    get_dataframe_bow,
    get_dataframe_bow_chunks,
    get_dataframe_tokens,
    get_ngram_count,
    get_tfidf,
    lemmatize_text,
    lemmatize_token,
    lemmatize_tokens,
    nltk_tag_to_wordnet_tag,
    nltk_tags_to_wordnet_tags,
    preprocess_country_name,
    preprocess_text,
    preprocess_texts,
    preprocess_token,
    preprocess_tokens,
    replace_sequences,
    sort_multiindex_dataframe,
    tag_token,
    tag_tokens,
    wordnet,
)


class TestNltkTagsToWordnetTags(TestCase):
    def test_simple_conversion(self):
        nltk_tags = [("I", "PRP"), ("run", "VBP"), ("fast", "RB")]
        expected_tags = [
            ("I", wordnet.NOUN),
            ("run", wordnet.VERB),
            ("fast", wordnet.ADV),
        ]
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
        expected_tags = [("!", "n")]
        self.assertEqual(nltk_tags_to_wordnet_tags(nltk_tags), expected_tags)


class TestTagTokens(TestCase):
    def test_tag_tokens_simple(self):
        tokens = ["A", "Man", "run", "fast"]
        expected_tags = [
            ("a", wordnet.NOUN),
            ("man", wordnet.NOUN),
            ("run", wordnet.VERB),
            ("fast", wordnet.ADV),
        ]
        self.assertEqual(
            tag_tokens(tokens), expected_tags
        )  # Convert map to list for comparison

    def test_tag_tokens_empty(self):
        tokens = []
        expected_tags = []
        self.assertEqual(tag_tokens(tokens), expected_tags)

    def test_tag_tokens_mixed_case(self):
        tokens = ["A", "Woman", "RUNs", "FaSt"]
        expected_tags = [
            ("a", wordnet.NOUN),
            ("woman", wordnet.NOUN),
            ("runs", wordnet.VERB),
            ("fast", wordnet.ADV),
        ]
        self.assertEqual(tag_tokens(tokens), expected_tags)

    # Additional test case for adjective:
    def test_tag_tokens_adjective(self):
        tokens = ["beautiful", "woman"]
        expected_tags = [("beautiful", wordnet.ADJ), ("woman", wordnet.NOUN)]
        self.assertEqual(tag_tokens(tokens), expected_tags)

    # Test case for unknown POS:
    def test_tag_tokens_unknown(self):
        tokens = ["!"]
        expected_tags = [("!", wordnet.NOUN)]
        self.assertEqual(tag_tokens(tokens), expected_tags)


class TestTagToken(TestCase):
    def test_tag_tokens_simple(self):
        token = "Man"
        expected_tags = [("man", wordnet.NOUN)]
        self.assertEqual(tag_token(token), expected_tags)

    def test_tag_tokens_empty(self):
        token = ""
        expected_tags = []
        self.assertEqual(tag_tokens(token), expected_tags)

    def test_tag_tokens_mixed_cases(self):
        tokens = ["Clear", "Run", "Flower", "Loudly", "I", "!"]
        expected_tags = [
            ("clear", wordnet.ADJ),
            ("run", wordnet.VERB),
            ("flower", wordnet.NOUN),
            ("loudly", wordnet.ADV),
            ("i", wordnet.NOUN),
            ("!", wordnet.NOUN),
        ]
        for token, expected in zip(tokens, expected_tags):
            self.assertEqual(tag_token(token), [expected])


class TestNltkTagToWordnetTag(TestCase):
    def test_noun_conversion(self):
        self.assertEqual(nltk_tag_to_wordnet_tag("NN"), wordnet.NOUN)
        self.assertEqual(nltk_tag_to_wordnet_tag("NNS"), wordnet.NOUN)
        self.assertEqual(nltk_tag_to_wordnet_tag("NNP"), wordnet.NOUN)

    def test_verb_conversion(self):
        self.assertEqual(nltk_tag_to_wordnet_tag("VB"), wordnet.VERB)
        self.assertEqual(nltk_tag_to_wordnet_tag("VBD"), wordnet.VERB)
        self.assertEqual(nltk_tag_to_wordnet_tag("VBG"), wordnet.VERB)

    def test_adjective_conversion(self):
        self.assertEqual(nltk_tag_to_wordnet_tag("JJ"), wordnet.ADJ)
        self.assertEqual(nltk_tag_to_wordnet_tag("JJR"), wordnet.ADJ)

    def test_adverb_conversion(self):
        self.assertEqual(nltk_tag_to_wordnet_tag("RB"), wordnet.ADV)
        self.assertEqual(nltk_tag_to_wordnet_tag("RBR"), wordnet.ADV)

    def test_unknown_conversion(self):
        self.assertEqual(nltk_tag_to_wordnet_tag("."), wordnet.NOUN)
        self.assertEqual(nltk_tag_to_wordnet_tag("WP$"), wordnet.NOUN)
        self.assertEqual(nltk_tag_to_wordnet_tag("UH"), wordnet.NOUN)


class TestLemmatizeText(TestCase):
    def test_default(self):
        text = "I runs fastest"
        expected_result = "i run fast"
        self.assertEqual(lemmatize_text(text), expected_result)


class TestLemmatizeTokens(TestCase):
    def test_default(self):
        tokens = ["I", "runs", "longest"]
        expected_result = ["i", "run", "long"]
        self.assertEqual(lemmatize_tokens(tokens), expected_result)


class TestLemmatizeToken(TestCase):
    def test_default(self):
        token = "longest"
        expected_result = "long"
        self.assertEqual(lemmatize_token(token), expected_result)


class TestPreprocessText(TestCase):
    def test_default_tokenization(self):
        text = "Hello, World! 123"
        expected_tokens = ["Hello", "World"]
        self.assertEqual(preprocess_text(text), expected_tokens)

    def test_custom_tokenizer(self):
        text = "Hello, World! 123"
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["Hello", "World", "123"]
        self.assertEqual(
            preprocess_text(text, tokenizer=mock_tokenizer), ["Hello", "World", "123"]
        )

    def test_stopword_removal(self):
        text = "This is a sample text."
        stop_words = ["this", "is", "a"]
        expected_tokens = ["sample", "text"]
        self.assertEqual(preprocess_text(text, stop_words=stop_words), expected_tokens)

    def test_lemmatization(self):
        text = "I runs fast"
        expected_tokens = ["run", "fast"]
        self.assertEqual(preprocess_text(text, lemmatize=True), expected_tokens)

    def test_custom_lemmatizer(self):
        text = "I runs fast"
        mock_lemmatizer = Mock()
        mock_lemmatizer.lemmatize.side_effect = lambda token, pos: token
        self.assertEqual(
            preprocess_text(text, lemmatize=True, lemmatizer=mock_lemmatizer),
            ["runs", "fast"],
        )


class TestPreprocessTexts(TestCase):
    def test_batch_default_tokenization(self):
        texts = ["Hello, World! 123", "This is a test."]
        expected_results = [["Hello", "World"], ["This", "is", "test"]]
        self.assertEqual(preprocess_texts(texts), expected_results)

    def test_batch_custom_tokenizer(self):
        texts = ["Hello, World! 123", "Another test."]
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.side_effect = lambda text: text.split()
        expected_results = [["Hello,", "World!", "123"], ["Another", "test."]]
        self.assertEqual(
            preprocess_texts(texts, tokenizer=mock_tokenizer), expected_results
        )

    def test_batch_stopword_removal(self):
        texts = ["This is a test.", "Another example."]
        stop_words = ["this", "is", "a", "another"]
        expected_results = [["test"], ["example"]]
        self.assertEqual(
            preprocess_texts(texts, stop_words=stop_words), expected_results
        )

    def test_batch_lemmatization(self):
        texts = ["I runs fast", "She plays guitar"]
        expected_results = [["run", "fast"], ["she", "play", "guitar"]]
        self.assertEqual(preprocess_texts(texts, lemmatize=True), expected_results)


class TestPreprocessToken(TestCase):
    def test_basic_token(self):
        token = "Hello"
        self.assertEqual(preprocess_token(token), "Hello")

    def test_lemmatization(self):
        token = "running"
        self.assertEqual(preprocess_token(token, lemmatize=True), "run")

    def test_custom_lemmatizer(self):
        token = "running"
        mock_lemmatizer = Mock()
        mock_lemmatizer.lemmatize.return_value = "runner"
        self.assertEqual(
            preprocess_token(token, lemmatize=True, lemmatizer=mock_lemmatizer),
            "runner",
        )

    def test_stopword_removal(self):
        token = "an"
        stop_words = ["an"]
        self.assertEqual(preprocess_token(token, stop_words=stop_words), "")

    def test_case_insensitivity(self):
        token = "An"
        stop_words = ["an"]
        self.assertEqual(preprocess_token(token, stop_words=stop_words), "")

    def test_no_lemmatize_no_stopword(self):
        token = "running"
        stop_words = ["run"]
        self.assertEqual(preprocess_token(token, stop_words=stop_words), "running")

    def test_empty_token(self):
        token = ""
        self.assertEqual(preprocess_token(token), "")


class TestPreprocessTokens(TestCase):
    def test_basic_tokens(self):
        tokens = ["Hello", "World"]
        self.assertEqual(preprocess_tokens(tokens), ["Hello", "World"])

    def test_lemmatization(self):
        tokens = ["running", "flies"]
        expected_tokens = ["run", "fly"]
        self.assertEqual(preprocess_tokens(tokens, lemmatize=True), expected_tokens)

    def test_custom_lemmatizer(self):
        tokens = ["running", "flies"]
        mock_lemmatizer = Mock()
        mock_lemmatizer.lemmatize.side_effect = lambda token, tag: token[:-1]
        expected_tokens = ["runnin", "flie"]
        self.assertEqual(
            preprocess_tokens(tokens, lemmatize=True, lemmatizer=mock_lemmatizer),
            expected_tokens,
        )

    def test_stopword_removal(self):
        tokens = ["This", "is", "a", "test"]
        stop_words = ["this", "is", "a"]
        expected_tokens = ["test"]
        self.assertEqual(
            preprocess_tokens(tokens, stop_words=stop_words), expected_tokens
        )

    def test_lemmatize_and_stopword_removal(self):
        tokens = ["This", "is", "running"]
        stop_words = ["this", "is", "be"]
        expected_tokens = ["run"]
        self.assertEqual(
            preprocess_tokens(tokens, stop_words=stop_words, lemmatize=True),
            expected_tokens,
        )

    def test_empty_tokens(self):
        tokens = []
        self.assertEqual(preprocess_tokens(tokens), [])


class TestGetTfidf(TestCase):
    def test_basic_dataframe(self):
        data = {"word1": [1, 0, 1], "word2": [0, 1, 1]}
        words_count = DataFrame(data)
        df_tfidf = get_tfidf(words_count)
        # Check if the returned dataframe has the expected columns
        self.assertListEqual(list(df_tfidf.columns), ["word1", "word2"])
        # Check if the returned dataframe has the expected shape
        self.assertEqual(df_tfidf.shape, (3, 2))
        # You can also add tests for the expected TF-IDF values.
        # Exact values will depend on the TfidfTransformer's behavior.
        # For instance:
        self.assertGreater(df_tfidf.loc[0, "word1"], 0)

    def test_empty_dataframe(self):
        words_count = DataFrame()
        with self.assertRaises(ValueError):
            get_tfidf(words_count)

    def test_dataframe_with_zeros(self):
        data = {"word1": [0, 0, 0], "word2": [0, 0, 0]}
        words_count = DataFrame(data)
        df_tfidf = get_tfidf(words_count)
        # Check if the returned dataframe has the expected columns
        self.assertListEqual(list(df_tfidf.columns), ["word1", "word2"])
        # Check if the returned dataframe's values are all zeros
        self.assertTrue((df_tfidf == 0).all().all())


class TestGetBowOfTokens(TestCase):
    def test_basic_tokens(self):
        tokens = ["apple", "banana", "apple"]
        expected_bow = {"apple": 2, "banana": 1}
        self.assertEqual(get_bow_of_tokens(tokens), expected_bow)

    def test_preprocess(self):
        tokens = ["apple", "APPLE", "ApPlE", "banana", "BANANA"]
        expected_bow = {"apple": 3, "banana": 2}
        self.assertEqual(get_bow_of_tokens(tokens, preprocess=True), expected_bow)

    def test_stopwords(self):
        tokens = ["apple", "banana", "cherry", "banana"]
        stop_words = ["banana"]
        expected_bow = {"apple": 1, "cherry": 1}
        self.assertEqual(
            get_bow_of_tokens(tokens, preprocess=True, stop_words=stop_words),
            expected_bow,
        )

    def test_sorted_output(self):
        tokens = ["apple", "banana", "cherry", "banana", "cherry", "cherry"]
        expected_bow = {"cherry": 3, "banana": 2, "apple": 1}
        self.assertEqual(get_bow_of_tokens(tokens), expected_bow)

    def test_empty_input(self):
        tokens = []
        with self.assertRaises(ValueError):
            get_bow_of_tokens(tokens)


class TestGetDataframeBow(TestCase):
    def setUp(self):
        self.data = {
            "text": [
                "This is a sample text.",
                "Another sample text here.",
                "Yet another text sample.",
            ]
        }
        self.df = DataFrame(self.data)

    def test_basic_functionality(self):
        bow_df = get_dataframe_bow(self.df, "text")
        self.assertIn("sample", bow_df.columns)
        self.assertEqual(bow_df["sample"].iloc[0], 1)
        self.assertEqual(bow_df["sample"].iloc[1], 1)
        self.assertEqual(bow_df["sample"].iloc[2], 1)
        self.assertEqual(bow_df["this"].iloc[0], 1)
        self.assertEqual(bow_df["this"].iloc[1], 0)

    def test_preprocess(self):
        bow_df = get_dataframe_bow(self.df, "text", preprocess=True)
        # Assuming preprocessing will convert words to lowercase
        self.assertIn("sample", bow_df.columns)
        self.assertNotIn("This", bow_df.columns)

    def test_stopwords(self):
        stop_words = ["this", "is", "a", "another", "yet"]
        bow_df = get_dataframe_bow(
            self.df, "text", preprocess=True, stop_words=stop_words
        )
        self.assertNotIn("this", bow_df.columns)
        self.assertNotIn("is", bow_df.columns)


class TestGetDataframeBowChunks(TestCase):
    def setUp(self):
        self.data = {
            "text": [
                "This is a sample text.",
                "Another sample text here.",
                "Yet another text sample.",
            ]
        }
        self.df = pd.DataFrame(self.data)

    def test_basic_functionality(self):
        bow_df = get_dataframe_bow_chunks(self.df, "text")
        self.assertIn("sample", bow_df.columns)
        self.assertEqual(bow_df["sample"].iloc[0], 1)
        self.assertEqual(bow_df["sample"].iloc[1], 1)
        self.assertEqual(bow_df["sample"].iloc[2], 1)
        self.assertEqual(bow_df["this"].iloc[0], 1)
        self.assertEqual(bow_df["this"].iloc[1], 0)

    def test_preprocess(self):
        bow_df = get_dataframe_bow_chunks(self.df, "text", preprocess=True)
        self.assertIn("sample", bow_df.columns)
        self.assertNotIn("This", bow_df.columns)

    def test_stopwords(self):
        stop_words = ["this", "is", "a", "another", "yet"]
        bow_df = get_dataframe_bow_chunks(
            self.df, "text", preprocess=True, stop_words=stop_words
        )
        self.assertNotIn("this", bow_df.columns)
        self.assertNotIn("is", bow_df.columns)

    def test_chunk_processing(self):
        large_data = {"text": ["Text sample."] * 5_000}  # More than 2 chunks
        large_df = pd.DataFrame(large_data)
        bow_df = get_dataframe_bow_chunks(large_df, "text", chunk_size=2_500)
        self.assertEqual(len(bow_df), 5_000)
        self.assertEqual(bow_df["sample"].sum(), 5_000)

    def test_small_dataframe(self):
        small_data = {"text": ["Small dataset."]}
        small_df = pd.DataFrame(small_data)
        bow_df = get_dataframe_bow_chunks(small_df, "text")
        self.assertEqual(len(bow_df), 1)
        self.assertIn("small", bow_df.columns)


class TestGetNgramCount(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame setup
        self.df = pd.DataFrame(
            {
                "text": ["This is a test", "Another test text", "More text data"],
                "text_id": [1, 2, 3],
            }
        )
        self.text_col = "text"
        self.id_col = "text_id"

    def test_basic_functionality(self):
        result = get_ngram_count(self.df, self.text_col, self.id_col)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[1], 7)  # 7 unique words in the sample data
        self.assertEqual(len(result), 3)  # 3 rows corresponding to 3 ids

    def test_different_ngram_range(self):
        result = get_ngram_count(
            self.df, self.text_col, self.id_col, ngram_range=(2, 2)
        )
        # Assuming bigrams are correctly formed, e.g., 'this is', but not 'is a' because of default tokenizer
        self.assertTrue(
            "this is" in result.columns and "another test" in result.columns
        )

    def test_with_custom_tokenizer(self):
        custom_tokenizer = RegexpTokenizer("[A-Za-z]{4,}")
        result = get_ngram_count(
            self.df, self.text_col, self.id_col, tokenizer=custom_tokenizer
        )
        # Expecting words with 4 or more letters only
        self.assertTrue("this" in result.columns and "test" in result.columns)
        self.assertFalse("is" in result.columns or "a" in result.columns)

    def test_with_stop_words(self):
        stop_words = ["is", "a"]
        result = get_ngram_count(
            self.df, self.text_col, self.id_col, stop_words=stop_words
        )
        # 'is' and 'a' should not be in the columns
        self.assertFalse("is" in result.columns or "a" in result.columns)

    def test_with_no_stop_words(self):
        stop_words = None
        result = get_ngram_count(
            self.df, self.text_col, self.id_col, stop_words=stop_words
        )
        # 'is' and 'a' should be in the columns
        self.assertTrue("is" in result.columns or "a" in result.columns)

    def test_with_max_features(self):
        result = get_ngram_count(self.df, self.text_col, self.id_col, max_features=2)
        # Only 2 features (most frequent) should be returned
        self.assertEqual(result.shape[1], 2)

    def test_lowercasing(self):
        result = get_ngram_count(
            self.df, self.text_col, self.id_col, ngram_range=(2, 2), lowercase=False
        )
        # Assuming bigrams are correctly formed, e.g., 'this is', but not 'is a' because of default tokenizer
        self.assertTrue(
            "This is" in result.columns and "Another test" in result.columns
        )

    def test_frequency(self):
        pass

    def test_error_handling(self):
        with self.assertRaises(KeyError):
            get_ngram_count(self.df, "wrong_col", self.id_col)


class TestGetDataFrameTokens(unittest.TestCase):
    def test_basic_functionality(self):
        df = DataFrame({"text_id": [1, 2], "text": ["Hello World", "Sample Text"]})
        result = get_dataframe_tokens(df, "text", "text_id")
        self.assertEqual(list(result.columns), [1, 2])
        self.assertEqual(result.iloc[0, 0], "hello")

    def test_stop_words_removal(self):
        df = DataFrame({"text_id": [1], "text": ["hello world"]})
        result = get_dataframe_tokens(df, "text", "text_id", stop_words=["world"])
        self.assertNotIn("world", result[1])

    def test_custom_tokenizer(self):
        # Define a simple custom tokenizer for the test
        class CustomTokenizer:
            def tokenize(self, text):
                return text.split()

        df = DataFrame({"text_id": [1], "text": ["hello world"]})
        result = get_dataframe_tokens(
            df, "text", "text_id", tokenizer=CustomTokenizer()
        )
        self.assertEqual(result.iloc[0, 0], "hello")

    def test_lowercasing(self):
        df = DataFrame({"text_id": [1], "text": ["Hello World"]})
        result = get_dataframe_tokens(df, "text", "text_id", lowercase=False)
        self.assertEqual(result.iloc[0, 0], "Hello")

    def test_empty_dataframe(self):
        df = DataFrame({"text_id": [], "text": []})
        result = get_dataframe_tokens(df, "text", "text_id")
        self.assertTrue(result.empty)


class TestGetClusteredDataFrameTokens(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "text_id": [1, 2, 3],
                "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
                "cluster": [0, 1, 0],
            }
        )

    def test_typical_case(self):
        # Test with typical data
        result = get_clustered_dataframe_tokens(self.df, "text", "text_id", "cluster")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        # Check if MultiIndex is correctly formatted
        self.assertTrue(
            all(
                c in ["Cluster 0", "Cluster 1"]
                for c in result.columns.get_level_values(0)[:2]
            )
        )
        self.assertTrue(
            all(c in [2, 3] for c in result.columns.get_level_values(1)[1:])
        )
        # Check if the number of texts matches
        self.assertEqual(result.shape[1], self.df.shape[0])

    def test_empty_dataframe(self):
        # Test with an empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(KeyError):
            get_clustered_dataframe_tokens(empty_df, "text", "text_id", "cluster")

    def test_missing_columns(self):
        # Test with missing columns
        with self.assertRaises(KeyError):
            get_clustered_dataframe_tokens(
                self.df, "nonexistent_text_col", "text_id", "cluster"
            )
        with self.assertRaises(KeyError):
            get_clustered_dataframe_tokens(
                self.df, "text_col", "nonexistent_text_id", "cluster"
            )
        with self.assertRaises(KeyError):
            get_clustered_dataframe_tokens(
                self.df, "text_col", "text_id", "nonexistent_cluster"
            )


class TestGetClustersNgrams(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "text_id": [1, 2, 3, 4],
                "text": [
                    "apple banana Apple banana melon",
                    "apple banana melon apple orange",
                    "apple banana durian melon banana apple apple",
                    "orange banana raspberry",
                ],
                "cluster_id": [1, 1, 2, 2],
            }
        )

    def test_normal_operation(self):
        result = get_clusters_ngrams(
            self.sample_data, "text", "text_id", "cluster_id", 2
        )
        from IPython.display import display

        display(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))

    def test_empty_data(self):
        empty_data = pd.DataFrame(columns=["text_id", "text", "cluster_id"])
        with self.assertRaises(ValueError):
            get_clusters_ngrams(empty_data, "text", "text_id", "cluster_id", 2)

    def test_stop_words(self):
        stop_words = ["melon"]
        result = get_clusters_ngrams(
            self.sample_data, "text", "text_id", "cluster_id", 2, stop_words=stop_words
        )
        self.assertNotIn("melon", result.iloc[:, 0])

    def test_ngram_range(self):
        ngram_range = (1, 3)
        result = get_clusters_ngrams(
            self.sample_data,
            "text",
            "text_id",
            "cluster_id",
            2,
            ngram_range=ngram_range,
        )
        # Check if n-grams within specified range are present
        self.assertIn(2, [len(col[0].split()) for col in result.columns])
        # Check if n-grams outside specified range are present
        self.assertNotIn(3, [len(col[0].split()) for col in result.columns])

    def test_lowercase_true(self):
        result = get_clusters_ngrams(
            self.sample_data, "text", "text_id", "cluster_id", 2, lowercase=True
        )
        all_lower = all(x.islower() for x in result[("Cluster 1", "1-Gram", "Gram")])
        self.assertTrue(all_lower)

    def test_lowercase_false(self):
        result = get_clusters_ngrams(
            self.sample_data, "text", "text_id", "cluster_id", 2, lowercase=False
        )
        all_lower = all(x.islower() for x in result[("Cluster 1", "1-Gram", "Gram")])
        self.assertFalse(all_lower)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            get_clusters_ngrams(self.sample_data, "text", "text_id", "cluster_id", -1)


class TestGetClusterNgrams(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "text_id": [1, 2, 3],
                "text": ["apple banana", "apple", "banana apple apple pear melon"],
            }
        )

    def test_normal_operation(self):
        result = get_cluster_ngrams(self.sample_data, "text", "text_id", 1, 2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(("Cluster 1", "2-Gram", "Gram"), result.columns)
        self.assertIn(("Cluster 1", "2-Gram", "Frequency"), result.columns)

    def test_max_features(self):
        result = get_cluster_ngrams(
            self.sample_data, "text", "text_id", 1, 2, max_features=1
        )
        self.assertEqual(len(result), 1)

    def test_stop_words(self):
        stop_words = ["apple"]
        result = get_cluster_ngrams(
            self.sample_data, "text", "text_id", 1, 2, stop_words=stop_words
        )
        self.assertNotIn("apple", result[("Cluster 1", "2-Gram", "Gram")].values)

    def test_frequency_false(self):
        result = get_cluster_ngrams(
            self.sample_data, "text", "text_id", 1, 2, frequency=False
        )
        self.assertIn(("Cluster 1", "2-Gram", "Count"), result.columns)
        all_ints = all(
            [isinstance(v, int) for v in result[("Cluster 1", "2-Gram", "Count")]]
        )
        self.assertTrue(all_ints)

    def test_lowercase_true(self):
        result = get_cluster_ngrams(
            self.sample_data, "text", "text_id", 1, 2, lowercase=True
        )
        all_lower = all(
            x.islower() for x in result[("Cluster 1", "2-Gram", "Gram")].values
        )
        self.assertTrue(all_lower)

    def test_lowercase_false(self):
        result = get_cluster_ngrams(
            self.sample_data, "text", "text_id", 1, 2, lowercase=False
        )
        all_lower = all(
            x.islower() for x in result[("Cluster 1", "2-Gram", "Gram")].values
        )
        self.assertFalse(all_lower)

    def test_empty_data(self):
        empty_data = pd.DataFrame(columns=["text_id", "text"])
        with self.assertRaises(ValueError):
            get_cluster_ngrams(empty_data, "text", "text_id", 1, 2)

    def test_str_cluster(self):
        result = get_cluster_ngrams(self.sample_data, "text", "text_id", "string", 2)
        self.assertIn(("Cluster string", "2-Gram", "Gram"), result.columns)


class TestGenClustersNgramsSankeyColors(unittest.TestCase):
    def test_return_type(self):
        result = gen_clusters_ngrams_sankey_colors(["a", "b"], ["x", "y"])
        self.assertIsInstance(result, Dict)

    def test_structure_of_return_value(self):
        result = gen_clusters_ngrams_sankey_colors(["a", "b"], ["x", "y"])
        for key, value in result.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, List)
            self.assertEqual(len(value), 4)
            for color_component in value:
                self.assertIsInstance(color_component, float)

    def test_specific_input(self):
        sources = ["a", "b"]
        targets = ["x", "y"]
        result = gen_clusters_ngrams_sankey_colors(sources, targets)
        for key in [*sources, *targets]:
            self.assertIn(key, result)

    def test_empty_input(self):
        result = gen_clusters_ngrams_sankey_colors([], [])
        self.assertEqual(result, {})

    def test_different_input(self):
        result = gen_clusters_ngrams_sankey_colors(["a"], ["x", "y"])
        self.assertIn("a", result)
        self.assertIn("x", result)
        self.assertIn("y", result)


class TestGenClustersNgramsSankeyPositions(unittest.TestCase):
    def test_return_type(self):
        result = gen_clusters_ngrams_sankey_positions(["a", "b"], len(["a"]))
        self.assertIsInstance(result, Tuple)

    def test_length_of_output(self):
        labels = ["a", "b", "c"]
        sources = ["a"]
        result = gen_clusters_ngrams_sankey_positions(labels, len(sources))
        self.assertEqual(len(result[0]), len(labels))
        self.assertEqual(len(result[1]), len(labels))

    def test_x_and_y_positions(self):
        labels = ["a", "b", "c"]
        sources = ["a"]
        x_expected = [0.001, 0.999, 0.999]
        y_expected = [0.001, 0.001, 0.999]
        result = gen_clusters_ngrams_sankey_positions(labels, len(sources))
        self.assertEqual(result[0], x_expected)
        self.assertEqual(result[1], y_expected)

    def test_value_range(self):
        labels = ["a", "b", "c"]
        sources = ["a"]
        result = gen_clusters_ngrams_sankey_positions(labels, len(sources))
        all_values = result[0] + result[1]
        self.assertTrue(all(0.001 <= v <= 0.999 for v in all_values))

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            gen_clusters_ngrams_sankey_positions([], 1)

    def test_single_element_input(self):
        result = gen_clusters_ngrams_sankey_positions(["a"], 1)
        self.assertEqual(result, ([0.001], [0.001]))


class TestGenClustersNgramsSankeyLinksColors(unittest.TestCase):
    def test_return_type(self):
        result = gen_clusters_ngrams_sankey_links_colors({}, [], {})
        self.assertIsInstance(result, List)

    def test_return_format(self):
        labels_ids = {"a": "1"}
        targets = ["1"]
        labels_colors = {"a": (255, 0, 0)}
        result = gen_clusters_ngrams_sankey_links_colors(
            labels_ids, targets, labels_colors
        )
        for color in result:
            self.assertIsInstance(color, str)
            self.assertRegex(color, r"rgba\(\d+,\d+,\d+,[\d\.]+\)")

    def test_correct_color_transformation(self):
        labels_ids = {"a": "1", "b": "2"}
        targets = ["1", "2"]
        labels_colors = {"a": (255, 0, 0), "b": (0, 255, 0)}
        result = gen_clusters_ngrams_sankey_links_colors(
            labels_ids, targets, labels_colors
        )
        expected_colors = ["rgba(255,0,0,0.5)", "rgba(0,255,0,0.5)"]
        self.assertEqual(result, expected_colors)

    def test_handling_unknown_labels(self):
        labels_ids = {"a": "1"}
        targets = ["1", "unknown"]
        labels_colors = {"a": (255, 0, 0)}
        with self.assertRaises(KeyError):
            gen_clusters_ngrams_sankey_links_colors(labels_ids, targets, labels_colors)

    def test_empty_input(self):
        result = gen_clusters_ngrams_sankey_links_colors({}, [], {})
        self.assertEqual(result, [])


class TestGenClustersNgramsSankeyNodesColors(unittest.TestCase):
    def test_return_type(self):
        result = gen_clusters_ngrams_sankey_nodes_colors([], {})
        self.assertIsInstance(result, List)

    def test_return_format(self):
        labels = ["a", "b"]
        labels_colors = {"a": (255, 0, 0, 1), "b": (0, 255, 0, 1)}
        result = gen_clusters_ngrams_sankey_nodes_colors(labels, labels_colors)
        for color in result:
            self.assertIsInstance(color, str)
            self.assertRegex(color, r"rgba\(\d+,\d+,\d+,[\d\.]+\)")

    def test_correct_color_transformation(self):
        labels = ["a", "b"]
        labels_colors = {"a": (255, 0, 0, 1), "b": (0, 255, 0, 1)}
        result = gen_clusters_ngrams_sankey_nodes_colors(labels, labels_colors)
        expected_colors = ["rgba(255,0,0,1)", "rgba(0,255,0,1)"]
        self.assertEqual(result, expected_colors)

    def test_handling_unknown_labels(self):
        labels = ["a", "unknown"]
        labels_colors = {"a": (255, 0, 0, 1)}
        with self.assertRaises(KeyError):
            gen_clusters_ngrams_sankey_nodes_colors(labels, labels_colors)

    def test_empty_input(self):
        result = gen_clusters_ngrams_sankey_nodes_colors([], {})
        self.assertEqual(result, [])


class TestPreprocessCountryName(unittest.TestCase):
    def test_unicode_conversion(self):
        self.assertEqual(preprocess_country_name("México"), "mexico")
        self.assertEqual(preprocess_country_name("Brásil"), "brasil")
        self.assertEqual(
            preprocess_country_name("São Tomé and Príncipe"), "sao tome and principe"
        )

    def test_lowercase_conversion(self):
        self.assertEqual(preprocess_country_name("CANADA"), "canada")
        self.assertEqual(preprocess_country_name("MONGOLIA"), "mongolia")

    def test_special_characters_removal(self):
        self.assertEqual(preprocess_country_name("New*Zealand!"), "new zealand")
        self.assertEqual(preprocess_country_name("South-Africa"), "south africa")

    def test_combinations(self):
        self.assertEqual(preprocess_country_name("Côte d'Ivoire"), "cote d ivoire")


class TestFindCountriesInToken(TestCase):
    def setUp(self):
        self.countries = ["united kingdom", "united states", "canada", "france"]
        self.demonyms = {
            "british": "united kingdom",
            "american": "united states",
            "canadian": "canada",
            "french": "france",
        }

    def test_exact_country_name(self):
        self.assertEqual(
            find_country_in_token("canada", self.countries, self.demonyms),
            ("canada", "canada"),
        )

    def test_demonym(self):
        self.assertEqual(
            find_country_in_token("british", self.countries, self.demonyms),
            ("united kingdom", "british"),
        )

    def test_special_cases(self):
        self.assertEqual(
            find_country_in_token("uk", self.countries, self.demonyms),
            ("united kingdom", "uk"),
        )

    def test_below_similarity_threshold(self):
        self.assertEqual(
            find_country_in_token("united st", self.countries, self.demonyms),
            (None, None),
        )

    def test_no_relation(self):
        self.assertEqual(
            find_country_in_token("apple", self.countries, self.demonyms), (None, None)
        )


class TestFindCountriesInDataframe(TestCase):
    def setUp(self):
        self.df_mock = pd.DataFrame(
            {
                "Text1": ["uk", "american", "french"],
                "Text2": ["canada", "british", "Juice"],
            }
        )
        self.countries = ["united kingdom", "united states", "canada", "france"]
        self.demonyms = {
            "british": "united kingdom",
            "american": "united states",
            "canadian": "canada",
            "french": "france",
        }

    def test_find_countries_in_dataframe(self):
        result_df = find_countries_in_dataframe(
            self.df_mock, self.countries, self.demonyms
        )
        expected_df = DataFrame(
            {
                "Text1": [
                    ("united kingdom", "uk"),
                    ("united states", "american"),
                    ("france", "french"),
                ],
                "Text2": [
                    ("canada", "canada"),
                    ("united kingdom", "british"),
                    (None, None),
                ],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)


class TestSortMultiIndexDataframe(TestCase):
    def setUp(self):
        arrays = [
            ["A", "A", "B", "B"],
            ["col1", "col2", "col1", "col2"],
        ]
        self.df_mock = pd.DataFrame(
            [
                [1, 2, 3, 12],
                [5, 6, 7, 8],
                [9, 10, 11, 4],
            ],
            columns=pd.MultiIndex.from_arrays(arrays, names=("top_level", "bot_level")),
        )

    def test_sort_multiindex_dataframe(self):
        result_df = sort_multiindex_dataframe(
            self.df_mock, ["col1", "col2"], sorting_col="col2", ascending=True
        )
        arrays_sorted = [
            ["A", "A", "B", "B"],
            ["col1", "col2", "col1", "col2"],
        ]
        expected_df = pd.DataFrame(
            [[1, 2, 11, 4], [5, 6, 7, 8], [9, 10, 3, 12]],
            columns=pd.MultiIndex.from_arrays(
                arrays_sorted, names=("top_level", "bot_level")
            ),
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        result_df = sort_multiindex_dataframe(
            self.df_mock, ["col1", "col2"], sorting_col="col1", ascending=False
        )
        arrays_sorted = [
            ["A", "A", "B", "B"],
            ["col1", "col2", "col1", "col2"],
        ]
        expected_df = pd.DataFrame(
            [[9, 10, 11, 4], [5, 6, 7, 8], [1, 2, 3, 12]],
            columns=pd.MultiIndex.from_arrays(
                arrays_sorted, names=("top_level", "bot_level")
            ),
        )
        pd.testing.assert_frame_equal(result_df, expected_df)


class TestStopwordsManager(unittest.TestCase):
    def setUp(self):
        self.manager = StopwordsManager()
        self.filename = "test.pkl"

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_add_single_stopword(self):
        self.manager.add_stopword("testword")
        self.assertIn("testword", self.manager.words)

    def test_add_multiple_stopwords(self):
        words = ["testword1", "testword2"]
        self.manager.add_stopwords(words)
        self.assertTrue(set(words).issubset(self.manager.words))

    def test_remove_single_stopword(self):
        self.manager.add_stopword("testword")
        self.manager.remove_stopword("testword")
        self.assertNotIn("testword", self.manager.words)

    def test_remove_multiple_stopwords(self):
        words = ["testword1", "testword2"]
        self.manager.add_stopwords(words)
        self.manager.remove_stopwords(words)
        self.assertFalse(set(words).issubset(self.manager.words))

    def test_save_load(self):
        self.manager.add_stopword("testword")
        self.manager.save(self.filename)
        loaded_manager = StopwordsManager.load(self.filename)
        self.assertIn("testword", loaded_manager.words)


class TestReplaceSequences(unittest.TestCase):
    def test_single_word_replacement(self):
        tokens = ["this", "is", "a"]
        mapping = {"test": ("a",)}
        expected_output = ["this", "is", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_single_word_replacement_with_repetition(self):
        tokens = ["this", "is", "a", "test"]
        mapping = {"test": ("a",)}
        expected_output = ["this", "is", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_multiple_words_replacement(self):
        tokens = ["this", "is", "another", "test"]
        mapping = {"another_test": ("another", "test")}
        expected_output = ["this", "is", "another_test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_multiple_occurrences(self):
        tokens = [
            "this",
            "is",
            "yet",
            "another",
            "test",
            "or",
            "yet",
            "another",
            "test",
        ]
        mapping = {"yet_another_test": ("yet", "another", "test")}
        expected_output = ["this", "is", "yet_another_test", "or", "yet_another_test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_multiple_occurrences_with_repetition(self):
        tokens = ["this", "is", "yet", "another", "test", "or", "yet", "another"]
        mapping = {"test": ("yet", "another")}
        expected_output = ["this", "is", "test", "or", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_multiple_occurrences_with_consequent_repetition(self):
        tokens = ["this", "is", "yet", "another", "test", "yet", "another"]
        mapping = {"test": ("yet", "another")}
        expected_output = ["this", "is", "test", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_overlapping_sequences(self):
        tokens = ["this", "is", "yet", "another", "test"]
        mapping = {
            "yet_another": ("yet", "another"),
            "another_test": ("another", "test"),
        }
        expected_output = ["this", "is", "yet_another", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_no_sequences_to_replace(self):
        tokens = ["this", "is", "a", "test"]
        mapping = {"not_present": ("not", "present")}
        expected_output = ["this", "is", "a", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)

    def test_list_value_sequences(self):
        tokens = ["this", "is", "a", "test", "this", "is", "a", "tests"]
        mapping = {"test": [("a", "test"), ("a", "tests")]}
        expected_output = ["this", "is", "test", "this", "is", "test"]
        self.assertEqual(replace_sequences(tokens, mapping), expected_output)


if __name__ == "__main__":
    unittest.main()
