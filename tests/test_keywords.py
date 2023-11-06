import unittest
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd
from pandas import DataFrame

from mitools.nlp import (find_countries_in_dataframe, find_country_in_token,
                         get_bow_of_tokens, get_dataframe_bow, get_tfidf,
                         lemmatize_text, lemmatize_token, lemmatize_tokens,
                         nltk_tag_to_wordnet_tag, nltk_tags_to_wordnet_tags,
                         preprocess_country_name, preprocess_text,
                         preprocess_texts, preprocess_token, preprocess_tokens,
                         sort_multiindex_dataframe, tag_token, tag_tokens,
                         wordnet)


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


class TestTagTokens(TestCase):

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

class TestTagToken(TestCase):

    def test_tag_tokens_simple(self):
        token = 'Man'
        expected_tags = [
            ("man", wordnet.NOUN)
            ]
        self.assertEqual(tag_token(token), expected_tags) 

    def test_tag_tokens_empty(self):
        token = ''
        expected_tags = []
        self.assertEqual(tag_tokens(token), expected_tags)

    def test_tag_tokens_mixed_cases(self):

        tokens = ['Clear', 'Run', 'Flower', 'Loudly', 'I', '!']
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
        self.assertEqual(preprocess_text(text, tokenizer=mock_tokenizer), ["Hello", "World", "123"])

    def test_stopword_removal(self):
        text = "This is a sample text."
        stopwords = ["this", "is", "a"]
        expected_tokens = ["sample", "text"]
        self.assertEqual(preprocess_text(text, stopwords=stopwords), expected_tokens)

    def test_lemmatization(self):
        text = "I runs fast"
        expected_tokens = ["run", "fast"]
        self.assertEqual(preprocess_text(text, lemmatize=True), expected_tokens)

    def test_custom_lemmatizer(self):
        text = "I runs fast"
        mock_lemmatizer = Mock()
        mock_lemmatizer.lemmatize.side_effect = lambda token, pos: token
        self.assertEqual(preprocess_text(text, lemmatize=True, lemmatizer=mock_lemmatizer), ["runs", "fast"])


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
        self.assertEqual(preprocess_texts(texts, tokenizer=mock_tokenizer), expected_results)

    def test_batch_stopword_removal(self):
        texts = ["This is a test.", "Another example."]
        stopwords = ["this", "is", "a", "another"]
        expected_results = [["test"], ["example"]]
        self.assertEqual(preprocess_texts(texts, stopwords=stopwords), expected_results)

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
        self.assertEqual(preprocess_token(token, lemmatize=True, lemmatizer=mock_lemmatizer), "runner")

    def test_stopword_removal(self):
        token = "an"
        stopwords = ["an"]
        self.assertEqual(preprocess_token(token, stopwords=stopwords), "")

    def test_case_insensitivity(self):
        token = "An"
        stopwords = ["an"]
        self.assertEqual(preprocess_token(token, stopwords=stopwords), "")

    def test_no_lemmatize_no_stopword(self):
        token = "running"
        stopwords = ["run"]
        self.assertEqual(preprocess_token(token, stopwords=stopwords), "running")

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
        self.assertEqual(preprocess_tokens(tokens, lemmatize=True, lemmatizer=mock_lemmatizer), expected_tokens)

    def test_stopword_removal(self):
        tokens = ["This", "is", "a", "test"]
        stopwords = ["this", "is", "a"]
        expected_tokens = ["test"]
        self.assertEqual(preprocess_tokens(tokens, stopwords=stopwords), expected_tokens)

    def test_lemmatize_and_stopword_removal(self):
        tokens = ["This", "is", "running"]
        stopwords = ["this", "is", 'be']
        expected_tokens = ["run"]
        self.assertEqual(preprocess_tokens(tokens, stopwords=stopwords, lemmatize=True), expected_tokens)

    def test_empty_tokens(self):
        tokens = []
        self.assertEqual(preprocess_tokens(tokens), [])


class TestGetTfidf(TestCase):

    def test_basic_dataframe(self):
        data = {
            'word1': [1, 0, 1],
            'word2': [0, 1, 1]
        }
        words_count = DataFrame(data)
        df_tfidf = get_tfidf(words_count)
        # Check if the returned dataframe has the expected columns
        self.assertListEqual(list(df_tfidf.columns), ['word1', 'word2'])
        # Check if the returned dataframe has the expected shape
        self.assertEqual(df_tfidf.shape, (3, 2))
        # You can also add tests for the expected TF-IDF values.
        # Exact values will depend on the TfidfTransformer's behavior.
        # For instance:
        self.assertGreater(df_tfidf.loc[0, 'word1'], 0)

    def test_empty_dataframe(self):
        words_count = DataFrame()
        with self.assertRaises(ValueError):
            get_tfidf(words_count)

    def test_dataframe_with_zeros(self):
        data = {
            'word1': [0, 0, 0],
            'word2': [0, 0, 0]
        }
        words_count = DataFrame(data)
        df_tfidf = get_tfidf(words_count)
        # Check if the returned dataframe has the expected columns
        self.assertListEqual(list(df_tfidf.columns), ['word1', 'word2'])
        # Check if the returned dataframe's values are all zeros
        self.assertTrue((df_tfidf == 0).all().all())


class TestGetBowOfTokens(TestCase):

    def test_basic_tokens(self):
        tokens = ["apple", "banana", "apple"]
        expected_bow = {'apple': 2, 'banana': 1}
        self.assertEqual(get_bow_of_tokens(tokens), expected_bow)

    def test_preprocess(self):
        tokens = ["apple", "APPLE", "ApPlE", "banana", "BANANA"]
        expected_bow = {'apple': 3, 'banana': 2}
        self.assertEqual(get_bow_of_tokens(tokens, preprocess=True), expected_bow)

    def test_stopwords(self):
        tokens = ["apple", "banana", "cherry", "banana"]
        stopwords = ["banana"]
        expected_bow = {'apple': 1, 'cherry': 1}
        self.assertEqual(get_bow_of_tokens(tokens, preprocess=True, stopwords=stopwords), expected_bow)

    def test_sorted_output(self):
        tokens = ["apple", "banana", "cherry", "banana", "cherry", "cherry"]
        expected_bow = {'cherry': 3, 'banana': 2, 'apple': 1}
        self.assertEqual(get_bow_of_tokens(tokens), expected_bow)

    def test_empty_input(self):
        tokens = []
        with self.assertRaises(ValueError):
            get_bow_of_tokens(tokens)


class TestGetDataframeBow(TestCase):

    def setUp(self):
        self.data = {
            'text': ['This is a sample text.', 'Another sample text here.', 'Yet another text sample.']
        }
        self.df = DataFrame(self.data)

    def test_basic_functionality(self):
        bow_df = get_dataframe_bow(self.df, 'text')
        self.assertIn('sample', bow_df.columns)
        self.assertEqual(bow_df['sample'].iloc[0], 1)
        self.assertEqual(bow_df['sample'].iloc[1], 1)
        self.assertEqual(bow_df['sample'].iloc[2], 1)
        self.assertEqual(bow_df['this'].iloc[0], 1)
        self.assertEqual(bow_df['this'].iloc[1], 0)

    def test_preprocess(self):
        bow_df = get_dataframe_bow(self.df, 'text', preprocess=True)
        # Assuming preprocessing will convert words to lowercase
        self.assertIn('sample', bow_df.columns)
        self.assertNotIn('This', bow_df.columns)
    
    def test_stopwords(self):
        stopwords = ['this', 'is', 'a', 'another', 'yet']
        bow_df = get_dataframe_bow(self.df, 'text', preprocess=True, stopwords=stopwords)
        self.assertNotIn('this', bow_df.columns)
        self.assertNotIn('is', bow_df.columns)


class TestPreprocessCountryName(unittest.TestCase):

    def test_unicode_conversion(self):
        self.assertEqual(preprocess_country_name('México'), 'mexico')
        self.assertEqual(preprocess_country_name('Brásil'), 'brasil')
        self.assertEqual(preprocess_country_name('São Tomé and Príncipe'), 'sao tome and principe')

    def test_lowercase_conversion(self):
        self.assertEqual(preprocess_country_name('CANADA'), 'canada')
        self.assertEqual(preprocess_country_name('MONGOLIA'), 'mongolia')

    def test_special_characters_removal(self):
        self.assertEqual(preprocess_country_name('New*Zealand!'), 'new zealand')
        self.assertEqual(preprocess_country_name('South-Africa'), 'south africa')

    def test_combinations(self):
        self.assertEqual(preprocess_country_name('Côte d\'Ivoire'), 'cote d ivoire')


class TestFindCountriesInToken(TestCase):

    def setUp(self):
        self.countries = ['united kingdom', 'united states', 'canada', 'france']
        self.demonyms = {
            'british': 'united kingdom',
            'american': 'united states',
            'canadian': 'canada',
            'french': 'france'
        }

    def test_exact_country_name(self):
        self.assertEqual(find_country_in_token('canada', self.countries, self.demonyms), ('canada', 'canada'))

    def test_demonym(self):
        self.assertEqual(find_country_in_token('british', self.countries, self.demonyms), ('united kingdom', 'british'))

    def test_special_cases(self):
        self.assertEqual(find_country_in_token('uk', self.countries, self.demonyms), ('united kingdom', 'uk'))

    def test_below_similarity_threshold(self):
        self.assertEqual(find_country_in_token('united st', self.countries, self.demonyms), (None, None))

    def test_no_relation(self):
        self.assertEqual(find_country_in_token('apple', self.countries, self.demonyms), (None, None))


class TestFindCountriesInDataframe(TestCase):

    def setUp(self):
        self.df_mock = pd.DataFrame({
            'Text1': ['uk', 'american', 'french'],
            'Text2': ['canada', 'british', 'Juice'],
        })
        self.countries = ['united kingdom', 'united states', 'canada', 'france']
        self.demonyms = {
            'british': 'united kingdom',
            'american': 'united states',
            'canadian': 'canada',
            'french': 'france'
        }

    def test_find_countries_in_dataframe(self):
        result_df = find_countries_in_dataframe(self.df_mock, self.countries, self.demonyms)
        expected_df = DataFrame({
            'Text1': [('united kingdom', 'uk'), ('united states', 'american'), ('france', 'french')],
            'Text2': [('canada', 'canada'), ('united kingdom', 'british'), (None, None)]
        })
        pd.testing.assert_frame_equal(result_df, expected_df)


class TestSortMultiIndexDataframe(TestCase):

    def setUp(self):
        arrays = [
            ['A', 'A', 'B', 'B'],
            ['col1', 'col2', 'col1', 'col2'],
        ]
        self.df_mock = pd.DataFrame(
            [
                [1, 2, 3, 12],
                [5, 6, 7, 8],
                [9, 10, 11, 4],
            ],
            columns=pd.MultiIndex.from_arrays(arrays, names=('top_level', 'bot_level'))
        )

    def test_sort_multiindex_dataframe(self):
        result_df = sort_multiindex_dataframe(self.df_mock, ['col1', 'col2'], sorting_col='col2', ascending=True)
        arrays_sorted = [
            ['A', 'A', 'B', 'B'],
            ['col1', 'col2', 'col1', 'col2'],
        ]
        expected_df = pd.DataFrame(
            [
                [1, 2, 11, 4],
                [5, 6, 7, 8],
                [9, 10, 3, 12]
            ],
            columns=pd.MultiIndex.from_arrays(arrays_sorted, names=('top_level', 'bot_level'))
        )
        pd.testing.assert_frame_equal(result_df, expected_df)
        result_df = sort_multiindex_dataframe(self.df_mock, ['col1', 'col2'], sorting_col='col1', ascending=False)
        arrays_sorted = [
            ['A', 'A', 'B', 'B'],
            ['col1', 'col2', 'col1', 'col2'],
        ]
        expected_df = pd.DataFrame(
            [
                [9, 10, 11, 4],
                [5, 6, 7, 8],
                [1, 2, 3, 12]
            ],
            columns=pd.MultiIndex.from_arrays(arrays_sorted, names=('top_level', 'bot_level'))
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()
