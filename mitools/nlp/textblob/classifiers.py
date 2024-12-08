from itertools import chain

import mitools.nlp.textblob.formats as formats
from mitools.nlp.textblob.tokenizers import word_tokenize
from mitools.nlp.textblob.utils import CachedProperty, is_filelike, strip_punctuation

BaseString = (str, bytes)


class FormatError(Exception):
    pass


def get_words_from_dataset(dataset):
    def tokenize(words):
        if isinstance(words, BaseString):
            return word_tokenize(words, include_punctuation=False)
        return words

    all_words = chain.from_iterable(tokenize(words) for words, _ in dataset)
    return set(all_words)


def get_document_tokens(document):
    if isinstance(document, BaseString):
        tokens = set(
            strip_punctuation(word, all=False)
            for word in word_tokenize(document, include_punctuation=False)
        )
    else:
        tokens = set(strip_punctuation(word, all=False) for word in document)
    return tokens


def basic_extractor(document, train_set):
    try:
        zero_item = next(iter(train_set))
    except StopIteration:
        return {}
    if isinstance(zero_item, BaseString):
        word_features = [word for word in chain([zero_item], train_set)]
    else:
        try:
            assert isinstance(zero_item[0], BaseString)
            word_features = get_words_from_dataset(chain([zero_item], train_set))
        except Exception as error:
            raise ValueError("train_set is probably malformed.") from error
    tokens = get_document_tokens(document)
    features = dict((f"contains({word})", (word in tokens)) for word in word_features)
    return features


class BaseClassifier:
    def __init__(
        self, train_set, feature_extractor=basic_extractor, format=None, **kwargs
    ):
        self.format_kwargs = kwargs
        self.feature_extractor = feature_extractor
        if is_filelike(train_set):
            self.train_set = self.read_data(train_set, format)
        else:
            self.train_set = train_set
        self.bag_of_words = get_words_from_dataset(self.train_set)
        self.train_features = None

    def read_data(self, dataset, format=None):
        if not format:
            format_class = formats.detect(dataset)
            if not format_class:
                raise FormatError(
                    "Could not automatically detect format for the given "
                    "data source."
                )
        else:
            registry = formats.get_registry()
            if format not in registry.keys():
                raise ValueError(f"Format {format} is not supported.")
            format_class = registry[format]
        return format_class(dataset, **self.format_kwargs).to_iterable()

    @CachedProperty
    def classifier(self):
        raise NotImplementedError('Must implement the "classifier" property.')

    def classify(self, text: str):
        raise NotImplementedError('Must implement the "classify" method.')

    def train(self, labeled_features):
        raise NotImplementedError('Must implement the "train" method.')

    def labels(self):
        raise NotImplementedError('Must implement the "labels" method.')

    def extract_features(self, text: str):
        try:
            return self.feature_extractor(text, self.bag_of_words)
        except (TypeError, AttributeError):
            return self.feature_extractor(text)
