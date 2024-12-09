from typing import Callable, Iterable, Tuple

import nltk

from mitools.nlp.typing import BaseString
from mitools.nlp.utils import basic_extractor, get_words_from_corpus
from mitools.utils.decorators import cached_property


class BaseClassifier:
    def __init__(
        self,
        train_set: Iterable[BaseString],
        feature_extractor: Callable = basic_extractor,
        **kwargs,
    ):
        self.format_kwargs = kwargs
        self.feature_extractor = feature_extractor
        self.train_set = train_set
        self.bag_of_words = get_words_from_corpus(self.train_set)
        self.train_features = None

    @cached_property
    def classifier(self):
        raise NotImplementedError('Must implement the "classifier" property.')

    def classify(self, text: BaseString):
        raise NotImplementedError('Must implement the "classify" method.')

    def train(self, labeled_features: Iterable[Tuple[BaseString, bool]]):
        raise NotImplementedError('Must implement the "train" method.')

    def labels(self):
        raise NotImplementedError('Must implement the "labels" method.')

    def extract_features(self, text: BaseString):
        try:
            return self.feature_extractor(text, self.bag_of_words)
        except (TypeError, AttributeError):
            return self.feature_extractor(text)


class NLTKClassifier(BaseClassifier):
    nltk_class = None

    def __init__(
        self,
        train_set: Iterable[BaseString],
        feature_extractor: Callable = None,
        **kwargs,
    ):
        super().__init__(train_set, feature_extractor, **kwargs)
        self.train_features = [(self.extract_features(d), c) for d, c in self.train_set]

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"<{class_name} trained on {len(self.train_set)} instances>"

    @cached_property
    def classifier(self):
        try:
            return self.train()
        except AttributeError as error:  # nltk_class has not been defined
            raise ValueError(
                "NLTKClassifier must have a nltk_class" " variable that is not None."
            ) from error

    def train(self, *args, **kwargs):
        try:
            self.classifier = self.nltk_class.train(
                self.train_features, *args, **kwargs
            )
            return self.classifier
        except AttributeError as error:
            raise ValueError(
                "NLTKClassifier must have a nltk_class" " variable that is not None."
            ) from error

    def labels(self):
        return self.classifier.labels()

    def classify(self, text: BaseString):
        text_features = self.extract_features(text)
        return self.classifier.classify(text_features)

    def accuracy(self, test_set: Iterable[Tuple[BaseString, bool]]):
        test_data = test_set
        test_features = [(self.extract_features(d), c) for d, c in test_data]
        return nltk.classify.accuracy(self.classifier, test_features)

    def update(self, new_data: Iterable[Tuple[BaseString, bool]], *args, **kwargs):
        self.train_set += new_data
        self.bag_of_words.update(get_words_from_corpus(new_data))
        self.train_features = [(self.extract_features(d), c) for d, c in self.train_set]
        try:
            self.classifier = self.nltk_class.train(
                self.train_features, *args, **kwargs
            )
        except AttributeError as error:  # Descendant has not defined nltk_class
            raise ValueError(
                "NLTKClassifier must have a nltk_class" " variable that is not None."
            ) from error
        return True
