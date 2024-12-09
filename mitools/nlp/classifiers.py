from typing import Callable, Iterable, Tuple

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
