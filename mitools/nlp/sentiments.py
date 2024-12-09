from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Callable, Dict, Iterable, Literal, Tuple, Union

import nltk

from mitools.nlp.typing import BaseString, SentimentType
from mitools.nlp.utils import (
    default_feature_extractor,
    pattern_sentiment,
    word_tokenize,
)


class BaseSentimentAnalyzer(ABCMeta):
    def __init__(self, kind: Literal["ds", "co"] = "ds"):
        self.kind = kind
        self._trained = False

    def train(self):
        self._trained = True

    @abstractmethod
    def analyze(self, text: str) -> Union[Tuple, float, Dict]:
        if not self._trained:
            self.train()
        return None


class PatternAnalyzer(BaseSentimentAnalyzer):
    def analyze(
        self, text: BaseString, keep_assessments: bool = False
    ) -> SentimentType:
        if keep_assessments:
            Sentiment = namedtuple(
                "Sentiment", ["polarity", "subjectivity", "assessments"]
            )
            assessments = pattern_sentiment(text).assessments
            polarity, subjectivity = pattern_sentiment(text)
            return Sentiment(polarity, subjectivity, assessments)

        else:
            Sentiment = namedtuple("Sentiment", ["polarity", "subjectivity"])
            return Sentiment(*pattern_sentiment(text))


class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
    RETURN_TYPE = namedtuple("Sentiment", ["classification", "p_pos", "p_neg"])

    def __init__(self, feature_extractor: Callable = None):
        super().__init__()
        self._classifier = None
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else default_feature_extractor
        )

    def train(self):
        super().train()
        neg_ids = nltk.corpus.movie_reviews.fileids("neg")
        pos_ids = nltk.corpus.movie_reviews.fileids("pos")
        neg_feats = [
            (
                self.feature_extractor(nltk.corpus.movie_reviews.words(fileids=[f])),
                "neg",
            )
            for f in neg_ids
        ]
        pos_feats = [
            (
                self.feature_extractor(nltk.corpus.movie_reviews.words(fileids=[f])),
                "pos",
            )
            for f in pos_ids
        ]
        train_data = neg_feats + pos_feats
        self._classifier = nltk.classify.NaiveBayesClassifier.train(train_data)

    def analyze(self, text: BaseString) -> SentimentType:
        super().analyze(text)
        word_tokens = word_tokenize(text, include_punc=False)
        filtered_words = (word.lower() for word in word_tokens if len(word) >= 3)
        word_features = self.feature_extractor(filtered_words)
        probability_distribution = self._classifier.prob_classify(word_features)
        return self.RETURN_TYPE(
            classification=probability_distribution.max(),
            p_pos=probability_distribution.prob("pos"),
            p_neg=probability_distribution.prob("neg"),
        )
