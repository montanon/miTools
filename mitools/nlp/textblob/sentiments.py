from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Dict, Literal, Tuple, Union

import nltk

from mitools.nlp.textblob.tokenizers import word_tokenize


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
    RETURN_TYPE = namedtuple("Sentiment", ["polarity", "subjectivity"])

    def analyze(self, text, keep_assessments=False):
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


def default_feature_extractor(words):
    return dict((word, True) for word in words)


class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
    RETURN_TYPE = namedtuple("Sentiment", ["classification", "p_pos", "p_neg"])

    def __init__(self, feature_extractor=default_feature_extractor):
        super().__init__()
        self.classifier = None
        self.feature_extractor = feature_extractor

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

    def analyze(self, text):
        super().analyze(text)
        tokens = word_tokenize(text, include_punc=False)
        filtered = (t.lower() for t in tokens if len(t) >= 3)
        feats = self.feature_extractor(filtered)
        prob_dist = self._classifier.prob_classify(feats)
        return self.RETURN_TYPE(
            classification=prob_dist.max(),
            p_pos=prob_dist.prob("pos"),
            p_neg=prob_dist.prob("neg"),
        )
