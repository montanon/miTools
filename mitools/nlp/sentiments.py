from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import nltk
import torch
from transformers import pipeline

from mitools.nlp.en import en_sentiment as pattern_sentiment
from mitools.nlp.nlp_typing import SentimentType
from mitools.nlp.tokenizers import BaseTokenizer, SentenceTokenizer
from mitools.nlp.utils import default_feature_extractor, word_tokenize


@dataclass
class SentimentResult:
    polarity: float  # Range from -1 to 1
    confidence: float  # Range from 0 to 1
    label: Optional[str] = None  # For categorical outputs (e.g. "positive", "negative")
    subjectivity: Optional[float] = None  # For Pattern analyzer
    probabilities: Optional[Dict[str, float]] = None  # For probability distributions
    raw_output: Optional[Dict] = None  # Store original model output

    @classmethod
    def from_huggingface(cls, result: Dict) -> "SentimentResult":
        label_polarity = 1 if result[0]["label"] == "POSITIVE" else -1
        return cls(
            polarity=label_polarity * result[0]["score"],
            confidence=result[0]["score"],
            label=result[0]["label"],
            raw_output=result,
        )

    @classmethod
    def from_naive_bayes(cls, result: Tuple) -> "SentimentResult":
        classification, p_pos, p_neg = result
        polarity = p_pos - p_neg  # Convert probabilities to [-1, 1] range
        return cls(
            polarity=polarity,
            confidence=max(p_pos, p_neg),
            label=classification,
            probabilities={"positive": p_pos, "negative": p_neg},
            raw_output={
                "classification": classification,
                "p_pos": p_pos,
                "p_neg": p_neg,
            },
        )


class BaseSentimentAnalyzer(ABC):
    def train(self):
        self._trained = True

    @abstractmethod
    def analyze(self, text: Union[str, str]) -> SentimentResult:
        if not self._trained:
            self.train()
        return None

    def analyze_sentences(
        self,
        sequence: Union[str, Sequence[str]],
        tokenizer: Union[BaseTokenizer, None] = None,
    ) -> Sequence[SentimentResult]:
        tokenizer = tokenizer if tokenizer is not None else SentenceTokenizer()
        if isinstance(sequence, str):
            sequence = tokenizer.tokenize(sequence)
        return [self.analyze(sent) for sent in sequence]


class HuggingFaceAnalyzer(BaseSentimentAnalyzer):
    def __init__(
        self,
        model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Union[int, str, torch.device] = None,
    ):
        super().__init__()
        self.model = pipeline("sentiment-analysis", model=model, device=device)

    def analyze(self, text: str) -> SentimentResult:
        result = self.model(text)
        return SentimentResult.from_huggingface(result)


class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, feature_extractor: Union[Callable, None] = None):
        super().__init__()
        self._classifier = None
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else default_feature_extractor
        )
        self._trained = False

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

    def analyze(self, text: str) -> SentimentResult:
        super().analyze(text)
        word_tokens = word_tokenize(text, include_punctuation=False)
        filtered_words = (word.lower() for word in word_tokens if len(word) >= 3)
        word_features = self.feature_extractor(filtered_words)
        probability_distribution = self._classifier.prob_classify(word_features)

        classification = probability_distribution.max()
        p_pos = probability_distribution.prob("pos")
        p_neg = probability_distribution.prob("neg")

        return SentimentResult(
            polarity=p_pos - p_neg,  # Convert to [-1, 1] range
            confidence=max(p_pos, p_neg),
            label=classification,
            probabilities={"positive": p_pos, "negative": p_neg},
            raw_output={
                "classification": classification,
                "p_pos": p_pos,
                "p_neg": p_neg,
            },
        )
