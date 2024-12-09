from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Tuple, Union
from xml.etree import ElementTree

import nltk

from mitools.nlp.definitions import (
    ADJECTIVE,
    ADVERB,
    EMOTICONS,
    IRONY,
    MOOD,
    NOUN,
    PUNCTUATION,
    RE_SYNSET,
    VERB,
)
from mitools.nlp.en import en_sentiment as pattern_sentiment
from mitools.nlp.typing import BaseString, PosTag, SentimentType
from mitools.nlp.utils import (
    avg,
    default_feature_extractor,
    find_tokens,
    word_tokenize,
)
from mitools.utils.helper_objects import LazyDict


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


class Score(tuple):
    def __new__(
        self, polarity: float, subjectivity: float, assessments: Iterable[float] = None
    ):
        if assessments is None:
            assessments = []
        return tuple.__new__(self, [polarity, subjectivity])

    def __init__(
        self, polarity: float, subjectivity: float, assessments: Iterable[float] = None
    ):
        if assessments is None:
            assessments = []
        self.assessments = assessments


class Sentiment(LazyDict):
    def __init__(
        self,
        path: Path = Path(""),
        language: str = None,
        synset: str = None,
        confidence: float = None,
        **kwargs,
    ):
        self._path = path  # XML file path.
        self._language = None  # XML language attribute ("en", "fr", ...)
        self._confidence = None  # XML confidence attribute threshold (>=).
        self._synset = synset  # XML synset attribute ("wordnet_id", "cornetto_id", ...)
        self._synsets = {}  # {"a-01123879": (1.0, 1.0, 1.0)}
        self.labeler = {}  # {"dammit": "profanity"}
        self.tokenizer = kwargs.get("tokenizer", find_tokens)
        self.negations = kwargs.get("negations", ("no", "not", "n't", "never"))
        self.modifiers = kwargs.get("modifiers", ("RB",))
        self.modifier = kwargs.get("modifier", lambda w: w.endswith("ly"))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @property
    def confidence(self):
        return self._confidence

    def load(self, file_path: Path = None):
        if not file_path:
            file_path = self._path
        if not file_path.exists():
            return
        word_scores = {}
        synset_scores = {}
        word_labels = {}
        xml_tree = ElementTree.parse(file_path)
        root = xml_tree.getroot()
        for word_elem in root.findall("word"):
            if self._confidence is None or self._confidence <= float(
                word_elem.attrib.get("confidence", 0.0)
            ):
                (
                    word_form,
                    part_of_speech,
                    polarity,
                    subjectivity,
                    intensity,
                    label,
                    synset_id,
                ) = (
                    word_elem.attrib.get("form"),
                    word_elem.attrib.get("pos"),
                    word_elem.attrib.get("polarity", 0.0),
                    word_elem.attrib.get("subjectivity", 0.0),
                    word_elem.attrib.get("intensity", 1.0),
                    word_elem.attrib.get("label"),
                    word_elem.attrib.get(self._synset),  # wordnet_id, cornetto_id, ...
                )

                sentiment_scores = (
                    float(polarity),
                    float(subjectivity),
                    float(intensity),
                )

                if word_form:
                    word_scores.setdefault(word_form, {}).setdefault(
                        part_of_speech, []
                    ).append(sentiment_scores)
                if word_form and label:
                    word_labels[word_form] = label
                if synset_id:
                    synset_scores.setdefault(synset_id, []).append(sentiment_scores)

        self._language = root.attrib.get("language", self._language)
        for word in word_scores:
            word_scores[word] = dict(
                (pos, [avg(each) for each in zip(*scores)])
                for pos, scores in word_scores[word].items()
            )
        for word, pos_scores in list(word_scores.items()):
            word_scores[word][None] = [avg(each) for each in zip(*pos_scores.values())]
        for synset_id, scores in synset_scores.items():
            synset_scores[synset_id] = [avg(each) for each in zip(*scores)]
        dict.update(self, word_scores)
        dict.update(self.labeler, word_labels)
        dict.update(self._synsets, synset_scores)

    def synset(
        self, synset_id: str, part_of_speech: PosTag = ADJECTIVE
    ) -> Tuple[float, float]:
        padded_id = str(synset_id).zfill(8)
        if not padded_id.startswith(("n-", "v-", "a-", "r-")):
            if part_of_speech == NOUN:
                padded_id = "n-" + padded_id
            if part_of_speech == VERB:
                padded_id = "v-" + padded_id
            if part_of_speech == ADJECTIVE:
                padded_id = "a-" + padded_id
            if part_of_speech == ADVERB:
                padded_id = "r-" + padded_id
        if dict.__len__(self) == 0:
            self.load()
        return tuple(self._synsets.get(padded_id, (0.0, 0.0))[:2])

    def __call__(
        self, input_text: BaseString, negation: bool = True, **kwargs
    ) -> Score:
        def calculate_weighted_average(
            sentiment_assessments: Iterable[Tuple[List[str], float]],
            weighting_function: Callable = lambda w: 1,
        ):
            sum_weighted_scores, total_weight = 0, 0
            for words, score in sentiment_assessments:
                weight = weighting_function(words)
                sum_weighted_scores += weight * score
                total_weight += weight
            return sum_weighted_scores / float(total_weight or 1)

        if hasattr(input_text, "gloss"):
            assessments = [
                (input_text.synonyms[0],)
                + self.synset(input_text.id, pos=input_text.pos)
                + (None,)
            ]
        elif (
            isinstance(input_text, BaseString)
            and RE_SYNSET.match(input_text)
            and hasattr(input_text, "synonyms")
        ):
            assessments = [
                (input_text.synonyms[0],)
                + self.synset(input_text.id, pos=input_text.pos)
                + (None,)
            ]
        elif isinstance(input_text, BaseString):
            assessments = self.assessments(
                (
                    (word.lower(), None)
                    for word in " ".join(self.tokenizer(input_text)).split()
                ),
                negation,
            )
        elif hasattr(input_text, "sentences"):
            assessments = self.assessments(
                (
                    (word.lemma or word.string.lower(), word.pos[:2])
                    for word in chain.from_iterable(input_text)
                ),
                negation,
            )
        elif hasattr(input_text, "lemmata"):
            assessments = self.assessments(
                (
                    (word.lemma or word.string.lower(), word.pos[:2])
                    for word in input_text.words
                ),
                negation,
            )
        elif hasattr(input_text, "lemma"):
            assessments = self.assessments(
                ((input_text.lemma or input_text.string.lower(), input_text.pos[:2]),),
                negation,
            )
        elif hasattr(input_text, "terms"):
            assessments = self.assessments(
                chain.from_iterable(
                    ((word, None), (None, None)) for word in input_text
                ),
                negation,
            )
            kwargs.setdefault("weight", lambda w: input_text.terms[w[0]])
        elif isinstance(input_text, dict):
            assessments = self.assessments(
                chain.from_iterable(
                    ((word, None), (None, None)) for word in input_text
                ),
                negation,
            )
            kwargs.setdefault("weight", lambda w: input_text[w[0]])
        elif isinstance(input_text, list):
            assessments = self.assessments(
                ((word, None) for word in input_text), negation
            )
        else:
            assessments = []

        weighting_function = kwargs.get("weight", lambda w: 1)
        return Score(
            polarity=calculate_weighted_average(
                [
                    (words, polarity)
                    for words, polarity, subjectivity, label in assessments
                ],
                weighting_function,
            ),
            subjectivity=calculate_weighted_average(
                [
                    (words, subjectivity)
                    for words, polarity, subjectivity, label in assessments
                ],
                weighting_function,
            ),
            assessments=assessments,
        )

    def assessments(
        self, words: Iterable[Tuple[str, str]], negation: bool = True
    ) -> List[Tuple[List[str], float, float, str]]:
        if words is None:
            words = []
        assessments_list = []
        modifier = None  # Preceding modifier (i.e., adverb or adjective).
        negation_word = None  # Preceding negation (e.g., "not beautiful").
        for word, pos_tag in words:
            # Only assess known words, preferably by part-of-speech tag.
            # Including unknown words (polarity 0.0 and subjectivity 0.0) lowers the average.
            if word is None:
                continue
            if word in self and pos_tag in self[word]:
                polarity, subjectivity, intensity = self[word][pos_tag]
                # Known word not preceded by a modifier ("good").
                if modifier is None:
                    assessments_list.append(
                        dict(
                            words=[word],
                            polarity=polarity,
                            subjectivity=subjectivity,
                            intensity=intensity,
                            negation_factor=1,
                            label=self.labeler.get(word),
                        )
                    )
                # Known word preceded by a modifier ("really good").
                if modifier is not None:
                    assessments_list[-1]["words"].append(word)
                    assessments_list[-1]["polarity"] = max(
                        -1.0, min(polarity * assessments_list[-1]["intensity"], +1.0)
                    )
                    assessments_list[-1]["subjectivity"] = max(
                        -1.0,
                        min(subjectivity * assessments_list[-1]["intensity"], +1.0),
                    )
                    assessments_list[-1]["intensity"] = intensity
                    assessments_list[-1]["label"] = self.labeler.get(word)
                # Known word preceded by a negation ("not really good").
                if negation_word is not None:
                    assessments_list[-1]["words"].insert(0, negation_word)
                    assessments_list[-1]["intensity"] = (
                        1.0 / assessments_list[-1]["intensity"]
                    )
                    assessments_list[-1]["negation_factor"] = -1
                # Known word may be a negation.
                # Known word may be modifying the next word (i.e., it is a known adverb).
                modifier = None
                negation_word = None
                if (
                    pos_tag
                    and pos_tag in self.modifiers
                    or any(map(self[word].__contains__, self.modifiers))
                ):
                    modifier = (word, pos_tag)
                if negation and word in self.negations:
                    negation_word = word
            else:
                # Unknown word may be a negation ("not good").
                if negation and word in self.negations:
                    negation_word = word
                # Unknown word. Retain negation across small words ("not a good").
                elif negation_word and len(word.strip("'")) > 1:
                    negation_word = None
                # Unknown word may be a negation preceded by a modifier ("really not good").
                if (
                    negation_word is not None
                    and modifier is not None
                    and (pos_tag in self.modifiers or self.modifier(modifier[0]))
                ):
                    assessments_list[-1]["words"].append(negation_word)
                    assessments_list[-1]["negation_factor"] = -1
                    negation_word = None
                # Unknown word. Retain modifier across small words ("really is a good").
                elif modifier and len(word) > 2:
                    modifier = None
                # Exclamation marks boost previous word.
                if word == "!" and len(assessments_list) > 0:
                    assessments_list[-1]["words"].append("!")
                    assessments_list[-1]["polarity"] = max(
                        -1.0, min(assessments_list[-1]["polarity"] * 1.25, +1.0)
                    )
                # Exclamation marks in parentheses indicate sarcasm.
                if word == "(!)":
                    assessments_list.append(
                        dict(
                            words=[word],
                            polarity=0.0,
                            subjectivity=1.0,
                            intensity=1.0,
                            negation_factor=1,
                            label=IRONY,
                        )
                    )
                # EMOTICONS: {("grin", +1.0): set((":-D", ":D"))}
                if (
                    word.isalpha() is False
                    and len(word) <= 5
                    and word not in PUNCTUATION
                ):  # speedup
                    for (
                        emoticon_type,
                        emoticon_polarity,
                    ), emoticons in EMOTICONS.items():
                        if word in map(lambda e: e.lower(), emoticons):
                            assessments_list.append(
                                dict(
                                    words=[word],
                                    polarity=emoticon_polarity,
                                    subjectivity=1.0,
                                    intensity=1.0,
                                    negation_factor=1,
                                    label=MOOD,
                                )
                            )
                            break
        final_assessments = []
        for assessment in assessments_list:
            words = assessment["words"]
            polarity = assessment["polarity"]
            subjectivity = assessment["subjectivity"]
            negation_factor = assessment["negation_factor"]
            label = assessment["label"]
            # "not good" = slightly bad, "not bad" = slightly good.
            final_assessments.append(
                (
                    words,
                    polarity * -0.5 if negation_factor < 0 else polarity,
                    subjectivity,
                    label,
                )
            )
        return final_assessments

    def annotate(
        self,
        word: BaseString,
        pos: PosTag = None,
        polarity: float = 0.0,
        subjectivity: float = 0.0,
        intensity: float = 1.0,
        label: str = None,
    ):
        word_scores = self.setdefault(word, {})
        sentiment_tuple = (polarity, subjectivity, intensity)
        word_scores[pos] = word_scores[None] = sentiment_tuple
        if label:
            self.labeler[word] = label
