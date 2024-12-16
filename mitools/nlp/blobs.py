import json
import sys
from collections import defaultdict
from typing import Callable, Literal, Sequence, Union

from nltk.corpus import wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize

from mitools.exceptions import ArgumentTypeError
from mitools.nlp.classifiers import BaseClassifier
from mitools.nlp.en.inflect import pluralize as en_pluralize
from mitools.nlp.en.inflect import singularize as en_singularize
from mitools.nlp.extractors import BaseNPExtractor, FastNPExtractor
from mitools.nlp.mixins import BlobComparableMixin, StringlikeMixin
from mitools.nlp.nlp_typing import BaseString, PosTag
from mitools.nlp.parsers import BaseParser, PatternParser
from mitools.nlp.sentiments import BaseSentimentAnalyzer, PatternAnalyzer
from mitools.nlp.taggers import BaseTagger, NLTKTagger
from mitools.nlp.tokenizers import BaseTokenizer, WordTokenizer
from mitools.nlp.utils import (
    penn_to_wordnet,
    sentence_tokenize,
    suggest,
    word_tokenize,
)
from mitools.utils.decorators import cached_property
from mitools.utils.helper_functions import PUNCTUATION_REGEX, lowerstrip


def singularize(word: BaseString, language: Literal["en", "other"] = "en") -> Callable:
    if language == "en":
        return en_singularize(word)
    raise NotImplementedError(
        f"Singularize is not implemented for language {language}."
    )


def pluralize(word: BaseString, language: Literal["en", "other"] = "en") -> Callable:
    if language == "en":
        return en_pluralize(word)
    raise NotImplementedError(f"Pluralize is not implemented for language {language}.")


class Word(str):
    def __new__(cls, string, pos_tag: PosTag = None):
        return super().__new__(cls, string)

    def __init__(self, string, pos_tag: PosTag = None):
        self.string = string
        self.pos_tag = pos_tag

    def __repr__(self):
        return repr(self.string)

    def __str__(self):
        return self.string

    def singularize(self):
        return Word(singularize(self.string))

    def pluralize(self):
        return Word(pluralize(self.string))

    def spellcheck(self):
        return suggest(self.string)

    def correct(self):
        return Word(self.spellcheck()[0][0])

    @cached_property
    def lemma(self):
        return self.lemmatize(pos=self.pos_tag)

    def lemmatize(
        self, pos: PosTag = None, lemmatizer: Union[WordNetLemmatizer, None] = None
    ):
        if pos is None:
            tag = wordnet.NOUN
        elif pos in wordnet._FILEMAP.keys():
            tag = pos
        else:
            tag = penn_to_wordnet(pos)
        if lemmatizer is None:
            lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(self.string, tag)

    def stem(self, stemmer: StemmerI = None):
        if stemmer is None:
            stemmer = PorterStemmer()
        return stemmer.stem(self.string)

    @cached_property
    def synsets(self):
        return self.get_synsets(pos_tag=None)

    @cached_property
    def definitions(self):
        return self.define(pos_tag=None)

    def get_synsets(self, pos_tag: PosTag = None):
        return wordnet.synsets(self.string, pos_tag)

    def define(self, pos_tag: PosTag = None):
        return [syn.definition() for syn in self.get_synsets(pos_tag=pos_tag)]


class WordList(list):
    def __init__(self, word_collection: Sequence[BaseString]):
        super().__init__([Word(w) for w in word_collection])

    def __str__(self):
        return super().__repr__()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({super().__repr__()})"

    def __getitem__(self, key: Union[int, slice]):
        item = super().__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(item)
        else:
            return item

    def __getslice__(self, i: int, j: int):
        return self.__class__(super().__getslice__(i, j))

    def __setitem__(self, index: int, obj: Union[BaseString, Word]):
        if isinstance(obj, BaseString):
            super().__setitem__(index, Word(obj))
        else:
            super().__setitem__(index, obj)

    def count(
        self,
        strg: Union[BaseString, Word],
        case_sensitive: bool = False,
        *args,
        **kwargs,
    ):
        if not case_sensitive:
            return [word.lower() for word in self].count(strg.lower(), *args, **kwargs)
        return super().count(strg, *args, **kwargs)

    def append(self, obj: Union[BaseString, Word]):
        if isinstance(obj, BaseString):
            super().append(Word(obj))
        else:
            super().append(obj)

    def extend(self, iterable: Sequence[Union[BaseString, Word]]):
        for e in iterable:
            self.append(e)

    def upper(self):
        return self.__class__([word.upper() for word in self])

    def lower(self):
        return self.__class__([word.lower() for word in self])

    def singularize(self):
        return self.__class__([word.singularize() for word in self])

    def pluralize(self):
        return self.__class__([word.pluralize() for word in self])

    def lemmatize(self):
        return self.__class__([word.lemmatize() for word in self])

    def stem(self, *args, **kwargs):
        return self.__class__([word.stem(*args, **kwargs) for word in self])

    def title(self):
        return self.__class__([word.title() for word in self])


class BaseBlob(StringlikeMixin, BlobComparableMixin):
    def __init__(
        self,
        text,
        tokenizer: BaseTokenizer = None,
        pos_tagger: BaseTagger = None,
        np_extractor: BaseNPExtractor = None,
        analyzer: BaseSentimentAnalyzer = None,
        parser: BaseParser = None,
        classifier: BaseClassifier = None,
    ):
        if not isinstance(text, (str, bytes)):
            raise ArgumentTypeError(
                "The `text` argument passed to `__init__(text)` "
                f"must be a string, not {type(text)}"
            )
        self.raw = text
        self.string = self.raw
        self.stripped = lowerstrip(self.raw, all=True)
        self.tokenizer = tokenizer if tokenizer is not None else WordTokenizer()
        self.pos_tagger = pos_tagger if pos_tagger is not None else NLTKTagger()
        self.np_extractor = (
            np_extractor if np_extractor is not None else FastNPExtractor()
        )
        self.analyzer = analyzer if analyzer is not None else PatternAnalyzer()
        self.parser = parser if parser is not None else PatternParser()
        self.classifier = classifier

    @cached_property
    def words(self):
        return WordList(word_tokenize(self.raw, include_punc=False))

    @cached_property
    def tokens(self):
        return WordList(self.tokenizer.tokenize(self.raw))

    def tokenize(self, tokenizer: BaseTokenizer = None):
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        return WordList(tokenizer.tokenize(self.raw))

    def parse(self, parser: BaseParser = None):
        p = parser if parser is not None else self.parser
        return p.parse(self.raw)

    def classify(self):
        if self.classifier is None:
            raise NameError("This blob has no classifier. Train one first!")
        return self.classifier.classify(self.raw)

    @cached_property
    def sentiment(self):
        return self.analyzer.analyze(self.raw)

    @cached_property
    def sentiment_assessments(self):
        return self.analyzer.analyze(self.raw, keep_assessments=True)

    @cached_property
    def polarity(self):
        return PatternAnalyzer().analyze(self.raw)[0]

    @cached_property
    def subjectivity(self):
        return PatternAnalyzer().analyze(self.raw)[1]

    @cached_property
    def noun_phrases(self):
        return WordList(
            [
                phrase.strip().lower()
                for phrase in self.np_extractor.extract(self.raw)
                if len(phrase) > 1
            ]
        )

    @cached_property
    def pos_tags(self):
        if isinstance(self, TextBlob):
            return [
                val
                for sublist in [s.pos_tags for s in self.sentences]
                for val in sublist
            ]
        else:
            return [
                (Word(str(word), pos_tag=t), str(t))
                for word, t in self.pos_tagger.tag_tokens(self)
                if not PUNCTUATION_REGEX.match(str(t))
            ]

    tags = pos_tags

    @cached_property
    def word_counts(self):
        counts = defaultdict(int)
        stripped_words = [lowerstrip(word) for word in self.words]
        for word in stripped_words:
            counts[word] += 1
        return counts

    @cached_property
    def np_counts(self):
        counts = defaultdict(int)
        for phrase in self.noun_phrases:
            counts[phrase] += 1
        return counts

    def ngrams(self, n: int = 3):
        if n <= 0:
            return []
        grams = [
            WordList(self.words[i : i + n]) for i in range(len(self.words) - n + 1)
        ]
        return grams

    def correct(self):
        tokens = regexp_tokenize(self.raw, r"\w+|[^\w\s]|\s")
        corrected = (Word(w).correct() for w in tokens)
        ret = "".join(corrected)
        return self.__class__(ret)

    def comparable_key(self):
        return self.raw

    def string_key(self):
        return self.raw

    def __hash__(self):
        return hash(self.comparable_key())

    def __add__(self, other: Union[BaseString, "BaseBlob"]):
        if isinstance(other, BaseString):
            return self.__class__(self.raw + other)
        elif isinstance(other, BaseBlob):
            return self.__class__(self.raw + other.raw)
        else:
            raise TypeError(
                f"Operands must be either strings or {self.__class__.__name__} objects"
            )

    def split(self, sep: str = None, maxsplit: int = sys.maxsize):
        return WordList(self._strkey().split(sep, maxsplit))


class TextBlob(BaseBlob):
    @cached_property
    def sentences(self):
        return self._create_sentence_objects()

    @cached_property
    def words(self):
        return WordList(word_tokenize(self.raw, include_punc=False))

    @property
    def raw_sentences(self):
        return [sentence.raw for sentence in self.sentences]

    @property
    def serialized(self):
        return [sentence.dict for sentence in self.sentences]

    def to_json(self, *args, **kwargs):
        return json.dumps(self.serialized, *args, **kwargs)

    @property
    def json(self):
        return self.to_json()

    def _create_sentence_objects(self):
        sentence_objects = []
        sentences = sentence_tokenize(self.raw)
        char_index = 0
        for sent in sentences:
            start_index = self.raw.index(sent, char_index)
            char_index += len(sent)
            end_index = start_index + len(sent)
            s = Sentence(
                sent,
                start_index=start_index,
                end_index=end_index,
                tokenizer=self.tokenizer,
                np_extractor=self.np_extractor,
                pos_tagger=self.pos_tagger,
                analyzer=self.analyzer,
                parser=self.parser,
                classifier=self.classifier,
            )
            sentence_objects.append(s)
        return sentence_objects


class Sentence(BaseBlob):
    def __init__(
        self,
        sentence: BaseString,
        start_index: int = 0,
        end_index: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(sentence, *args, **kwargs)
        self.start = self.start_index = start_index
        self.end = self.end_index = end_index or len(sentence) - 1

    @property
    def dict(self):
        return {
            "raw": self.raw,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "stripped": self.stripped,
            "noun_phrases": self.noun_phrases,
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
        }


class Blobber:
    def __init__(
        self,
        tokenizer: BaseTokenizer = None,
        pos_tagger: BaseTagger = None,
        np_extractor: BaseNPExtractor = None,
        analyzer: BaseSentimentAnalyzer = None,
        parser: BaseParser = None,
        classifier: BaseClassifier = None,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else WordTokenizer()
        self.pos_tagger = pos_tagger if pos_tagger is not None else NLTKTagger()
        self.np_extractor = (
            np_extractor if np_extractor is not None else FastNPExtractor()
        )
        self.analyzer = analyzer if analyzer is not None else PatternAnalyzer()
        self.parser = parser if parser is not None else PatternParser()
        self.classifier = classifier

    def __call__(self, text):
        return TextBlob(
            text,
            tokenizer=self.tokenizer,
            pos_tagger=self.pos_tagger,
            np_extractor=self.np_extractor,
            analyzer=self.analyzer,
            parser=self.parser,
            classifier=self.classifier,
        )

    def __repr__(self):
        classifier_name = (
            self.classifier.__class__.__name__ + "()" if self.classifier else "None"
        )
        return (
            f"Blobber(tokenizer={self.tokenizer.__class__.__name__}(), "
            f"pos_tagger={self.pos_tagger.__class__.__name__}(), "
            f"np_extractor={self.np_extractor.__class__.__name__}(), "
            f"analyzer={self.analyzer.__class__.__name__}(), "
            f"parser={self.parser.__class__.__name__}(), "
            f"classifier={classifier_name})"
        )

    def __str__(self):
        return self.__repr__()
