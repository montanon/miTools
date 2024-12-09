import json
import sys
from collections import defaultdict

import nltk

from mitools.nlp.textblob.en import suggest
from mitools.nlp.textblob.en.inflect import pluralize, singularize
from mitools.nlp.textblob.extractors import BaseNPExtractor, FastNPExtractor
from mitools.nlp.textblob.mixins import BlobComparableMixin, StringlikeMixin
from mitools.nlp.textblob.parsers import BaseParser, PatternParser
from mitools.nlp.textblob.sentiments import BaseSentimentAnalyzer, PatternAnalyzer
from mitools.nlp.textblob.taggers import BaseTagger, NLTKTagger
from mitools.nlp.textblob.tokenizers import (
    BaseTokenizer,
    SentenceTokenizer,
    WordTokenizer,
    word_tokenize,
)
from mitools.nlp.textblob.utils import PUNCTUATION_REGEX, CachedProperty, lowerstrip

BaseString = (str, bytes)
sentence_tokenize = SentenceTokenizer().itokenize
wordnet = nltk.corpus.wordnet
PorterStemmer = nltk.stem.porter.PorterStemmer()
LancasterStemmer = nltk.stem.lancaster.LancasterStemmer()
SnowballStemmer = nltk.stem.snowball.SnowballStemmer("english")


def penn_to_wordnet(tag):
    if tag in ("NN", "NNS", "NNP", "NNPS"):
        return wordnet.NOUN
    if tag in ("JJ", "JJR", "JJS"):
        return wordnet.ADJ
    if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return wordnet.VERB
    if tag in ("RB", "RBR", "RBS"):
        return wordnet.ADV
    return None


def validated_param(obj, name, base_class, default, base_class_name=None):
    base_class_name = base_class_name if base_class_name else base_class.__name__
    if obj is not None and not isinstance(obj, base_class):
        raise ValueError(f"{name} must be an instance of {base_class_name}")
    return obj or default


def initialize_models(
    obj, tokenizer, pos_tagger, np_extractor, analyzer, parser, classifier
):
    obj.tokenizer = validated_param(
        tokenizer,
        "tokenizer",
        base_class=(BaseTokenizer, nltk.tokenize.api.TokenizerI),
        default=BaseBlob.tokenizer,
        base_class_name="BaseTokenizer",
    )
    obj.np_extractor = validated_param(
        np_extractor,
        "np_extractor",
        base_class=BaseNPExtractor,
        default=BaseBlob.np_extractor,
    )
    obj.pos_tagger = validated_param(
        pos_tagger, "pos_tagger", BaseTagger, BaseBlob.pos_tagger
    )
    obj.analyzer = validated_param(
        analyzer, "analyzer", BaseSentimentAnalyzer, BaseBlob.analyzer
    )
    obj.parser = validated_param(parser, "parser", BaseParser, BaseBlob.parser)
    obj.classifier = classifier


class Word(str):
    def __new__(cls, string, pos_tag=None):
        return super().__new__(cls, string)

    def __init__(self, string, pos_tag=None):
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

    @CachedProperty
    def lemma(self):
        return self.lemmatize(pos=self.pos_tag)

    def lemmatize(self, pos=None):
        if pos is None:
            tag = wordnet.NOUN
        elif pos in wordnet._FILEMAP.keys():
            tag = pos
        else:
            tag = penn_to_wordnet(pos)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(self.string, tag)

    def stem(self, stemmer=PorterStemmer):
        return stemmer.stem(self.string)

    @CachedProperty
    def synsets(self):
        return self.get_synsets(pos=None)

    @CachedProperty
    def definitions(self):
        return self.define(pos=None)

    def get_synsets(self, pos=None):
        return wordnet.synsets(self.string, pos)

    def define(self, pos=None):
        return [syn.definition() for syn in self.get_synsets(pos=pos)]


class WordList(list):
    def __init__(self, collection):
        super().__init__([Word(w) for w in collection])

    def __str__(self):
        return super().__repr__()

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({super().__repr__()})"

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(item)
        else:
            return item

    def __getslice__(self, i, j):
        return self.__class__(super().__getslice__(i, j))

    def __setitem__(self, index, obj):
        if isinstance(obj, BaseString):
            super().__setitem__(index, Word(obj))
        else:
            super().__setitem__(index, obj)

    def count(self, strg, case_sensitive=False, *args, **kwargs):
        if not case_sensitive:
            return [word.lower() for word in self].count(strg.lower(), *args, **kwargs)
        return super().count(strg, *args, **kwargs)

    def append(self, obj):
        if isinstance(obj, BaseString):
            super().append(Word(obj))
        else:
            super().append(obj)

    def extend(self, iterable):
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


class BaseBlob(StringlikeMixin, BlobComparableMixin):
    np_extractor = FastNPExtractor()
    pos_tagger = NLTKTagger()
    tokenizer = WordTokenizer()
    analyzer = PatternAnalyzer()
    parser = PatternParser()

    def __init__(
        self,
        text,
        tokenizer=None,
        pos_tagger=None,
        np_extractor=None,
        analyzer=None,
        parser=None,
        classifier=None,
    ):
        if not isinstance(text, BaseString):
            raise TypeError(
                "The `text` argument passed to `__init__(text)` "
                f"must be a string, not {type(text)}"
            )
        self.raw = self.string = text
        self.stripped = lowerstrip(self.raw, all=True)
        initialize_models(
            self, tokenizer, pos_tagger, np_extractor, analyzer, parser, classifier
        )

    @CachedProperty
    def words(self):
        return WordList(word_tokenize(self.raw, include_punc=False))

    @CachedProperty
    def tokens(self):
        return WordList(self.tokenizer.tokenize(self.raw))

    def tokenize(self, tokenizer=None):
        t = tokenizer if tokenizer is not None else self.tokenizer
        return WordList(t.tokenize(self.raw))

    def parse(self, parser=None):
        p = parser if parser is not None else self.parser
        return p.parse(self.raw)

    def classify(self):
        if self.classifier is None:
            raise NameError("This blob has no classifier. Train one first!")
        return self.classifier.classify(self.raw)

    @CachedProperty
    def sentiment(self):
        return self.analyzer.analyze(self.raw)

    @CachedProperty
    def sentiment_assessments(self):
        return self.analyzer.analyze(self.raw, keep_assessments=True)

    @CachedProperty
    def polarity(self):
        return PatternAnalyzer().analyze(self.raw)[0]

    @CachedProperty
    def subjectivity(self):
        return PatternAnalyzer().analyze(self.raw)[1]

    @CachedProperty
    def noun_phrases(self):
        return WordList(
            [
                phrase.strip().lower()
                for phrase in self.np_extractor.extract(self.raw)
                if len(phrase) > 1
            ]
        )

    @CachedProperty
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
                for word, t in self.pos_tagger.tag(self)
                if not PUNCTUATION_REGEX.match(str(t))
            ]

    tags = pos_tags

    @CachedProperty
    def word_counts(self):
        counts = defaultdict(int)
        stripped_words = [lowerstrip(word) for word in self.words]
        for word in stripped_words:
            counts[word] += 1
        return counts

    @CachedProperty
    def np_counts(self):
        counts = defaultdict(int)
        for phrase in self.noun_phrases:
            counts[phrase] += 1
        return counts

    def ngrams(self, n=3):
        if n <= 0:
            return []
        grams = [
            WordList(self.words[i : i + n]) for i in range(len(self.words) - n + 1)
        ]
        return grams

    def correct(self):
        tokens = nltk.tokenize.regexp_tokenize(self.raw, r"\w+|[^\w\s]|\s")
        corrected = (Word(w).correct() for w in tokens)
        ret = "".join(corrected)
        return self.__class__(ret)

    def _cmpkey(self):
        return self.raw

    def _strkey(self):
        return self.raw

    def __hash__(self):
        return hash(self._cmpkey())

    def __add__(self, other):
        if isinstance(other, BaseString):
            return self.__class__(self.raw + other)
        elif isinstance(other, BaseBlob):
            return self.__class__(self.raw + other.raw)
        else:
            raise TypeError(
                f"Operands must be either strings or {self.__class__.__name__} objects"
            )

    def split(self, sep=None, maxsplit=sys.maxsize):
        return WordList(self._strkey().split(sep, maxsplit))


class TextBlob(BaseBlob):
    @CachedProperty
    def sentences(self):
        return self._create_sentence_objects()

    @CachedProperty
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
