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
