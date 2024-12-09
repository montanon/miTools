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
