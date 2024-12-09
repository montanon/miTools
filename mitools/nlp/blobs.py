from nltk.corpus import wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from mitools.nlp.typing import PosTag
from mitools.nlp.utils import penn_to_wordnet, pluralize, singularize, suggest
from mitools.utils.decorators import cached_property


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

    def lemmatize(self, pos: PosTag = None, lemmatizer: WordNetLemmatizer = None):
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
        return self.get_synsets(pos=None)

    @cached_property
    def definitions(self):
        return self.define(pos=None)

    def get_synsets(self, pos=None):
        return wordnet.synsets(self.string, pos)

    def define(self, pos=None):
        return [syn.definition() for syn in self.get_synsets(pos=pos)]
