from typing import Sequence, Union

from nltk.corpus import wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from mitools.nlp.typing import BaseString, PosTag
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
