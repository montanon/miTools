import pickle
from typing import Literal, Sequence

from nltk.corpus import stopwords

from mitools.nlp.definitions import TOKENS
from mitools.nlp.typing import BaseString, PosTag


class StopwordsManager:
    def __init__(self, language="english"):
        self.language = language
        self._words = set(stopwords.words(language))

    def add_stopword(self, word):
        self._words.add(word)

    def add_stopwords(self, words):
        self._words.update(words)

    def remove_stopword(self, word):
        self._words.discard(word)

    def remove_stopwords(self, words):
        self._words.difference_update(words)

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    @property
    def words(self):
        return list(self._words)


class TaggedString(str):
    def __new__(
        self,
        string: BaseString,
        tags: Sequence[PosTag] = None,
        language: Literal["en"] = None,
    ):
        if tags is None:
            tags = ["word"]
        if isinstance(string, str) and hasattr(string, "tags"):
            tags, language = string.tags, string.language
        if isinstance(string, list):
            string = [
                [[x.replace("/", "&slash;") for x in token] for token in s]
                for s in string
            ]
            string = "\n".join(" ".join("/".join(token) for token in s) for s in string)
        s = str.__new__(self, string)
        s.tags = list(tags)
        s.language = language
        return s

    def split(self, sep: str = TOKENS):
        if sep != TOKENS:
            return str.split(self, sep)
        if len(self) == 0:
            return []
        return [
            [
                [x.replace("&slash;", "/") for x in token.split("/")]
                for token in sentence.split(" ")
            ]
            for sentence in str.split(self, "\n")
        ]
