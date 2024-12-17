from abc import ABCMeta, abstractmethod
from typing import Iterator, Sequence, Tuple, Union

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.api import StemmerI
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

from mitools.nlp.nlp_typing import PosTag
from mitools.nlp.tags_translator import translate_tags


class BaseLemmatizer(ABCMeta):
    @abstractmethod
    def lemmatize(self, token: str, pos: PosTag = "n") -> Sequence[str]:
        pass

    def lemmatize_tokens(
        self, tokens: Union[Sequence[str], Sequence[Tuple[str, PosTag]]]
    ) -> Sequence[str]:
        return [
            self.lemmatize(token[0], token[1])
            if isinstance(token, tuple)
            else self.lemmatize(token)
            for token in tokens
        ]

    def ilemmatize_tokens(
        self, tokens: Union[Sequence[str], Sequence[Tuple[str, PosTag]]]
    ) -> Iterator[str]:
        return (self.lemmatize(token[0], token[1]) for token in tokens)


class WordnetLemmatizer(BaseLemmatizer):
    _instance = None
    _lemmatizer = WordNetLemmatizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def lemmatize(self, token: str, pos: PosTag = "n") -> Sequence[str]:
        return self._lemmatizer.lemmatize(token, pos)


class NLTKLemmatizer(BaseLemmatizer):
    _instance = None
    _lemmatizer = WordNetLemmatizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def lemmatize(self, token: Union[str, Tuple[str, PosTag]]) -> Sequence[str]:
        if isinstance(token, tuple):
            word, tag = token
            return self._lemmatizer.lemmatize(
                word, pos=translate_tags(tag, "nltk", "wordnet")
            )
        return self._lemmatizer.lemmatize(token)


class BaseStemmer(StemmerI, metaclass=ABCMeta):
    @abstractmethod
    def stem(self, token: str) -> Sequence[str]:
        pass

    def stem_tokens(self, tokens: Sequence[str]) -> Sequence[str]:
        return [self.stem(token) for token in tokens]

    def istem_tokens(self, tokens: Sequence[str]) -> Iterator[str]:
        return (self.stem(token) for token in tokens)


class PorterStemmer(BaseStemmer):
    _instance = None
    _stemmer = PorterStemmer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stem(self, token: str) -> Sequence[str]:
        return self._stemmer.stem(token)


class SpanishStemmer(BaseStemmer):
    _instance = None
    _stemmer = SpanishStemmer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stem(self, token: str) -> Sequence[str]:
        return self._stemmer.stem(token)


class EnglishStemmer(BaseStemmer):
    _instance = None
    _stemmer = EnglishStemmer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stem(self, token: str) -> Sequence[str]:
        return self._stemmer.stem(token)
