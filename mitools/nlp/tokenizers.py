import re
from abc import ABCMeta, abstractmethod
from typing import Iterator, List, Sequence, Tuple

import nltk
from nltk.tokenize.api import TokenizerI

from mitools.nlp.string_utils import strip_punctuation
from mitools.nlp.typing import BaseString


class BaseTokenizer(TokenizerI, metaclass=ABCMeta):
    @abstractmethod
    def tokenize(self, text: BaseString) -> Sequence[BaseString]:
        pass

    def itokenize(self, text: BaseString, *args, **kwargs) -> Iterator[BaseString]:
        return (t for t in self.tokenize(text, *args, **kwargs))

    def span_tokenize(self, text: BaseString) -> Iterator[Tuple[int, int]]:
        return TokenizerI.span_tokenize(self, text)

    def tokenize_sents(self, strings: List[BaseString]) -> List[List[BaseString]]:
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(
        self, strings: List[BaseString]
    ) -> Iterator[List[Tuple[int, int]]]:
        for s in strings:
            yield list(self.span_tokenize(s))


class WordTokenizer(BaseTokenizer):
    _instance = None
    _tokenizer = nltk.tokenize.word_tokenize

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(self, text: BaseString, include_punctuation: bool = True):
        tokens = self._tokenizer(text)
        if include_punctuation:
            return tokens
        else:
            return [
                word if word.startswith("'") else strip_punctuation(word, all=False)
                for word in tokens
                if strip_punctuation(word, all=False)
            ]


class SentenceTokenizer(BaseTokenizer):
    _instance = None
    _tokenizer = nltk.tokenize.sent_tokenize

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(self, text: BaseString) -> Sequence[BaseString]:
        return self._tokenizer(text)


class RegexpTokenizer(BaseTokenizer):
    _instances = {}

    def __new__(
        cls,
        pattern: str,
        gaps: bool = False,
        discard_empty: bool = True,
        flags: int = re.UNICODE | re.MULTILINE | re.DOTALL,
    ):
        key = (pattern, gaps, discard_empty, flags)
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._tokenizer = nltk.tokenize.RegexpTokenizer(
                pattern=pattern, gaps=gaps, discard_empty=discard_empty, flags=flags
            )
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        pattern: str,
        gaps: bool = False,
        discard_empty: bool = True,
        flags: int = re.UNICODE | re.MULTILINE | re.DOTALL,
    ):
        self.pattern = pattern
        self.gaps = gaps
        self.discard_empty = discard_empty
        self.flags = flags

    def tokenize(self, text: BaseString) -> Sequence[BaseString]:
        return self._tokenizer.tokenize(text)


class WhiteSpaceTokenizer(TokenizerI):
    _instance = None
    _tokenizer = nltk.tokenize.WhitespaceTokenizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(self, text: BaseString) -> Sequence[BaseString]:
        return self._tokenizer.tokenize(text)
