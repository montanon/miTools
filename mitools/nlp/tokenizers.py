import re
from abc import ABCMeta, abstractmethod
from typing import Iterator, List, Sequence, Tuple

import nltk
from nltk.tokenize.api import TokenizerI

from mitools.utils.helper_functions import strip_punctuation


class BaseTokenizer(TokenizerI, metaclass=ABCMeta):
    @abstractmethod
    def tokenize(self, text: str, lower: bool = False) -> Sequence[str]:
        pass

    def itokenize(self, text: str, *args, **kwargs) -> Iterator[str]:
        return (t for t in self.tokenize(text, *args, **kwargs))

    def span_tokenize(self, text: str) -> Iterator[Tuple[int, int]]:
        return TokenizerI.span_tokenize(self, text)

    def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(
        self, strings: List[str]
    ) -> Iterator[List[Tuple[int, int]]]:
        for s in strings:
            yield list(self.span_tokenize(s))


class WordTokenizer(BaseTokenizer):
    _instance = None
    _tokenizer = staticmethod(nltk.tokenize.word_tokenize)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(
        self,
        text: str,
        include_punctuation: bool = True,
        lower: bool = False,
        *args,
        **kwargs,
    ):
        tokens = self._tokenizer(text, *args, **kwargs)
        if lower:
            tokens = [token.lower() for token in tokens]
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
    _tokenizer = staticmethod(nltk.tokenize.sent_tokenize)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(
        self, text: str, lower: bool = False, *args, **kwargs
    ) -> Sequence[str]:
        tokens = self._tokenizer(text, *args, **kwargs)
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens


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
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        pattern: str,
        gaps: bool = False,
        discard_empty: bool = True,
        flags: int = re.UNICODE | re.MULTILINE | re.DOTALL,
    ):
        if not hasattr(self, "_initialized") or not self._initialized:
            self.pattern = pattern
            self.gaps = gaps
            self.discard_empty = discard_empty
            self.flags = flags
            self._tokenizer = nltk.tokenize.RegexpTokenizer(
                pattern=pattern, gaps=gaps, discard_empty=discard_empty, flags=flags
            )
            self._initialized = True

    def tokenize(
        self, text: str, lower: bool = False, *args, **kwargs
    ) -> Sequence[str]:
        tokens = self._tokenizer.tokenize(text, *args, **kwargs)
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens


class WhiteSpaceTokenizer(TokenizerI):
    _instance = None
    _tokenizer = nltk.tokenize.WhitespaceTokenizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(self, text: str, lower: bool = False) -> Sequence[str]:
        tokens = self._tokenizer.tokenize(text)
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens


class WordPunctTokenizer(BaseTokenizer):
    _instance = None
    _tokenizer = nltk.tokenize.WordPunctTokenizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(
        self, text: str, lower: bool = False, *args, **kwargs
    ) -> Sequence[str]:
        tokens = self._tokenizer.tokenize(text, *args, **kwargs)
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens


class BlanklineTokenizer(BaseTokenizer):
    _instance = None
    _tokenizer = nltk.tokenize.BlanklineTokenizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tokenize(
        self, text: str, lower: bool = False, *args, **kwargs
    ) -> Sequence[str]:
        tokens = self._tokenizer.tokenize(text, *args, **kwargs)
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens
