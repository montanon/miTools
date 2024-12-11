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
    def tokenize(self, text: BaseString, include_punctuation: bool = True):
        tokens = nltk.tokenize.word_tokenize(text)
        if include_punctuation:
            return tokens
        else:
            return [
                word if word.startswith("'") else strip_punctuation(word, all=False)
                for word in tokens
                if strip_punctuation(word, all=False)
            ]


class SentenceTokenizer(BaseTokenizer):
    def tokenize(self, text: BaseString) -> Sequence[BaseString]:
        return nltk.tokenize.sent_tokenize(text)


class RegexpTokenizer(BaseTokenizer):
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
        return nltk.tokenize.RegexpTokenizer(
            pattern=self.pattern,
            gaps=self.gaps,
            discard_empty=self.discard_empty,
            flags=self.flags,
        ).tokenize(text)
