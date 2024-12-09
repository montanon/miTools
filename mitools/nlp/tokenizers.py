from abc import ABCMeta, abstractmethod
from typing import Iterator, Sequence

import nltk
from nltk.tokenize.api import TokenizerI

from mitools.nlp.typing import BaseString
from mitools.nlp.utils import strip_punctuation


class BaseTokenizer(TokenizerI, ABCMeta):
    @abstractmethod
    def tokenize(self, text: BaseString) -> Sequence[BaseString]:
        pass

    def itokenize(self, text: BaseString, *args, **kwargs) -> Iterator[BaseString]:
        return (t for t in self.tokenize(text, *args, **kwargs))


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
