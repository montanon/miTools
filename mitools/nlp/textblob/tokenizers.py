from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Iterator, Sequence

import nltk
from nltk.tokenize.api import TokenizerI

from mitools.nlp.textblob.utils import strip_punctuation


class BaseTokenizer(TokenizerI, ABCMeta):
    @abstractmethod
    def tokenize(self, text: str) -> Sequence[str]:
        pass

    def itokenize(self, text: str, *args, **kwargs) -> Iterator[str]:
        return (t for t in self.tokenize(text, *args, **kwargs))


class WordTokenizer(BaseTokenizer):
    def tokenize(self, text, include_punctuation=True):
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
    def tokenise(self, text: str) -> Sequence[str]:
        return nltk.tokenize.sent_tokenize(text)


def word_tokenize(
    text: str, include_punctuation: bool = True, *args, **kwargs
) -> Sequence[str]:
    word_tokenizer = WordTokenizer()
    sent_tokenizer = SentenceTokenizer()
    words = chain.from_iterable(
        word_tokenizer.itokenize(
            sentence, include_punctuation=include_punctuation, *args, **kwargs
        )
        for sentence in sent_tokenizer.itokenize(text)
    )
    return words
