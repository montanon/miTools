from abc import ABC, abstractmethod
from typing import Iterator, Sequence, Tuple

import nltk

from mitools.nlp.en import tag as pattern_tag
from mitools.nlp.nlp_typing import BaseString, PosTag


class BaseTagger(ABC):
    @abstractmethod
    def tag_tokens(
        self, tokens: Sequence[BaseString]
    ) -> Sequence[Tuple[BaseString, PosTag]]:
        pass

    def itag_tokens(
        self, tokens: Sequence[BaseString]
    ) -> Iterator[Tuple[BaseString, PosTag]]:
        return (t for t in self.tag_tokens(tokens))


class PatternTagger(BaseTagger):
    def tag_tokens(
        self, tokens: Sequence[BaseString]
    ) -> Sequence[Tuple[BaseString, PosTag]]:
        return pattern_tag(tokens)


class NLTKTagger(BaseTagger):
    _instance = None
    _tagger = nltk.tag.pos_tag

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tag_tokens(
        self, tokens: Sequence[BaseString]
    ) -> Sequence[Tuple[BaseString, PosTag]]:
        return self._tagger(tokens)
