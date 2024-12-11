from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import nltk

from mitools.nlp.en import tag as pattern_tag
from mitools.nlp.nlp_typing import BaseString, PosTag
from mitools.nlp.tokenizers import BaseTokenizer, WordTokenizer


class BaseTagger(ABC):
    @abstractmethod
    def tag(self, tokens: Sequence[BaseString]) -> Sequence[Tuple[BaseString, PosTag]]:
        pass


class PatternTagger(BaseTagger):
    def tag(self, tokens: Sequence[BaseString]) -> Sequence[Tuple[BaseString, PosTag]]:
        return pattern_tag(tokens)


class NLTKTagger(BaseTagger):
    _instance = None
    _tagger = nltk.tag.pos_tag

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tag(self, tokens: Sequence[BaseString]) -> Sequence[Tuple[BaseString, PosTag]]:
        return self._tagger(tokens)
