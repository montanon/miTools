from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple

import nltk

from mitools.nlp.en import tag as pattern_tag
from mitools.nlp.nlp_typing import BaseString
from mitools.nlp.tokenizers import BaseTokenizer, WordTokenizer


class BaseTagger(ABCMeta):
    @abstractmethod
    def tag(self, text: BaseString, tokenize: bool = True) -> Sequence[Tuple[str, str]]:
        pass


class PatternTagger(BaseTagger):
    def tag(self, text: BaseString, tokenize: bool = True) -> str:
        if not isinstance(text, str):
            text = text.raw
        return pattern_tag(text, tokenize=tokenize)


class NLTKTagger(BaseTagger):
    def tag(self, text: BaseString, tokenizer: BaseTokenizer = None):
        tokenizer = tokenizer if tokenizer is not None else WordTokenizer()
        return nltk.tag.pos_tag(tokenizer.tokenize(text))
