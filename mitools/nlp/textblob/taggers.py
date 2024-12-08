import re
import string
import sys
from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Dict, Iterator, Literal, Sequence, Tuple, Union

import nltk
from nltk.tokenize.api import TokenizerI
from nltk.tree import Tree

wordnet = nltk.corpus.wordnet
Synset = nltk.corpus.wordnet.synset
Lemma = nltk.corpus.wordnet.lemma

VERB, NOUN, ADJ, ADV = wordnet.VERB, wordnet.NOUN, wordnet.ADJ, wordnet.ADV
PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")


class BaseTagger(ABCMeta):
    @abstractmethod
    def tag(self, text, tokenize=True) -> Sequence[Tuple[str, str]]:
        pass


class PatternTagger(BaseTagger):
    def tag(self, text: str, tokenize: bool = True) -> str:
        if not isinstance(text, str):
            text = text.raw
        return pattern_tag(text, tokenize=tokenize)


class NLTKTagger(BaseTagger):
    def tag(self, text: str):
        if isinstance(text, str):
            text = TextBlob(text)
        return nltk.tag.pos_tag(text.tokens)
