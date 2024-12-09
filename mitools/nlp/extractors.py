from abc import ABCMeta, abstractmethod
from typing import Iterable, Sequence, Set

from nltk import ChunkParserI
from nltk.tokenize import sent_tokenize
from nltk.tree import Tree

from mitools.nlp.parsers import BaseParser, ChunkParser
from mitools.nlp.taggers import BaseTagger, PatternTagger
from mitools.nlp.typing import BaseString
from mitools.nlp.utils import (
    filter_insignificant,
    is_match,
    normalize_tags,
    tree_to_string,
)


class BaseNPExtractor(ABCMeta):
    @abstractmethod
    def extract(self, text: BaseString) -> Sequence[str]:
        pass


class ConllExtractor(BaseNPExtractor):
    CFG = {
        ("NNP", "NNP"): "NNP",
        ("NN", "NN"): "NNI",
        ("NNI", "NN"): "NNI",
        ("JJ", "JJ"): "JJ",
        ("JJ", "NN"): "NNI",
    }
    INSIGNIFICANT_SUFFIXES = ["DT", "CC", "PRP$", "PRP"]

    def __init__(self, parser: BaseParser = None, tagger: BaseTagger = None):
        self.parser = parser if parser is not None else ChunkParser()
        self.tagger = tagger if tagger is not None else PatternTagger()

    def extract(self, text: BaseString) -> Sequence[str]:
        sentences = sent_tokenize(text)
        noun_phrases = []
        for sentence in sentences:
            parsed = self._parse_sentence(sentence)
            phrases = [
                normalize_tags(filter_insignificant(each, self.INSIGNIFICANT_SUFFIXES))
                for each in parsed
                if isinstance(each, Tree)
                and each.label() == "NP"
                and len(filter_insignificant(each)) >= 1
                and is_match(each, cfg=self.CFG)
            ]
            nps = [tree_to_string(phrase) for phrase in phrases]
            noun_phrases.extend(nps)
        return noun_phrases

    def _parse_sentence(self, sentence: BaseString) -> Sequence[Tree]:
        tagged = self.tagger.tag(sentence)
        return self.parser.parse(tagged)
