from abc import ABCMeta, abstractmethod
from typing import Sequence

import nltk
from nltk import BigramTagger, RegexpTagger, UnigramTagger
from nltk.tokenize import sent_tokenize
from nltk.tree import Tree

from mitools.nlp.parsers import BaseParser, ChunkParser
from mitools.nlp.taggers import BaseTagger, PatternTagger
from mitools.nlp.utils import (
    filter_insignificant,
    is_match,
    normalize_tags,
    tree_to_string,
)


class BaseNPExtractor(ABCMeta):
    @abstractmethod
    def extract(self, text: str) -> Sequence[str]:
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

    def extract(self, text: str) -> Sequence[str]:
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

    def _parse_sentence(self, sentence: str) -> Sequence[Tree]:
        tagged = self.tagger.tag_tokens(sentence)
        return self.parser.parse(tagged)


class FastNPExtractor(BaseNPExtractor):
    CFG = {
        ("NNP", "NNP"): "NNP",
        ("NN", "NN"): "NNI",
        ("NNI", "NN"): "NNI",
        ("JJ", "JJ"): "JJ",
        ("JJ", "NN"): "NNI",
    }

    def __init__(self):
        self._trained = False

    def train(self):
        train_data = nltk.corpus.brown.tagged_sents(categories="news")
        regexp_tagger = RegexpTagger(
            [
                (r"^-?[0-9]+(.[0-9]+)?$", "CD"),
                (r"(-|:|;)$", ":"),
                (r"\'*$", "MD"),
                (r"(The|the|A|a|An|an)$", "AT"),
                (r".*able$", "JJ"),
                (r"^[A-Z].*$", "NNP"),
                (r".*ness$", "NN"),
                (r".*ly$", "RB"),
                (r".*s$", "NNS"),
                (r".*ing$", "VBG"),
                (r".*ed$", "VBD"),
                (r".*", "NN"),
            ]
        )
        unigram_tagger = UnigramTagger(train_data, backoff=regexp_tagger)
        self.tagger = BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True
        return None

    def _tokenize_sentence(self, sentence: str) -> Sequence[str]:
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def extract(self, sentence: str) -> Sequence[str]:
        if not self._trained:
            self.train()
        word_tokens = self._tokenize_sentence(sentence)
        pos_tagged_tokens = self.tagger.tag(word_tokens)
        normalized_pos_tags = normalize_tags(pos_tagged_tokens)
        can_merge_tokens = True
        while can_merge_tokens:
            can_merge_tokens = False
            for token_idx in range(0, len(normalized_pos_tags) - 1):
                current_token = normalized_pos_tags[token_idx]
                next_token = normalized_pos_tags[token_idx + 1]
                pos_tag_pair = current_token[1], next_token[1]
                merged_pos_tag = self.CFG.get(pos_tag_pair, "")
                if merged_pos_tag:
                    can_merge_tokens = True
                    normalized_pos_tags.pop(token_idx)
                    normalized_pos_tags.pop(token_idx)
                    merged_text = f"{current_token[0]} {next_token[0]}"
                    normalized_pos_tags.insert(token_idx, (merged_text, merged_pos_tag))
                    break
        noun_phrases = [
            token[0] for token in normalized_pos_tags if token[1] in ["NNP", "NNI"]
        ]
        return noun_phrases
