from abc import ABCMeta, abstractmethod

import nltk
from nltk import BigramTagger, ChunkParserI, UnigramTagger
from nltk.tree import Tree

from mitools.nlp.en import en_parser as pattern_parse


class BaseParser(metaclass=ABCMeta):
    @abstractmethod
    def parse(self, text: str) -> Tree:
        pass


class ChunkParser(ChunkParserI):
    def __init__(self):
        self._trained = False

    def train(self):
        train_data = [
            [(t, c) for _, t, c in nltk.chunk.tree2conlltags(sent)]
            for sent in nltk.corpus.conll2000.chunked_sents(
                "train.txt", chunk_types=["NP"]
            )
        ]
        unigram_tagger = UnigramTagger(train_data)
        self.tagger = BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True

    def parse(self, sentence_tokens):
        if not self._trained:
            self.train()
        part_of_speech_tags = [pos_tag for (word, pos_tag) in sentence_tokens]
        pos_with_chunk_tags = self.tagger.tag(part_of_speech_tags)
        chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in pos_with_chunk_tags]
        word_pos_chunk_tuples = [
            (word, pos_tag, chunk_tag)
            for ((word, pos_tag), chunk_tag) in zip(sentence_tokens, chunk_tags)
        ]
        return nltk.chunk.util.conlltags2tree(word_pos_chunk_tuples)


class PatternParser(BaseParser):
    def parse(self, text: str):
        return pattern_parse(text)
