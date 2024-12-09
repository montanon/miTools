from abc import ABCMeta, abstractmethod
from typing import Dict, Sequence

import nltk
from nltk import BigramTagger, ChunkParserI, UnigramTagger
from nltk.tree import Tree

from mitools.nlp.definitions import ABBREVIATIONS, PUNCTUATION, REPLACEMENTS
from mitools.nlp.objects import TaggedString
from mitools.nlp.typing import BaseString
from mitools.nlp.utils import (
    decode_string,
    find_chunks,
    find_prepositions,
    find_relations,
    find_tags,
    find_tokens,
)


class BaseParser(ABCMeta):
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
    def parse(self, text: BaseString):
        return pattern_parse(text)


class Parser:
    def __init__(
        self,
        lexicon: Dict[str, float] = None,
        default: Sequence[str] = ("NN", "NNP", "CD"),
        language: str = None,
    ):
        if lexicon is None:
            lexicon = {}
        self.lexicon = lexicon
        self.default = default
        self.language = language

    def find_tokens(self, string: BaseString, **kwargs):
        # "The cat purs." => ["The cat purs ."]
        return find_tokens(
            str(string),
            punctuation=kwargs.get("punctuation", PUNCTUATION),
            abbreviations=kwargs.get("abbreviations", ABBREVIATIONS),
            replace=kwargs.get("replace", REPLACEMENTS),
            linebreak=r"\n{2,}",
        )

    def find_tags(self, tokens: Sequence[BaseString], **kwargs):
        # ["The", "cat", "purs"] => [["The", "DT"], ["cat", "NN"], ["purs", "VB"]]
        return find_tags(
            tokens,
            language=kwargs.get("language", self.language),
            lexicon=kwargs.get("lexicon", self.lexicon),
            default=kwargs.get("default", self.default),
            map=kwargs.get("map", None),
        )

    def find_chunks(self, tokens: Sequence[BaseString], **kwargs):
        # [["The", "DT"], ["cat", "NN"], ["purs", "VB"]] =>
        # [["The", "DT", "B-NP"], ["cat", "NN", "I-NP"], ["purs", "VB", "B-VP"]]
        return find_prepositions(
            find_chunks(tokens, language=kwargs.get("language", self.language))
        )

    def find_prepositions(self, tokens: Sequence[BaseString], **kwargs):
        return find_prepositions(tokens)  # See also Parser.find_chunks().

    def find_labels(self, tokens: Sequence[BaseString], **kwargs):
        return find_relations(tokens)

    def find_lemmata(self, tokens: Sequence[BaseString], **kwargs):
        return [token + [token[0].lower()] for token in tokens]

    def parse(
        self,
        text: BaseString,
        tokenize: bool = True,
        tags: bool = True,
        chunks: bool = True,
        relations: bool = False,
        lemmata: bool = False,
        encoding: str = "utf-8",
        **kwargs,
    ):
        # Tokenizer.
        if tokenize:
            text = self.find_tokens(text, **kwargs)
        if isinstance(text, (list, tuple)):
            text = [
                isinstance(text, BaseString) and text.split(" ") or text
                for text in text
            ]
        if isinstance(text, BaseString):
            text = [text.split(" ") for text in text.split("\n")]
        # Unicode.
        for sentence_idx in range(len(text)):
            for token_idx in range(len(text[sentence_idx])):
                if isinstance(text[sentence_idx][token_idx], bytes):
                    text[sentence_idx][token_idx] = decode_string(
                        text[sentence_idx][token_idx], encoding
                    )
            # Tagger (required by chunker, labeler & lemmatizer).
            if tags or chunks or relations or lemmata:
                text[sentence_idx] = self.find_tags(text[sentence_idx], **kwargs)
            else:
                text[sentence_idx] = [[word] for word in text[sentence_idx]]
            # Chunker.
            if chunks or relations:
                text[sentence_idx] = self.find_chunks(text[sentence_idx], **kwargs)
            # Labeler.
            if relations:
                text[sentence_idx] = self.find_labels(text[sentence_idx], **kwargs)
            # Lemmatizer.
            if lemmata:
                text[sentence_idx] = self.find_lemmata(text[sentence_idx], **kwargs)
        # Slash-formatted tagged string.
        # With collapse=False (or split=True), returns raw list
        # (this output is not usable by tree.Text).
        if not kwargs.get("collapse", True) or kwargs.get("split", False):
            return text
        # Construct TaggedString.format.
        # (this output is usable by tree.Text).
        tag_format = ["word"]
        if tags:
            tag_format.append("part-of-speech")
        if chunks:
            tag_format.extend(("chunk", "preposition"))
        if relations:
            tag_format.append("relation")
        if lemmata:
            tag_format.append("lemma")
        # Collapse raw list.
        # Sentences are separated by newlines, tokens by spaces, tags by slashes.
        # Slashes in words are encoded with &slash;
        for sentence_idx in range(len(text)):
            for token_idx in range(len(text[sentence_idx])):
                text[sentence_idx][token_idx][0] = text[sentence_idx][token_idx][
                    0
                ].replace("/", "&slash;")
                text[sentence_idx][token_idx] = "/".join(text[sentence_idx][token_idx])
            text[sentence_idx] = " ".join(text[sentence_idx])
        text = "\n".join(text)
        tagged_text = TaggedString(
            str(text), tag_format, language=kwargs.get("language", self.language)
        )
        return tagged_text
