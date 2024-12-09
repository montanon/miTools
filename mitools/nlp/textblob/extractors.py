from abc import ABCMeta, abstractmethod
from typing import Sequence

import nltk
from nltk import ChunkParserI


class BaseNPExtractor(ABCMeta):
    @abstractmethod
    def extract(self, text: str) -> Sequence[str]:
        pass


class ConllExtractor(BaseNPExtractor):
    POS_TAGGER = PatternTagger()
    CFG = {
        ("NNP", "NNP"): "NNP",
        ("NN", "NN"): "NNI",
        ("NNI", "NN"): "NNI",
        ("JJ", "JJ"): "JJ",
        ("JJ", "NN"): "NNI",
    }
    INSIGNIFICANT_SUFFIXES = ["DT", "CC", "PRP$", "PRP"]

    def __init__(self, parser=None):
        self.parser = ChunkParser() if not parser else parser

    def extract(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        noun_phrases = []
        for sentence in sentences:
            parsed = self._parse_sentence(sentence)
            # Get the string representation of each subtree that is a
            # noun phrase tree
            phrases = [
                normalize_tags(filter_insignificant(each, self.INSIGNIFICANT_SUFFIXES))
                for each in parsed
                if isinstance(each, nltk.tree.Tree)
                and each.label() == "NP"
                and len(filter_insignificant(each)) >= 1
                and is_match(each, cfg=self.CFG)
            ]
            nps = [tree_to_str(phrase) for phrase in phrases]
            noun_phrases.extend(nps)
        return noun_phrases

    def _parse_sentence(self, sentence):
        tagged = self.POS_TAGGER.tag(sentence)
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
        regexp_tagger = nltk.RegexpTagger(
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
        unigram_tagger = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True
        return None

    def _tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def extract(self, sentence):
        if not self._trained:
            self.train()
        tokens = self._tokenize_sentence(sentence)
        tagged = self.tagger.tag(tokens)
        tags = normalize_tags(tagged)
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = t1[1], t2[1]
                value = self.CFG.get(key, "")
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = f"{t1[0]} {t2[0]}"
                    pos = value
                    tags.insert(x, (match, pos))
                    break

        matches = [t[0] for t in tags if t[1] in ["NNP", "NNI"]]
        return matches


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
        unigram_tagger = nltk.UnigramTagger(train_data)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True

    def parse(self, sentence):
        if not self._trained:
            self.train()
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [
            (word, pos, chunktag)
            for ((word, pos), chunktag) in zip(sentence, chunktags)
        ]
        return nltk.chunk.util.conlltags2tree(conlltags)


def normalize_tags(chunk):
    ret = []
    for word, tag in chunk:
        if tag == "NP-TL" or tag == "NP":
            ret.append((word, "NNP"))
            continue
        if tag.endswith("-TL"):
            ret.append((word, tag[:-3]))
            continue
        if tag.endswith("S"):
            ret.append((word, tag[:-1]))
            continue
        ret.append((word, tag))
    return ret


def is_match(tagged_phrase, cfg):
    copy = list(tagged_phrase)  # A copy of the list
    merge = True
    while merge:
        merge = False
        for i in range(len(copy) - 1):
            first, second = copy[i], copy[i + 1]
            key = first[1], second[1]  # Tuple of tags e.g. ('NN', 'JJ')
            value = cfg.get(key, None)
            if value:
                merge = True
                copy.pop(i)
                copy.pop(i)
                match = f"{first[0]} {second[0]}"
                pos = value
                copy.insert(i, (match, pos))
                break
    match = any([t[1] in ("NNP", "NNI") for t in copy])
    return match
