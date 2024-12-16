import os
from typing import Sequence, Tuple

from mitools.nlp.definitions import CHUNK, PENN, PNP, POS, UNIVERSAL, WORD
from mitools.nlp.en.inflect import singularize
from mitools.nlp.objects import (
    Lexicon,
    Parser,
    Sentiment,
    Spelling,
)
from mitools.nlp.tags_translator import translate_tag

try:
    MODULE = os.path.dirname(os.path.abspath(__file__))
except Exception:
    MODULE = ""

en_spelling = Spelling(path=os.path.join(MODULE, "en-spelling.txt"))

# --- ENGLISH PARSER --------------------------------------------------------------------------------


def find_lemmata(tokens: Sequence[Tuple[str, str]]):
    for token in tokens:
        word, pos, lemma = token[0], token[1], token[0]
        # cats => cat
        if pos == "NNS":
            lemma = singularize(word)
        # sat => sit
        if pos.startswith(("VB", "MD")):
            lemma = word  # conjugate(word, INFINITIVE) or word
        token.append(lemma.lower())
    return tokens


class EnParser(Parser):
    def find_lemmata(self, tokens, **kwargs):
        return find_lemmata(tokens, **kwargs)

    def find_tags(self, tokens, **kwargs):
        if kwargs.get("tagset") in (PENN, None):
            kwargs.setdefault("map", lambda token, tag: (token, tag))
        if kwargs.get("tagset") == UNIVERSAL:
            kwargs.setdefault(
                "map",
                lambda token, pos_tag: (
                    token,
                    translate_tag(pos_tag, source="penn", target="universal"),
                ),
            )
        return Parser.find_tags(self, tokens, **kwargs)


class EnSentiment(Sentiment):
    def load(self, path=None):
        Sentiment.load(self, path)
        # Map "terrible" to adverb "terribly" (+1% accuracy)
        if not path:
            for w, pos in list(dict.items(self)):
                if "JJ" in pos:
                    if w.endswith("y"):
                        w = w[:-1] + "i"
                    if w.endswith("le"):
                        w = w[:-2]
                    p, s, i = pos["JJ"]
                    self.annotate(w + "ly", "RB", p, s, i)


en_lexicon = Lexicon(
    path=os.path.join(MODULE, "en-lexicon.txt"),
    morphology=os.path.join(MODULE, "en-morphology.txt"),
    context=os.path.join(MODULE, "en-context.txt"),
    entities=os.path.join(MODULE, "en-entities.txt"),
    language="en",
)
en_parser = EnParser(lexicon=en_lexicon, default=("NN", "NNP", "CD"), language="en")

en_sentiment = EnSentiment(
    path=os.path.join(MODULE, "en-sentiment.xml"),
    synset="wordnet_id",
    negations=("no", "not", "n't", "never"),
    modifiers=("RB",),
    modifier=lambda w: w.endswith("ly"),
    tokenizer=en_parser.find_tokens,
    language="en",
)


def tokenize(s, *args, **kwargs):
    """Returns a list of sentences, where punctuation marks have been split from words."""
    return en_parser.find_tokens(str(s), *args, **kwargs)


def parse(s, *args, **kwargs):
    """Returns a tagged str string."""
    return en_parser.parse(str(s), *args, **kwargs)


def parsetree(s, *args, **kwargs):
    """Returns a parsed Text from the given string."""
    return Text(parse(str(s), *args, **kwargs))


def split(s, token=None):
    """Returns a parsed Text from the given parsed string."""
    if token is None:
        token = [WORD, POS, CHUNK, PNP]
    return Text(str(s), token)


def tag(s, tokenize=True, encoding="utf-8"):
    """Returns a list of (token, tag)-tuples from the given string."""
    tags = []
    for sentence in parse(s, tokenize, True, False, False, False, encoding).split():
        for token in sentence:
            tags.append((token[0], token[1]))
    return tags


def suggest(w):
    """Returns a list of (word, confidence)-tuples of spelling corrections."""
    return en_spelling.suggest(w)


def polarity(s, **kwargs):
    """Returns the sentence polarity (positive/negative) between -1.0 and 1.0."""
    return en_sentiment(str(s), **kwargs)[0]


def subjectivity(s, **kwargs):
    """Returns the sentence subjectivity (objective/subjective) between 0.0 and 1.0."""
    return en_sentiment(str(s), **kwargs)[1]


def positive(s, threshold=0.1, **kwargs):
    """Returns True if the given sentence has a positive sentiment (polarity >= threshold)."""
    return polarity(str(s), **kwargs) >= threshold
