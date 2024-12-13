from typing import Literal, Union

from nltk.corpus import wordnet

from mitools.nlp.nlp_typing import (
    BaseString,
    NLTKTag,
    PennTag,
    PosTag,
    UniversalTag,
    WordNetTag,
)

NOUN, VERB, ADJ, ADV, PRON, DET, PREP, ADP, NUM, CONJ, INTJ, PRT, PUNC, X = (
    "NN",
    "VB",
    "JJ",
    "RB",
    "PR",
    "DT",
    "PP",
    "PP",
    "NO",
    "CJ",
    "UH",
    "PT",
    ".",
    "X",
)

WORDNET_TO_UNIVERSAL = {
    wordnet.ADJ: ADJ,
    wordnet.ADJ_SAT: ADJ,
    wordnet.ADV: ADV,
    wordnet.NOUN: NOUN,
    wordnet.VERB: VERB,
}

UNIVERSAL_TO_WORDNET = {v: k for k, v in WORDNET_TO_UNIVERSAL.items()}

PENN_TO_UNIVERSAL = {
    "NN": NOUN,
    "NNS": NOUN,
    "NNP": NOUN,
    "NNPS": NOUN,
    "NP": NOUN,
    "MD": VERB,
    "VB": VERB,
    "VBD": VERB,
    "VBG": VERB,
    "VBN": VERB,
    "VBP": VERB,
    "VBZ": VERB,
    "JJ": ADJ,
    "JJR": ADJ,
    "JJS": ADJ,
    "RB": ADV,
    "RBR": ADV,
    "RBS": ADV,
    "WRB": ADV,
    "PRP": PRON,
    "PRP$": PRON,
    "WP": PRON,
    "WP$": PRON,
    "DT": DET,
    "PDT": DET,
    "WDT": DET,
    "EX": DET,
    "IN": PREP,
    "CD": NUM,
    "CC": CONJ,
    "UH": INTJ,
    "POS": PRT,
    "RP": PRT,
    "TO": PRT,
    "SYM": PUNC,
    "LS": PUNC,
    ".": PUNC,
    "!": PUNC,
    "?": PUNC,
    ",": PUNC,
    ":": PUNC,
    "(": PUNC,
    ")": PUNC,
    '"': PUNC,
    "#": PUNC,
    "$": PUNC,
}

UNIVERSAL_TO_PENN = {
    NOUN: "NN",
    VERB: "VB",
    ADJ: "JJ",
    ADV: "RB",
    PRON: "PR",
    DET: "DT",
    PREP: "PP",
    NUM: "NO",
    CONJ: "CJ",
    INTJ: "UH",
    PRT: "PT",
    PUNC: ".",
    X: "X",
}

NLTK_TO_UNIVERSAL = {
    "J": ADJ,
    "V": VERB,
    "N": NOUN,
    "R": ADV,
}

UNIVERSAL_TO_NLTK = {v: k for k, v in NLTK_TO_UNIVERSAL.items()}


def translate_tag(
    tag: Union[PennTag, UniversalTag, WordNetTag, NLTKTag],
    source_format: Literal["penn", "universal", "wordnet", "nltk"],
    target_format: Literal["penn", "universal", "wordnet", "nltk"],
) -> Union[PennTag, UniversalTag, WordNetTag, NLTKTag]:
    if tag == X:
        return X
    tag_maps = {
        "penn-universal": PENN_TO_UNIVERSAL,
        "universal-penn": UNIVERSAL_TO_PENN,
        "wordnet-universal": WORDNET_TO_UNIVERSAL,
        "universal-wordnet": UNIVERSAL_TO_WORDNET,
        "nltk-universal": NLTK_TO_UNIVERSAL,
        "universal-nltk": UNIVERSAL_TO_NLTK,
    }
    translations = f"{source_format}-{target_format}"
    if translations not in tag_maps:
        translations = [f"{source_format}-universal", f"universal-{target_format}"]
    if not isinstance(translations, list):
        translations = [translations]
    for translation in translations:
        tag = tag_maps[translation].get(tag, X)
        if tag == X:
            return X
    return tag
