import re
import string
from itertools import chain
from typing import Dict, Iterable, Sequence, Set, Tuple, Union

from nltk.tree import Tree

from mitools.exceptions import ArgumentStructureError
from mitools.nlp.blobs import Word
from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.nlp.typing import BaseString, PennTag, WordNetTag

PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
SLASH, WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA = (
    "&slash;",
    "word",
    "part-of-speech",
    "chunk",
    "preposition",
    "relation",
    "anchor",
    "lemma",
)
UNIVERSAL = "universal"
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
TOKEN = re.compile(r"(\S+)\s")
PUNCTUATION = punctuation = ".,;:!?()[]{}`''\"@#$^&*+-|=~_"
ABBREVIATIONS = abbreviations = set(
    (
        "a.",
        "adj.",
        "adv.",
        "al.",
        "a.m.",
        "c.",
        "cf.",
        "comp.",
        "conf.",
        "def.",
        "ed.",
        "e.g.",
        "esp.",
        "etc.",
        "ex.",
        "f.",
        "fig.",
        "gen.",
        "id.",
        "i.e.",
        "int.",
        "l.",
        "m.",
        "Med.",
        "Mil.",
        "Mr.",
        "n.",
        "n.q.",
        "orig.",
        "pl.",
        "pred.",
        "pres.",
        "p.m.",
        "ref.",
        "v.",
        "vs.",
        "w/",
    )
)
RE_ABBR1 = re.compile(r"^[A-Za-z]\.$")  # single letter, "T. De Smedt"
RE_ABBR2 = re.compile(r"^([A-Za-z]\.)+$")  # alternating letters, "U.S."
RE_ABBR3 = re.compile(
    "^[A-Z]["
    + "|".join(  # capital followed by consonants, "Mr."
        "bcdfghjklmnpqrstvwxz"
    )
    + "]+.$"
)
EMOTICONS = {  # (facial expression, sentiment)-keys
    ("love", +1.00): set(("<3", "♥")),
    ("grin", +1.00): set(
        (">:D", ":-D", ":D", "=-D", "=D", "X-D", "x-D", "XD", "xD", "8-D")
    ),
    ("taunt", +0.75): set(
        (">:P", ":-P", ":P", ":-p", ":p", ":-b", ":b", ":c)", ":o)", ":^)")
    ),
    ("smile", +0.50): set(
        (">:)", ":-)", ":)", "=)", "=]", ":]", ":}", ":>", ":3", "8)", "8-)")
    ),
    ("wink", +0.25): set((">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", "*-)", "*)")),
    ("gasp", +0.05): set((">:o", ":-O", ":O", ":o", ":-o", "o_O", "o.O", "°O°", "°o°")),
    ("worry", -0.25): set(
        (">:/", ":-/", ":/", ":\\", ">:\\", ":-.", ":-s", ":s", ":S", ":-S", ">.>")
    ),
    ("frown", -0.75): set(
        (">:[", ":-(", ":(", "=(", ":-[", ":[", ":{", ":-<", ":c", ":-c", "=/")
    ),
    ("cry", -1.00): set((":'(", ":'''(", ";'(")),
}
RE_EMOTICONS = [
    r" ?".join([re.escape(each) for each in e]) for v in EMOTICONS.values() for e in v
]
RE_EMOTICONS = re.compile(r"(%s)($|\s)" % "|".join(RE_EMOTICONS))
RE_SARCASM = re.compile(r"\( ?\! ?\)")
REPLACEMENTS = {
    "'d": " 'd",
    "'m": " 'm",
    "'s": " 's",
    "'ll": " 'll",
    "'re": " 're",
    "'ve": " 've",
    "n't": " n't",
}
EOS = "END-OF-SENTENCE"


def singularize(word: Union[str, Word]) -> Word:
    pass


def pluralize(word: Union[str, Word]) -> Word:
    pass


def suggest(word: Union[str, Word]) -> Word:
    pass


def penn_to_wordnet(penn_tag: PennTag) -> WordNetTag:
    pass


def lowerstrip(s: str, all: bool = False) -> str:
    return strip_punctuation(s.lower().strip(), all=all)


def strip_punctuation(s: str, all: bool = False) -> str:
    if all:
        return PUNCTUATION_REGEX.sub("", s.strip())
    else:
        return s.strip().strip(string.punctuation)


def word_tokenize(
    text: BaseString,
    word_tokenizer: WordTokenizer = None,
    sent_tokenizer: SentenceTokenizer = None,
    include_punctuation: bool = True,
    *args,
    **kwargs,
) -> Sequence[str]:
    word_tokenizer = word_tokenizer if word_tokenizer is not None else WordTokenizer()
    sent_tokenizer = (
        sent_tokenizer if sent_tokenizer is not None else SentenceTokenizer()
    )
    words = chain.from_iterable(
        word_tokenizer.itokenize(
            sentence, include_punctuation=include_punctuation, *args, **kwargs
        )
        for sentence in sent_tokenizer.itokenize(text)
    )
    return words


def get_words_from_corpus(corpus: Iterable[BaseString]) -> Set[Word]:
    def tokenize(words):
        if isinstance(words, BaseString):
            return word_tokenize(words, include_punctuation=False)
        return words

    all_words = chain.from_iterable(tokenize(words) for words, _ in corpus)
    return set(all_words)


def get_corpus_tokens(corpus: Iterable[BaseString]) -> Set[Word]:
    if isinstance(corpus, BaseString):
        tokens = set(
            strip_punctuation(word, all=False)
            for word in word_tokenize(corpus, include_punctuation=False)
        )
    else:
        tokens = set(strip_punctuation(word, all=False) for word in corpus)
    return tokens


def basic_extractor(
    corpus: Iterable[BaseString], train_set: Iterable[BaseString]
) -> Dict[str, bool]:
    try:
        zero_item = next(iter(train_set))
    except StopIteration:
        return {}
    if isinstance(zero_item, BaseString):
        word_features = [word for word in chain([zero_item], train_set)]
    else:
        try:
            assert isinstance(zero_item[0], BaseString)
            word_features = get_words_from_corpus(chain([zero_item], train_set))
        except Exception as error:
            raise ArgumentStructureError("train_set is probably malformed.") from error
    tokens = get_corpus_tokens(corpus)
    features = dict((f"contains({word})", (word in tokens)) for word in word_features)
    return features


def contains_extractor(corpus: Iterable[BaseString]) -> Dict[str, bool]:
    tokens = get_corpus_tokens(corpus)
    features = dict((f"contains({word})", True) for word in tokens)
    return features


def normalize_tags(chunk: Sequence[Tuple[str, str]]) -> Sequence[Tuple[str, str]]:
    normalized_tags = []
    for word, pos_tag in chunk:
        if pos_tag == "NP-TL" or pos_tag == "NP":
            normalized_tags.append((word, "NNP"))
            continue
        if pos_tag.endswith("-TL"):
            normalized_tags.append((word, pos_tag[:-3]))
            continue
        if pos_tag.endswith("S"):
            normalized_tags.append((word, pos_tag[:-1]))
            continue
        normalized_tags.append((word, pos_tag))
    return normalized_tags


def is_match(
    tagged_phrase: Sequence[Tuple[str, str]], cfg: Dict[Tuple[str, str], str]
) -> bool:
    phrase_tokens = list(tagged_phrase)  # A copy of the list
    can_merge_tokens = True
    while can_merge_tokens:
        can_merge_tokens = False
        for token_idx in range(len(phrase_tokens) - 1):
            current_token, next_token = (
                phrase_tokens[token_idx],
                phrase_tokens[token_idx + 1],
            )
            pos_tag_pair = (
                current_token[1],
                next_token[1],
            )  # Tuple of tags e.g. ('NN', 'JJ')
            merged_pos_tag = cfg.get(pos_tag_pair, None)
            if merged_pos_tag:
                can_merge_tokens = True
                phrase_tokens.pop(token_idx)
                phrase_tokens.pop(token_idx)
                merged_text = f"{current_token[0]} {next_token[0]}"
                merged_token = (merged_text, merged_pos_tag)
                phrase_tokens.insert(token_idx, merged_token)
                break
    contains_noun_phrase = any([token[1] in ("NNP", "NNI") for token in phrase_tokens])
    return contains_noun_phrase


def filter_insignificant(
    chunk: Sequence[str], tag_suffixes: Tuple[str] = ("DT", "CC", "PRP$", "PRP")
) -> bool:
    significant_tokens = []
    for word, pos_tag in chunk:
        is_significant = True
        for insignificant_suffix in tag_suffixes:
            if pos_tag.endswith(insignificant_suffix):
                is_significant = False
                break
        if is_significant:
            significant_tokens.append((word, pos_tag))
    return significant_tokens


def tree_to_string(tree: Tree, concat: str = " ") -> str:
    return concat.join([word for (word, tag) in tree])


def default_feature_extractor(words: Iterable[str]) -> Dict[str, bool]:
    return dict((word, True) for word in words)


def decode_string(value, encoding="utf-8"):
    if isinstance(encoding, BaseString):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(value, bytes):
        for e in encoding:
            try:
                return value.decode(*e)
            except Exception:
                pass
        return value
    return str(value)


def encode_string(value, encoding="utf-8"):
    if isinstance(encoding, BaseString):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(value, str):
        for e in encoding:
            try:
                return value.encode(*e)
            except Exception:
                pass
        return value
    return str(value)


def is_numeric(value: BaseString) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def penntreebank_to_universal(token, tag):
    if tag.startswith(("NNP-", "NNPS-")):
        return (token, "{}-{}".format(NOUN, tag.split("-")[-1]))
    if tag in ("NN", "NNS", "NNP", "NNPS", "NP"):
        return (token, NOUN)
    if tag in ("MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return (token, VERB)
    if tag in ("JJ", "JJR", "JJS"):
        return (token, ADJ)
    if tag in ("RB", "RBR", "RBS", "WRB"):
        return (token, ADV)
    if tag in ("PRP", "PRP$", "WP", "WP$"):
        return (token, PRON)
    if tag in ("DT", "PDT", "WDT", "EX"):
        return (token, DET)
    if tag in ("IN",):
        return (token, PREP)
    if tag in ("CD",):
        return (token, NUM)
    if tag in ("CC",):
        return (token, CONJ)
    if tag in ("UH",):
        return (token, INTJ)
    if tag in ("POS", "RP", "TO"):
        return (token, PRT)
    if tag in ("SYM", "LS", ".", "!", "?", ",", ":", "(", ")", '"', "#", "$"):
        return (token, PUNC)
    return (token, X)


def find_tokens(
    text: BaseString,
    punctuation_chars: Tuple[str] = PUNCTUATION,
    known_abbreviations: Set[str] = ABBREVIATIONS,
    contraction_replacements: Dict[str, str] = REPLACEMENTS,
    paragraph_break: str = r"\n{2,}",
):
    # Handle periods separately.
    punctuation_chars = tuple(punctuation_chars.replace(".", ""))
    # Handle replacements (contractions).
    for contraction, expanded in list(contraction_replacements.items()):
        text = re.sub(contraction, expanded, text)
    # Handle Unicode quotes.
    if isinstance(text, str):
        text = (
            str(text)
            .replace("“", " “ ")
            .replace("”", " ” ")
            .replace("‘", " ‘ ")
            .replace("’", " ’ ")
            .replace("'", " ' ")
            .replace('"', ' " ')
        )
    # Collapse whitespace.
    text = re.sub("\r\n", "\n", text)
    text = re.sub(paragraph_break, " %s " % EOS, text)
    text = re.sub(r"\s+", " ", text)
    token_list = []
    for current_token in TOKEN.findall(text + " "):
        if len(current_token) > 0:
            trailing_punctuation = []
            while (
                current_token.startswith(punctuation_chars)
                and current_token not in contraction_replacements
            ):
                # Split leading punctuation.
                if current_token.startswith(punctuation_chars):
                    token_list.append(current_token[0])
                    current_token = current_token[1:]
            while (
                current_token.endswith(punctuation_chars + (".",))
                and current_token not in contraction_replacements
            ):
                # Split trailing punctuation.
                if current_token.endswith(punctuation_chars):
                    trailing_punctuation.append(current_token[-1])
                    current_token = current_token[:-1]
                # Split ellipsis (...) before splitting period.
                if current_token.endswith("..."):
                    trailing_punctuation.append("...")
                    current_token = current_token[:-3].rstrip(".")
                # Split period (if not an abbreviation).
                if current_token.endswith("."):
                    if (
                        current_token in known_abbreviations
                        or RE_ABBR1.match(current_token) is not None
                        or RE_ABBR2.match(current_token) is not None
                        or RE_ABBR3.match(current_token) is not None
                    ):
                        break
                    else:
                        trailing_punctuation.append(current_token[-1])
                        current_token = current_token[:-1]
            if current_token != "":
                token_list.append(current_token)
            token_list.extend(reversed(trailing_punctuation))
    sentence_list, start_idx, current_idx = [[]], 0, 0
    while current_idx < len(token_list):
        if token_list[current_idx] in ("...", ".", "!", "?", EOS):
            # Handle citations, trailing parenthesis, repeated punctuation (!?).
            while current_idx < len(token_list) and token_list[current_idx] in (
                "'",
                '"',
                "”",
                "’",
                "...",
                ".",
                "!",
                "?",
                ")",
                EOS,
            ):
                if (
                    token_list[current_idx] in ("'", '"')
                    and sentence_list[-1].count(token_list[current_idx]) % 2 == 0
                ):
                    break  # Balanced quotes.
                current_idx += 1
            sentence_list[-1].extend(
                token for token in token_list[start_idx:current_idx] if token != EOS
            )
            sentence_list.append([])
            start_idx = current_idx
        current_idx += 1
    sentence_list[-1].extend(token_list[start_idx:current_idx])
    sentence_strings = (
        " ".join(sentence) for sentence in sentence_list if len(sentence) > 0
    )
    sentence_strings = (
        RE_SARCASM.sub("(!)", sentence) for sentence in sentence_strings
    )
    sentence_strings = [
        RE_EMOTICONS.sub(lambda m: m.group(1).replace(" ", "") + m.group(2), sentence)
        for sentence in sentence_strings
    ]
    return sentence_strings
