import codecs
import re
import string
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union
from xml.etree import ElementTree

from nltk.tree import Tree

from mitools.exceptions import ArgumentStructureError
from mitools.nlp.definitions import (
    ABBREVIATIONS,
    ADJ,
    ADJECTIVE,
    ADV,
    ADVERB,
    CD,
    CHUNKS,
    CONJ,
    DET,
    EMOTICONS,
    EOS,
    INTJ,
    IRONY,
    MOOD,
    NOUN,
    NUM,
    PREP,
    PRON,
    PRT,
    PUNC,
    PUNCTUATION,
    PUNCTUATION_REGEX,
    RE_ABBR1,
    RE_ABBR2,
    RE_ABBR3,
    RE_EMOTICONS,
    RE_ENTITY1,
    RE_ENTITY2,
    RE_ENTITY3,
    RE_SARCASM,
    RE_SYNSET,
    REPLACEMENTS,
    SEPARATOR,
    TOKEN,
    VERB,
    X,
)
from mitools.nlp.nlp_typing import BaseString, PennTag, PosTag, WordNetTag
from mitools.nlp.objects import TaggedString
from mitools.nlp.string_utils import strip_punctuation
from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.utils.helper_objects import LazyDict, LazyList


def find_relations(tokens: Sequence[BaseString]):
    raise NotImplementedError("find_relations() is not implemented.")


def avg(list: Iterable[float]) -> float:
    return sum(list) / float(len(list) or 1)


def suggest(word: BaseString) -> BaseString:
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
    sentence_tokenizer: SentenceTokenizer = None,
    include_punctuation: bool = True,
    *args,
    **kwargs,
) -> Sequence[str]:
    word_tokenizer = word_tokenizer if word_tokenizer is not None else WordTokenizer()
    words = chain.from_iterable(
        word_tokenizer.itokenize(
            sentence, include_punctuation=include_punctuation, *args, **kwargs
        )
        for sentence in sentence_tokenize(
            text, sentence_tokenizer, include_punctuation, *args, **kwargs
        )
    )
    return words


def sentence_tokenize(
    text: BaseString,
    sentence_tokenizer: SentenceTokenizer = None,
    include_punctuation: bool = True,
    *args,
    **kwargs,
):
    sentence_tokenizer = (
        sentence_tokenizer if sentence_tokenizer is not None else SentenceTokenizer()
    )
    return sentence_tokenizer.itokenize(
        text, include_punctuation=include_punctuation, *args, **kwargs
    )


def get_words_from_corpus(corpus: Iterable[BaseString]) -> Set[BaseString]:
    def tokenize(words):
        if isinstance(words, BaseString):
            return word_tokenize(words, include_punctuation=False)
        return words

    all_words = chain.from_iterable(tokenize(words) for words, _ in corpus)
    return set(all_words)


def get_corpus_tokens(corpus: Iterable[BaseString]) -> Set[BaseString]:
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


def decode_string(value: BaseString, encoding: str = "utf-8") -> str:
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


def encode_string(value: BaseString, encoding: str = "utf-8") -> BaseString:
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


def penntreebank_to_universal(
    token: BaseString, tag: PennTag
) -> Tuple[BaseString, WordNetTag]:
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


def read(path: Path, encoding="utf-8", comment=";;;"):
    if path:
        if isinstance(path, BaseString) and path.exists():
            f = open(path, encoding=encoding)
        elif isinstance(path, BaseString):
            f = path.splitlines()
        elif hasattr(path, "read"):
            f = path.read().splitlines()
        else:
            f = path
        for i, line in enumerate(f):
            line = (
                line.strip(codecs.BOM_UTF8)
                if i == 0 and isinstance(line, bytes)
                else line
            )
            line = line.strip()
            line = decode_string(line, encoding=encoding)
            if not line or (comment and line.startswith(comment)):
                continue
            yield line
    return


class Lexicon(LazyDict):
    def __init__(
        self,
        path: Path = Path(""),
        morphology: Path = None,
        context: Path = None,
        entities: Path = None,
        NNP: str = "NNP",
        language: str = None,
    ):
        self._path = path
        self._language = language
        self.morphology = Morphology(self, path=morphology)
        self.context = Context(self, path=context)
        self.entities = Entities(self, path=entities, tag=NNP)

    def load(self):
        dict.update(self, (x.split(" ")[:2] for x in read(self._path) if x.strip()))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language


class Rules:
    def __init__(
        self,
        lexicon: Union[Dict[str, str], None] = None,
        morph_operations: Union[Dict[str, str], None] = None,
    ):
        if morph_operations is None:
            morph_operations = {}
        if lexicon is None:
            lexicon = {}
        self.lexicon, self.morp_operations = lexicon, morph_operations

    def apply(self, x):
        return x


class Morphology(LazyList, Rules):
    def __init__(
        self, lexicon: Union[Dict[str, str], None] = None, path: Path = Path("")
    ):
        if lexicon is None:
            lexicon = {}
        morph_operations = (
            "char",  # Word contains x.
            "haspref",  # Word starts with x.
            "hassuf",  # Word end with x.
            "addpref",  # x + word is in lexicon.
            "addsuf",  # Word + x is in lexicon.
            "deletepref",  # Word without x at the start is in lexicon.
            "deletesuf",  # Word without x at the end is in lexicon.
            "goodleft",  # Word preceded by word x.
            "goodright",  # Word followed by word x.
        )
        morph_operations = dict.fromkeys(morph_operations, True)
        morph_operations.update(
            ("f" + operation, apply)
            for operation, apply in list(morph_operations.items())
        )
        Rules.__init__(self, lexicon, morph_operations)
        self._path = path

    @property
    def path(self):
        return self._path

    def load(self):
        list.extend(self, (x.split() for x in read(self._path)))

    def apply(
        self,
        token: Tuple[str, str],
        previous: Tuple[str, str] = (None, None),
        next: Tuple[str, str] = (None, None),
    ):
        word = token[0]
        for rule in self:
            if rule[1] in self.morp_operations:  # Rule = ly hassuf 2 RB x
                is_forward_rule = bool(0)
                affix = rule[0]
                target_pos = rule[-2]
                morph_operation = rule[1].lower()
            if rule[2] in self.morp_operations:  # Rule = NN s fhassuf 1 NNS x
                is_forward_rule = bool(1)
                affix = rule[1]
                target_pos = rule[-2]
                morph_operation = rule[2].lower().lstrip("f")
            if is_forward_rule and token[1] != rule[0]:
                continue
            if (
                (morph_operation == "char" and affix in word)
                or (morph_operation == "haspref" and word.startswith(affix))
                or (morph_operation == "hassuf" and word.endswith(affix))
                or (morph_operation == "addpref" and affix + word in self.lexicon)
                or (morph_operation == "addsuf" and word + affix in self.lexicon)
                or (
                    morph_operation == "deletepref"
                    and word.startswith(affix)
                    and word[len(affix) :] in self.lexicon
                )
                or (
                    morph_operation == "deletesuf"
                    and word.endswith(affix)
                    and word[: -len(affix)] in self.lexicon
                )
                or (morph_operation == "goodleft" and affix == next[0])
                or (morph_operation == "goodright" and affix == previous[0])
            ):
                token[1] = target_pos
        return token

    def insert(
        self,
        position,
        target_pos_tag,
        affix_pattern,
        morphological_operation="hassuf",
        source_pos_tag=None,
    ):
        if affix_pattern.startswith("-") and affix_pattern.endswith("-"):
            affix_pattern, morphological_operation = affix_pattern[1:-1], "char"
        if affix_pattern.startswith("-"):
            affix_pattern, morphological_operation = affix_pattern[1:], "hassuf"
        if affix_pattern.endswith("-"):
            affix_pattern, morphological_operation = affix_pattern[:-1], "haspref"
        if source_pos_tag:
            rule = [
                source_pos_tag,
                affix_pattern,
                "f" + morphological_operation.lstrip("f"),
                target_pos_tag,
                "x",
            ]
        else:
            rule = [
                affix_pattern,
                morphological_operation.lstrip("f"),
                target_pos_tag,
                "x",
            ]
        LazyList.insert(self, position, rule)

    def append(self, *args, **kwargs):
        self.insert(len(self) - 1, *args, **kwargs)

    def extend(self, rules: Iterable[Tuple[str]] = None):
        if rules is None:
            rules = []
        for rule in rules:
            self.append(*rule)


class Context(LazyList, Rules):
    def __init__(
        self, lexicon: Union[Dict[str, str], None] = None, path: Path = Path("")
    ):
        if lexicon is None:
            lexicon = {}
        morph_operations = (
            "prevtag",  # Preceding word is tagged x.
            "nexttag",  # Following word is tagged x.
            "prev2tag",  # Word 2 before is tagged x.
            "next2tag",  # Word 2 after is tagged x.
            "prev1or2tag",  # One of 2 preceding words is tagged x.
            "next1or2tag",  # One of 2 following words is tagged x.
            "prev1or2or3tag",  # One of 3 preceding words is tagged x.
            "next1or2or3tag",  # One of 3 following words is tagged x.
            "surroundtag",  # Preceding word is tagged x and following word is tagged y.
            "curwd",  # Current word is x.
            "prevwd",  # Preceding word is x.
            "nextwd",  # Following word is x.
            "prev1or2wd",  # One of 2 preceding words is x.
            "next1or2wd",  # One of 2 following words is x.
            "next1or2or3wd",  # One of 3 preceding words is x.
            "prev1or2or3wd",  # One of 3 following words is x.
            "prevwdtag",  # Preceding word is x and tagged y.
            "nextwdtag",  # Following word is x and tagged y.
            "wdprevtag",  # Current word is y and preceding word is tagged x.
            "wdnexttag",  # Current word is x and following word is tagged y.
            "wdand2aft",  # Current word is x and word 2 after is y.
            "wdand2tagbfr",  # Current word is y and word 2 before is tagged x.
            "wdand2tagaft",  # Current word is x and word 2 after is tagged y.
            "lbigram",  # Current word is y and word before is x.
            "rbigram",  # Current word is x and word after is y.
            "prevbigram",  # Preceding word is tagged x and word before is tagged y.
            "nextbigram",  # Following word is tagged x and word after is tagged y.
        )
        Rules.__init__(self, lexicon, dict.fromkeys(morph_operations, True))
        self._path = path

    @property
    def path(self):
        return self._path

    def load(self):
        list.extend(self, (x.split() for x in read(self._path)))

    def apply(self, tokens: Iterable[Tuple[str, str]]):
        padding_tokens = [("STAART", "STAART")] * 3
        padded_tokens = padding_tokens + tokens + padding_tokens
        for current_pos, current_token in enumerate(padded_tokens):
            for transformation_rule in self:
                if current_token[1] == "STAART":
                    continue
                if (
                    current_token[1] != transformation_rule[0]
                    and transformation_rule[0] != "*"
                ):
                    continue
                operation = transformation_rule[2].lower()
                pattern = transformation_rule[3]
                pattern2 = (
                    transformation_rule[4] if len(transformation_rule) > 4 else ""
                )

                if (
                    (
                        operation == "prevtag"
                        and pattern == padded_tokens[current_pos - 1][1]
                    )
                    or (
                        operation == "nexttag"
                        and pattern == padded_tokens[current_pos + 1][1]
                    )
                    or (
                        operation == "prev2tag"
                        and pattern == padded_tokens[current_pos - 2][1]
                    )
                    or (
                        operation == "next2tag"
                        and pattern == padded_tokens[current_pos + 2][1]
                    )
                    or (
                        operation == "prev1or2tag"
                        and pattern
                        in (
                            padded_tokens[current_pos - 1][1],
                            padded_tokens[current_pos - 2][1],
                        )
                    )
                    or (
                        operation == "next1or2tag"
                        and pattern
                        in (
                            padded_tokens[current_pos + 1][1],
                            padded_tokens[current_pos + 2][1],
                        )
                    )
                    or (
                        operation == "prev1or2or3tag"
                        and pattern
                        in (
                            padded_tokens[current_pos - 1][1],
                            padded_tokens[current_pos - 2][1],
                            padded_tokens[current_pos - 3][1],
                        )
                    )
                    or (
                        operation == "next1or2or3tag"
                        and pattern
                        in (
                            padded_tokens[current_pos + 1][1],
                            padded_tokens[current_pos + 2][1],
                            padded_tokens[current_pos + 3][1],
                        )
                    )
                    or (
                        operation == "surroundtag"
                        and pattern == padded_tokens[current_pos - 1][1]
                        and pattern2 == padded_tokens[current_pos + 1][1]
                    )
                    or (
                        operation == "curwd"
                        and pattern == padded_tokens[current_pos][0]
                    )
                    or (
                        operation == "prevwd"
                        and pattern == padded_tokens[current_pos - 1][0]
                    )
                    or (
                        operation == "nextwd"
                        and pattern == padded_tokens[current_pos + 1][0]
                    )
                    or (
                        operation == "prev1or2wd"
                        and pattern
                        in (
                            padded_tokens[current_pos - 1][0],
                            padded_tokens[current_pos - 2][0],
                        )
                    )
                    or (
                        operation == "next1or2wd"
                        and pattern
                        in (
                            padded_tokens[current_pos + 1][0],
                            padded_tokens[current_pos + 2][0],
                        )
                    )
                    or (
                        operation == "prevwdtag"
                        and pattern == padded_tokens[current_pos - 1][0]
                        and pattern2 == padded_tokens[current_pos - 1][1]
                    )
                    or (
                        operation == "nextwdtag"
                        and pattern == padded_tokens[current_pos + 1][0]
                        and pattern2 == padded_tokens[current_pos + 1][1]
                    )
                    or (
                        operation == "wdprevtag"
                        and pattern == padded_tokens[current_pos - 1][1]
                        and pattern2 == padded_tokens[current_pos][0]
                    )
                    or (
                        operation == "wdnexttag"
                        and pattern == padded_tokens[current_pos][0]
                        and pattern2 == padded_tokens[current_pos + 1][1]
                    )
                    or (
                        operation == "wdand2aft"
                        and pattern == padded_tokens[current_pos][0]
                        and pattern2 == padded_tokens[current_pos + 2][0]
                    )
                    or (
                        operation == "wdand2tagbfr"
                        and pattern == padded_tokens[current_pos - 2][1]
                        and pattern2 == padded_tokens[current_pos][0]
                    )
                    or (
                        operation == "wdand2tagaft"
                        and pattern == padded_tokens[current_pos][0]
                        and pattern2 == padded_tokens[current_pos + 2][1]
                    )
                    or (
                        operation == "lbigram"
                        and pattern == padded_tokens[current_pos - 1][0]
                        and pattern2 == padded_tokens[current_pos][0]
                    )
                    or (
                        operation == "rbigram"
                        and pattern == padded_tokens[current_pos][0]
                        and pattern2 == padded_tokens[current_pos + 1][0]
                    )
                    or (
                        operation == "prevbigram"
                        and pattern == padded_tokens[current_pos - 2][1]
                        and pattern2 == padded_tokens[current_pos - 1][1]
                    )
                    or (
                        operation == "nextbigram"
                        and pattern == padded_tokens[current_pos + 1][1]
                        and pattern2 == padded_tokens[current_pos + 2][1]
                    )
                ):
                    padded_tokens[current_pos] = [
                        padded_tokens[current_pos][0],
                        transformation_rule[1],
                    ]
        return padded_tokens[len(padding_tokens) : -len(padding_tokens)]

    def insert(
        self,
        position: int,
        source_tag: PosTag,
        target_tag: PosTag,
        operation: str = "prevtag",
        pattern1: str = None,
        pattern2: str = None,
    ):
        if " < " in source_tag and not pattern1 and not pattern2:
            source_tag, pattern1 = source_tag.split(" < ")
            operation = "prevtag"
        if " > " in source_tag and not pattern1 and not pattern2:
            pattern1, source_tag = source_tag.split(" > ")
            operation = "nexttag"
        LazyList.insert(
            self,
            position,
            [source_tag, target_tag, operation, pattern1 or "", pattern2 or ""],
        )

    def append(self, *args, **kwargs):
        self.insert(len(self) - 1, *args, **kwargs)

    def extend(self, rules: Iterable[Tuple[str]] = None):
        if rules is None:
            rules = []
        for rule in rules:
            self.append(*rule)


class Entities(LazyDict, Rules):
    def __init__(
        self,
        lexicon: Union[Dict[str, str], None] = None,
        path: Path = Path(""),
        tag: PosTag = "NNP",
    ):
        if lexicon is None:
            lexicon = {}
        morph_operations = (
            "pers",  # Persons: George/NNP-PERS
            "loc",  # Locations: Washington/NNP-LOC
            "org",  # Organizations: Google/NNP-ORG
        )
        Rules.__init__(self, lexicon, morph_operations)
        self._path = path
        self.tag = tag

    @property
    def path(self):
        return self._path

    def load(self):
        for x in read(self.path):
            x = [x.lower() for x in x.split()]
            dict.setdefault(self, x[0], []).append(x)

    def apply(self, tokens: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
        current_index = 0
        while current_index < len(tokens):
            current_word = tokens[current_index][0].lower()
            if (
                RE_ENTITY1.match(current_word)
                or RE_ENTITY2.match(current_word)
                or RE_ENTITY3.match(current_word)
            ):
                tokens[current_index][1] = self.tag
            if current_word in self:
                for entity_parts in self[current_word]:
                    entity_words, entity_type = (
                        (entity_parts[:-1], "-" + entity_parts[-1].upper())
                        if entity_parts[-1] in self.cmd
                        else (entity_parts, "")
                    )
                    matches_entity = True
                    for word_offset, entity_word in enumerate(entity_words):
                        if (
                            current_index + word_offset >= len(tokens)
                            or tokens[current_index + word_offset][0].lower()
                            != entity_word
                        ):
                            matches_entity = False
                            break
                    if matches_entity:
                        for token in tokens[
                            current_index : current_index + word_offset + 1
                        ]:
                            token[1] = (
                                token[1] if token[1] == "NNPS" else self.tag
                            ) + entity_type
                        current_index += word_offset
                        break
            current_index += 1
        return tokens

    def append(self, entity: str, name: str = "pers"):
        e = [s.lower() for s in entity.split(" ") + [name]]
        self.setdefault(e[0], []).append(e)

    def extend(self, entities: Iterable[Tuple[str, str]]):
        for entity, name in entities:
            self.append(entity, name)


def suffix_rules(token: BaseString, tag: PosTag = "NN"):
    if isinstance(token, (list, tuple)):
        token, tag = token
    if token.endswith("ing"):
        tag = "VBG"
    if token.endswith("ly"):
        tag = "RB"
    if token.endswith("s") and not token.endswith(("is", "ous", "ss")):
        tag = "NNS"
    if (
        token.endswith(
            ("able", "al", "ful", "ible", "ient", "ish", "ive", "less", "tic", "ous")
        )
        or "-" in token
    ):
        tag = "JJ"
    if token.endswith("ed"):
        tag = "VBN"
    if token.endswith(("ate", "ify", "ise", "ize")):
        tag = "VBP"
    return [token, tag]


def find_tags(
    word_tokens,
    lexicon: Dict[str, PosTag] = None,
    language_model: Callable = None,
    morphology_analyzer: Morphology = None,
    context_analyzer: Context = None,
    entity_recognizer: Entities = None,
    default_tags: Tuple[PosTag, PosTag, PosTag] = ("NN", "NNP", "CD"),
    language_code: str = "en",
    tag_mapper: Callable = None,
):
    if lexicon is None:
        lexicon = {}
    tagged_tokens = []
    # Tag known words.
    for i, word in enumerate(word_tokens):
        tagged_tokens.append(
            [word, lexicon.get(word, i == 0 and lexicon.get(word.lower()) or None)]
        )
    # Tag unknown words.
    for i, (word, word_tag) in enumerate(tagged_tokens):
        prev_token, next_token = (None, None), (None, None)
        if i > 0:
            prev_token = tagged_tokens[i - 1]
        if i < len(tagged_tokens) - 1:
            next_token = tagged_tokens[i + 1]
        if word_tag is None or word in (
            language_model is not None and language_model.unknown or ()
        ):
            # Use language model (i.e., SLP).
            if language_model is not None:
                tagged_tokens[i] = language_model.apply(
                    [word, None], prev_token, next_token
                )
            # Use NNP for capitalized words (except in German).
            elif word.istitle() and language_code != "de":
                tagged_tokens[i] = [word, default_tags[1]]
            # Use CD for digits and numbers.
            elif CD.match(word) is not None:
                tagged_tokens[i] = [word, default_tags[2]]
            # Use suffix rules (e.g., -ly = RB).
            elif morphology_analyzer is not None:
                tagged_tokens[i] = morphology_analyzer.apply(
                    [word, default_tags[0]], prev_token, next_token
                )
            # Use suffix rules (English default).
            elif language_code == "en":
                tagged_tokens[i] = suffix_rules([word, default_tags[0]])
            # Use most frequent tag (NN).
            else:
                tagged_tokens[i] = [word, default_tags[0]]
    # Tag words by context.
    if context_analyzer is not None and language_model is None:
        tagged_tokens = context_analyzer.apply(tagged_tokens)
    # Tag named entities.
    if entity_recognizer is not None:
        tagged_tokens = entity_recognizer.apply(tagged_tokens)
    # Map tags with a custom function.
    if tag_mapper is not None:
        tagged_tokens = [
            list(tag_mapper(word, tag)) or [word, default_tags[0]]
            for word, tag in tagged_tokens
        ]
    return tagged_tokens


def find_chunks(
    tagged_tokens: List[Tuple[BaseString, PosTag]], language: str = "en"
) -> List[Tuple[BaseString, PosTag, str]]:
    tokens_with_chunks = [x for x in tagged_tokens]
    tag_sequence = "".join(f"{tag}{SEPARATOR}" for token, tag in tagged_tokens)
    # Use Germanic or Romance chunking rules according to given language
    is_romance = int(language in ("ca", "es", "pt", "fr", "it", "pt", "ro"))
    for chunk_type, chunk_pattern in CHUNKS[is_romance]:
        for match in chunk_pattern.finditer(tag_sequence):
            # Find the start of chunks inside the tags-string.
            # Number of preceding separators = number of preceding tokens.
            match_start = match.start()
            token_start_idx = tag_sequence[:match_start].count(SEPARATOR)
            chunk_length = match.group(0).count(SEPARATOR)
            for token_idx in range(token_start_idx, token_start_idx + chunk_length):
                if len(tokens_with_chunks[token_idx]) == 3:
                    continue
                if len(tokens_with_chunks[token_idx]) < 3:
                    # A conjunction can not be start of a chunk.
                    if token_idx == token_start_idx and tokens_with_chunks[token_idx][
                        1
                    ] in ("CC", "CJ", "KON", "Conj(neven)"):
                        token_start_idx += 1
                    # Mark first token in chunk with B-.
                    elif token_idx == token_start_idx:
                        tokens_with_chunks[token_idx].append("B-" + chunk_type)
                    # Mark other tokens in chunk with I-.
                    else:
                        tokens_with_chunks[token_idx].append("I-" + chunk_type)
    # Mark chinks (tokens outside of a chunk) with O-.
    for unchunked_token in filter(lambda x: len(x) < 3, tokens_with_chunks):
        unchunked_token.append("O")
    # Post-processing corrections.
    for i, (_word, pos_tag, chunk_label) in enumerate(tokens_with_chunks):
        if pos_tag.startswith("RB") and chunk_label == "B-NP":
            # "Very nice work" (NP) <=> "Perhaps" (ADVP) + "you" (NP).
            if i < len(tokens_with_chunks) - 1 and not tokens_with_chunks[i + 1][
                1
            ].startswith("JJ"):
                tokens_with_chunks[i + 0][2] = "B-ADVP"
                tokens_with_chunks[i + 1][2] = "B-NP"
    return tokens_with_chunks


def find_prepositions(token_sequence: List[Tuple[BaseString, PosTag, str]]):
    for token_item in token_sequence:
        token_item.append("O")
    for current_idx, current_token in enumerate(token_sequence):
        is_preposition = current_token[2].endswith("PP") and current_token[-1] == "O"
        if is_preposition:
            has_valid_next_token = current_idx < len(token_sequence) - 1 and (
                token_sequence[current_idx + 1][2].endswith(("NP", "PP"))
                or token_sequence[current_idx + 1][1] in ("VBG", "VBN")
            )
            if has_valid_next_token:
                current_token[-1] = "B-PNP"
                is_prep_phrase = True
                for next_token in token_sequence[current_idx + 1 :]:
                    is_valid_continuation = next_token[2].endswith(
                        ("NP", "PP")
                    ) or next_token[1] in ("VBG", "VBN")
                    if not is_valid_continuation:
                        break
                    if next_token[2].endswith("PP") and is_prep_phrase:
                        next_token[-1] = "I-PNP"
                    if not next_token[2].endswith("PP"):
                        next_token[-1] = "I-PNP"
                        is_prep_phrase = False
    return token_sequence


class Spelling(LazyDict):
    ALPHA = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, path: Path = Path("")):
        self._path = path

    def load(self):
        for x in read(self._path):
            x = x.split()
            dict.__setitem__(self, x[0], int(x[1]))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @classmethod
    def train(self, text: BaseString, output_path: Path = Path("spelling.txt")):
        word_frequencies = {}
        for word in re.findall("[a-z]+", text.lower()):
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        formatted_entries = (
            f"{word} {count}" for word, count in sorted(word_frequencies.items())
        )
        model_content = "\n".join(formatted_entries)

        with open(output_path, "w") as output_file:
            output_file.write(model_content)

    def _edit1(self, word: BaseString) -> Set[BaseString]:
        # Of all spelling errors, 80% is covered by edit distance 1.
        # Edit distance 1 = one character deleted, swapped, replaced or inserted.
        word_splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletions = [prefix + suffix[1:] for prefix, suffix in word_splits if suffix]
        transpositions = [
            prefix + suffix[1] + suffix[0] + suffix[2:]
            for prefix, suffix in word_splits
            if len(suffix) > 1
        ]
        replacements = [
            prefix + new_char + suffix[1:]
            for prefix, suffix in word_splits
            for new_char in Spelling.ALPHA
            if suffix
        ]
        insertions = [
            prefix + new_char + suffix
            for prefix, suffix in word_splits
            for new_char in Spelling.ALPHA
        ]
        return set(deletions + transpositions + replacements + insertions)

    def _edit2(self, word: BaseString) -> Set[BaseString]:
        # Of all spelling errors, 99% is covered by edit distance 2.
        # Only keep candidates that are actually known words (20% speedup).
        return set(
            second_edit
            for first_edit in self._edit1(word)
            for second_edit in self._edit1(first_edit)
            if second_edit in self
        )

    def _known(self, candidate_words: Sequence[BaseString] = None) -> Set[BaseString]:
        if candidate_words is None:
            candidate_words = []
        return set(word for word in candidate_words if word in self)

    def suggest(self, word: BaseString) -> List[Tuple[BaseString, float]]:
        if len(self) == 0:
            self.load()
        if len(word) == 1:
            return [(word, 1.0)]  # I
        if word in PUNCTUATION:
            return [(word, 1.0)]  # .?!
        if word in string.whitespace:
            return [(word, 1.0)]  # \n
        if word.replace(".", "").isdigit():
            return [(word, 1.0)]  # 1.5
        # Find valid word candidates with increasing edit distance
        spelling_candidates = (
            self._known([word])
            or self._known(self._edit1(word))
            or self._known(self._edit2(word))
            or [word]
        )
        # Get frequency scores for each candidate
        scored_candidates = [
            (self.get(candidate, 0.0), candidate) for candidate in spelling_candidates
        ]
        # Normalize scores to probabilities
        total_frequency = float(sum(freq for freq, word in scored_candidates) or 1)
        normalized_candidates = sorted(
            ((freq / total_frequency, word) for freq, word in scored_candidates),
            reverse=True,
        )
        # Preserve original capitalization
        if word.istitle():
            final_candidates = [
                (word.title(), probability)
                for probability, word in normalized_candidates
            ]
        else:
            final_candidates = [
                (word, probability) for probability, word in normalized_candidates
            ]

        return final_candidates


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


class Score(tuple):
    def __new__(
        self, polarity: float, subjectivity: float, assessments: Iterable[float] = None
    ):
        if assessments is None:
            assessments = []
        return tuple.__new__(self, [polarity, subjectivity])

    def __init__(
        self, polarity: float, subjectivity: float, assessments: Iterable[float] = None
    ):
        if assessments is None:
            assessments = []
        self.assessments = assessments


class Sentiment(LazyDict):
    def __init__(
        self,
        path: Path = Path(""),
        language: str = None,
        synset: str = None,
        confidence: float = None,
        **kwargs,
    ):
        self._path = path  # XML file path.
        self._language = None  # XML language attribute ("en", "fr", ...)
        self._confidence = None  # XML confidence attribute threshold (>=).
        self._synset = synset  # XML synset attribute ("wordnet_id", "cornetto_id", ...)
        self._synsets = {}  # {"a-01123879": (1.0, 1.0, 1.0)}
        self.labeler = {}  # {"dammit": "profanity"}
        self.tokenizer = kwargs.get("tokenizer", find_tokens)
        self.negations = kwargs.get("negations", ("no", "not", "n't", "never"))
        self.modifiers = kwargs.get("modifiers", ("RB",))
        self.modifier = kwargs.get("modifier", lambda w: w.endswith("ly"))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @property
    def confidence(self):
        return self._confidence

    def load(self, file_path: Path = None):
        if not file_path:
            file_path = self._path
        if not file_path.exists():
            return
        word_scores = {}
        synset_scores = {}
        word_labels = {}
        xml_tree = ElementTree.parse(file_path)
        root = xml_tree.getroot()
        for word_elem in root.findall("word"):
            if self._confidence is None or self._confidence <= float(
                word_elem.attrib.get("confidence", 0.0)
            ):
                (
                    word_form,
                    part_of_speech,
                    polarity,
                    subjectivity,
                    intensity,
                    label,
                    synset_id,
                ) = (
                    word_elem.attrib.get("form"),
                    word_elem.attrib.get("pos"),
                    word_elem.attrib.get("polarity", 0.0),
                    word_elem.attrib.get("subjectivity", 0.0),
                    word_elem.attrib.get("intensity", 1.0),
                    word_elem.attrib.get("label"),
                    word_elem.attrib.get(self._synset),  # wordnet_id, cornetto_id, ...
                )

                sentiment_scores = (
                    float(polarity),
                    float(subjectivity),
                    float(intensity),
                )

                if word_form:
                    word_scores.setdefault(word_form, {}).setdefault(
                        part_of_speech, []
                    ).append(sentiment_scores)
                if word_form and label:
                    word_labels[word_form] = label
                if synset_id:
                    synset_scores.setdefault(synset_id, []).append(sentiment_scores)

        self._language = root.attrib.get("language", self._language)
        for word in word_scores:
            word_scores[word] = dict(
                (pos, [avg(each) for each in zip(*scores)])
                for pos, scores in word_scores[word].items()
            )
        for word, pos_scores in list(word_scores.items()):
            word_scores[word][None] = [avg(each) for each in zip(*pos_scores.values())]
        for synset_id, scores in synset_scores.items():
            synset_scores[synset_id] = [avg(each) for each in zip(*scores)]
        dict.update(self, word_scores)
        dict.update(self.labeler, word_labels)
        dict.update(self._synsets, synset_scores)

    def synset(
        self, synset_id: str, part_of_speech: PosTag = ADJECTIVE
    ) -> Tuple[float, float]:
        padded_id = str(synset_id).zfill(8)
        if not padded_id.startswith(("n-", "v-", "a-", "r-")):
            if part_of_speech == NOUN:
                padded_id = "n-" + padded_id
            if part_of_speech == VERB:
                padded_id = "v-" + padded_id
            if part_of_speech == ADJECTIVE:
                padded_id = "a-" + padded_id
            if part_of_speech == ADVERB:
                padded_id = "r-" + padded_id
        if dict.__len__(self) == 0:
            self.load()
        return tuple(self._synsets.get(padded_id, (0.0, 0.0))[:2])

    def __call__(
        self, input_text: BaseString, negation: bool = True, **kwargs
    ) -> Score:
        def calculate_weighted_average(
            sentiment_assessments: Iterable[Tuple[List[str], float]],
            weighting_function: Callable = lambda w: 1,
        ):
            sum_weighted_scores, total_weight = 0, 0
            for words, score in sentiment_assessments:
                weight = weighting_function(words)
                sum_weighted_scores += weight * score
                total_weight += weight
            return sum_weighted_scores / float(total_weight or 1)

        if hasattr(input_text, "gloss"):
            assessments = [
                (input_text.synonyms[0],)
                + self.synset(input_text.id, pos=input_text.pos)
                + (None,)
            ]
        elif (
            isinstance(input_text, BaseString)
            and RE_SYNSET.match(input_text)
            and hasattr(input_text, "synonyms")
        ):
            assessments = [
                (input_text.synonyms[0],)
                + self.synset(input_text.id, pos=input_text.pos)
                + (None,)
            ]
        elif isinstance(input_text, BaseString):
            assessments = self.assessments(
                (
                    (word.lower(), None)
                    for word in " ".join(self.tokenizer(input_text)).split()
                ),
                negation,
            )
        elif hasattr(input_text, "sentences"):
            assessments = self.assessments(
                (
                    (word.lemma or word.string.lower(), word.pos[:2])
                    for word in chain.from_iterable(input_text)
                ),
                negation,
            )
        elif hasattr(input_text, "lemmata"):
            assessments = self.assessments(
                (
                    (word.lemma or word.string.lower(), word.pos[:2])
                    for word in input_text.words
                ),
                negation,
            )
        elif hasattr(input_text, "lemma"):
            assessments = self.assessments(
                ((input_text.lemma or input_text.string.lower(), input_text.pos[:2]),),
                negation,
            )
        elif hasattr(input_text, "terms"):
            assessments = self.assessments(
                chain.from_iterable(
                    ((word, None), (None, None)) for word in input_text
                ),
                negation,
            )
            kwargs.setdefault("weight", lambda w: input_text.terms[w[0]])
        elif isinstance(input_text, dict):
            assessments = self.assessments(
                chain.from_iterable(
                    ((word, None), (None, None)) for word in input_text
                ),
                negation,
            )
            kwargs.setdefault("weight", lambda w: input_text[w[0]])
        elif isinstance(input_text, list):
            assessments = self.assessments(
                ((word, None) for word in input_text), negation
            )
        else:
            assessments = []

        weighting_function = kwargs.get("weight", lambda w: 1)
        return Score(
            polarity=calculate_weighted_average(
                [
                    (words, polarity)
                    for words, polarity, subjectivity, label in assessments
                ],
                weighting_function,
            ),
            subjectivity=calculate_weighted_average(
                [
                    (words, subjectivity)
                    for words, polarity, subjectivity, label in assessments
                ],
                weighting_function,
            ),
            assessments=assessments,
        )

    def assessments(
        self, words: Iterable[Tuple[str, str]], negation: bool = True
    ) -> List[Tuple[List[str], float, float, str]]:
        if words is None:
            words = []
        assessments_list = []
        modifier = None  # Preceding modifier (i.e., adverb or adjective).
        negation_word = None  # Preceding negation (e.g., "not beautiful").
        for word, pos_tag in words:
            # Only assess known words, preferably by part-of-speech tag.
            # Including unknown words (polarity 0.0 and subjectivity 0.0) lowers the average.
            if word is None:
                continue
            if word in self and pos_tag in self[word]:
                polarity, subjectivity, intensity = self[word][pos_tag]
                # Known word not preceded by a modifier ("good").
                if modifier is None:
                    assessments_list.append(
                        dict(
                            words=[word],
                            polarity=polarity,
                            subjectivity=subjectivity,
                            intensity=intensity,
                            negation_factor=1,
                            label=self.labeler.get(word),
                        )
                    )
                # Known word preceded by a modifier ("really good").
                if modifier is not None:
                    assessments_list[-1]["words"].append(word)
                    assessments_list[-1]["polarity"] = max(
                        -1.0, min(polarity * assessments_list[-1]["intensity"], +1.0)
                    )
                    assessments_list[-1]["subjectivity"] = max(
                        -1.0,
                        min(subjectivity * assessments_list[-1]["intensity"], +1.0),
                    )
                    assessments_list[-1]["intensity"] = intensity
                    assessments_list[-1]["label"] = self.labeler.get(word)
                # Known word preceded by a negation ("not really good").
                if negation_word is not None:
                    assessments_list[-1]["words"].insert(0, negation_word)
                    assessments_list[-1]["intensity"] = (
                        1.0 / assessments_list[-1]["intensity"]
                    )
                    assessments_list[-1]["negation_factor"] = -1
                # Known word may be a negation.
                # Known word may be modifying the next word (i.e., it is a known adverb).
                modifier = None
                negation_word = None
                if (
                    pos_tag
                    and pos_tag in self.modifiers
                    or any(map(self[word].__contains__, self.modifiers))
                ):
                    modifier = (word, pos_tag)
                if negation and word in self.negations:
                    negation_word = word
            else:
                # Unknown word may be a negation ("not good").
                if negation and word in self.negations:
                    negation_word = word
                # Unknown word. Retain negation across small words ("not a good").
                elif negation_word and len(word.strip("'")) > 1:
                    negation_word = None
                # Unknown word may be a negation preceded by a modifier ("really not good").
                if (
                    negation_word is not None
                    and modifier is not None
                    and (pos_tag in self.modifiers or self.modifier(modifier[0]))
                ):
                    assessments_list[-1]["words"].append(negation_word)
                    assessments_list[-1]["negation_factor"] = -1
                    negation_word = None
                # Unknown word. Retain modifier across small words ("really is a good").
                elif modifier and len(word) > 2:
                    modifier = None
                # Exclamation marks boost previous word.
                if word == "!" and len(assessments_list) > 0:
                    assessments_list[-1]["words"].append("!")
                    assessments_list[-1]["polarity"] = max(
                        -1.0, min(assessments_list[-1]["polarity"] * 1.25, +1.0)
                    )
                # Exclamation marks in parentheses indicate sarcasm.
                if word == "(!)":
                    assessments_list.append(
                        dict(
                            words=[word],
                            polarity=0.0,
                            subjectivity=1.0,
                            intensity=1.0,
                            negation_factor=1,
                            label=IRONY,
                        )
                    )
                # EMOTICONS: {("grin", +1.0): set((":-D", ":D"))}
                if (
                    word.isalpha() is False
                    and len(word) <= 5
                    and word not in PUNCTUATION
                ):  # speedup
                    for (
                        emoticon_type,
                        emoticon_polarity,
                    ), emoticons in EMOTICONS.items():
                        if word in map(lambda e: e.lower(), emoticons):
                            assessments_list.append(
                                dict(
                                    words=[word],
                                    polarity=emoticon_polarity,
                                    subjectivity=1.0,
                                    intensity=1.0,
                                    negation_factor=1,
                                    label=MOOD,
                                )
                            )
                            break
        final_assessments = []
        for assessment in assessments_list:
            words = assessment["words"]
            polarity = assessment["polarity"]
            subjectivity = assessment["subjectivity"]
            negation_factor = assessment["negation_factor"]
            label = assessment["label"]
            # "not good" = slightly bad, "not bad" = slightly good.
            final_assessments.append(
                (
                    words,
                    polarity * -0.5 if negation_factor < 0 else polarity,
                    subjectivity,
                    label,
                )
            )
        return final_assessments

    def annotate(
        self,
        word: BaseString,
        pos: PosTag = None,
        polarity: float = 0.0,
        subjectivity: float = 0.0,
        intensity: float = 1.0,
        label: str = None,
    ):
        word_scores = self.setdefault(word, {})
        sentiment_tuple = (polarity, subjectivity, intensity)
        word_scores[pos] = word_scores[None] = sentiment_tuple
        if label:
            self.labeler[word] = label
