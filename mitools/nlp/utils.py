import codecs
import re
import string
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Sequence, Set, Tuple, Union

from nltk.tree import Tree

from mitools.exceptions import ArgumentStructureError
from mitools.nlp.blobs import Word
from mitools.nlp.definitions import (
    ABBREVIATIONS,
    ADJ,
    ADV,
    CONJ,
    DET,
    EOS,
    INTJ,
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
    REPLACEMENTS,
    TOKEN,
    VERB,
    X,
)
from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.nlp.typing import BaseString, PennTag, PosTag, WordNetTag
from mitools.utils.helper_objects import LazyDict, LazyList


def avg(list: Iterable[float]) -> float:
    return sum(list) / float(len(list) or 1)


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
