import pickle
import re
import string
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Sequence, Set, Tuple, Union
from xml.etree import ElementTree

from nltk.corpus import stopwords

from mitools.nlp.definitions import (
    ABBREVIATIONS,
    ADJECTIVE,
    ADVERB,
    EMOTICONS,
    IRONY,
    MOOD,
    NOUN,
    PUNCTUATION,
    RE_ENTITY1,
    RE_ENTITY2,
    RE_ENTITY3,
    RE_SYNSET,
    REPLACEMENTS,
    TOKENS,
    VERB,
)
from mitools.nlp.nlp_typing import BaseString, PosTag
from mitools.nlp.utils import (
    avg,
    find_chunks,
    find_prepositions,
    find_relations,
    find_tags,
    find_tokens,
    read_text,
)
from mitools.utils.helper_functions import decode_string
from mitools.utils.helper_objects import LazyDict, LazyList


class StopwordsManager:
    def __init__(self, language="english"):
        self.language = language
        self._words = set(stopwords.words(language))

    def add_stopword(self, word):
        self._words.add(word)

    def add_stopwords(self, words):
        self._words.update(words)

    def remove_stopword(self, word):
        self._words.discard(word)

    def remove_stopwords(self, words):
        self._words.difference_update(words)

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    @property
    def words(self):
        return list(self._words)


class TaggedString(str):
    def __new__(
        self,
        string: BaseString,
        tags: Sequence[PosTag] = None,
        language: Literal["en"] = None,
    ):
        if tags is None:
            tags = ["word"]
        if isinstance(string, str) and hasattr(string, "tags"):
            tags, language = string.tags, string.language
        if isinstance(string, list):
            string = [
                [[x.replace("/", "&slash;") for x in token] for token in s]
                for s in string
            ]
            string = "\n".join(" ".join("/".join(token) for token in s) for s in string)
        s = str.__new__(self, string)
        s.tags = list(tags)
        s.language = language
        return s

    def split(self, sep: str = TOKENS):
        if sep != TOKENS:
            return str.split(self, sep)
        if len(self) == 0:
            return []
        return [
            [
                [x.replace("&slash;", "/") for x in token.split("/")]
                for token in sentence.split(" ")
            ]
            for sentence in str.split(self, "\n")
        ]


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
        dict.update(
            self, (x.split(" ")[:2] for x in read_text(self._path) if x.strip())
        )

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
    def __init__(self, lexicon: Union[Dict[str, str], None] = None, path: Path = None):
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
        list.extend(self, (x.split() for x in read_text(self._path)))

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
        list.extend(self, (x.split() for x in read_text(self._path)))

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
        for x in read_text(self.path):
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


class Spelling(LazyDict):
    ALPHA = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, path: Path = Path("")):
        self._path = path

    def load(self):
        for x in read_text(self._path):
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
            punctuation_chars=kwargs.get("punctuation_chars", PUNCTUATION),
            known_abbreviations=kwargs.get("known_abbreviations", ABBREVIATIONS),
            contraction_replacements=kwargs.get(
                "contraction_replacements", REPLACEMENTS
            ),
            paragraph_break=r"\n{2,}",
        )

    def find_tags(self, tokens: Sequence[BaseString], **kwargs):
        # ["The", "cat", "purs"] => [["The", "DT"], ["cat", "NN"], ["purs", "VB"]]
        return find_tags(
            word_tokens=tokens,
            language_code=kwargs.get("language_code", self.language),
            lexicon=kwargs.get("lexicon", self.lexicon),
            default_tags=kwargs.get("default_tags", self.default),
            tag_mapper=kwargs.get("tag_mapper", None),
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
                isinstance(text, (str, bytes)) and text.split(" ") or text
                for text in text
            ]
        if isinstance(text, (str, bytes)):
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
            isinstance(input_text, (str, bytes))
            and RE_SYNSET.match(input_text)
            and hasattr(input_text, "synonyms")
        ):
            assessments = [
                (input_text.synonyms[0],)
                + self.synset(input_text.id, pos=input_text.pos)
                + (None,)
            ]
        elif isinstance(input_text, (str, bytes)):
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
