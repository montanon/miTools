import codecs
import re
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

from nltk.tree import Tree

from mitools.exceptions import ArgumentStructureError
from mitools.nlp.definitions import (
    ABBREVIATIONS,
    CD,
    CHUNKS,
    EOS,
    PUNCTUATION,
    RE_ABBR1,
    RE_ABBR2,
    RE_ABBR3,
    RE_EMOTICONS,
    RE_SARCASM,
    REPLACEMENTS,
    SEPARATOR,
    TOKEN,
)
from mitools.nlp.nlp_typing import PennTag, PosTag, WordNetTag
from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.utils.helper_functions import decode_string, strip_punctuation


def find_relations(tokens: Sequence[str]):
    raise NotImplementedError("find_relations() is not implemented.")


def avg(list: Iterable[float]) -> float:
    return sum(list) / float(len(list) or 1)


def suggest(word: str) -> str:
    raise NotImplementedError("suggest() is not implemented.")


def penn_to_wordnet(penn_tag: PennTag) -> WordNetTag:
    raise NotImplementedError("penn_to_wordnet() is not implemented.")


def word_tokenize(
    text: str,
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
        for sentence in sentence_tokenize(text, sentence_tokenizer, *args, **kwargs)
    )
    return words


def sentence_tokenize(
    text: str,
    sentence_tokenizer: SentenceTokenizer = None,
    *args,
    **kwargs,
):
    sentence_tokenizer = (
        sentence_tokenizer if sentence_tokenizer is not None else SentenceTokenizer()
    )
    return sentence_tokenizer.itokenize(text, *args, **kwargs)


def get_words_from_corpus(
    corpus: Union[str, Sequence[str]],
    word_tokenizer: WordTokenizer = None,
    sentence_tokenizer: SentenceTokenizer = None,
    include_punctuation: bool = False,
    *args,
    **kwargs,
) -> Set[str]:
    def tokenize(words, word_tokenizer, sentence_tokenizer, *args, **kwargs):
        if isinstance(words, (str, bytes)):
            return word_tokenize(
                words,
                word_tokenizer=word_tokenizer,
                sentence_tokenizer=sentence_tokenizer,
                include_punctuation=include_punctuation,
                *args,
                **kwargs,
            )
        return words

    if isinstance(corpus, str):
        corpus = [corpus]
    all_words = chain.from_iterable(
        tokenize(words, word_tokenizer, sentence_tokenizer, *args, **kwargs)
        for words in corpus
    )
    return set(all_words)


def get_corpus_tokens(corpus: Iterable[str]) -> Set[str]:
    if isinstance(corpus, (str, bytes)):
        tokens = set(
            strip_punctuation(word, all=False)
            for word in word_tokenize(corpus, include_punctuation=False)
        )
    else:
        tokens = set(strip_punctuation(word, all=False) for word in corpus)
    return tokens


def basic_extractor(corpus: Iterable[str], train_set: Iterable[str]) -> Dict[str, bool]:
    try:
        zero_item = next(iter(train_set))
    except StopIteration:
        return {}
    if isinstance(zero_item, (str, bytes)):
        word_features = [word for word in chain([zero_item], train_set)]
    else:
        try:
            assert isinstance(zero_item[0], (str, bytes))
            word_features = get_words_from_corpus(chain([zero_item], train_set))
        except Exception as error:
            raise ArgumentStructureError("train_set is probably malformed.") from error
    tokens = get_corpus_tokens(corpus)
    features = dict((f"contains({word})", (word in tokens)) for word in word_features)
    return features


def contains_extractor(corpus: Iterable[str]) -> Dict[str, bool]:
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


def is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def suffix_rules(token: str, tag: PosTag = "NN"):
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


def find_tokens(
    text: str,
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


def read_text(
    source: Union[Path, str, TextIO, Iterable[str]], encoding="utf-8", comment=";;;"
):
    if source is None:
        return None
    if isinstance(source, (str, bytes)) and Path(source).exists():
        f = open(source, encoding=encoding)
    elif isinstance(source, (str, bytes)):
        f = source.splitlines()
    elif hasattr(source, "read"):
        f = source.read().splitlines()
    else:
        f = source
    for i, line in enumerate(f):
        line = (
            line.strip(codecs.BOM_UTF8) if i == 0 and isinstance(line, bytes) else line
        )
        line = line.strip()
        line = decode_string(line, encoding=encoding)
        if not line or (comment and line.startswith(comment)):
            continue
        yield line


def find_tags(
    word_tokens,
    lexicon: Dict[str, PosTag] = None,
    language_model: Callable = None,
    morphology_analyzer: Any = None,
    context_analyzer: Any = None,
    entity_recognizer: Any = None,
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
    tagged_tokens: List[Tuple[str, PosTag]], language: str = "en"
) -> List[Tuple[str, PosTag, str]]:
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


def find_prepositions(token_sequence: List[Tuple[str, PosTag, str]]):
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
