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
