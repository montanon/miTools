from itertools import chain
from typing import (
    Iterable,
    Sequence,
    Set,
    Union,
)

from mitools.nlp.tokenizers import SentenceTokenizer, WordTokenizer
from mitools.utils.helper_functions import strip_punctuation


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
