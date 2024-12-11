from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.api import StemmerI
from nltk.stem.snowball import EnglishStemmer, SpanishStemmer

from mitools.nlp.nlp_typing import BaseString, PosTag
from mitools.nlp.taggers import BaseTagger, NLTKTagger
from mitools.nlp.tokenizers import BaseTokenizer, WordTokenizer
from mitools.nlp.utils import nltk_tag_to_wordnet_tag


class BaseLemmatizer(ABCMeta):
    @abstractmethod
    def lemmatize(self, text: BaseString, pos: PosTag = "n") -> Sequence[BaseString]:
        pass

    def lemmatize_tokens(
        self, tokens: Union[Sequence[BaseString], Sequence[Tuple[BaseString, PosTag]]]
    ) -> Sequence[BaseString]:
        return [
            self.lemmatize(token[0], token[1])
            if isinstance(token, tuple)
            else self.lemmatize(token)
            for token in tokens
        ]

    def lemmatize_text(
        self, text: BaseString, tokenizer: BaseTokenizer = None
    ) -> Sequence[BaseString]:
        return (
            self.lemmatize_tokens(text.split())
            if tokenizer is None
            else self.lemmatize_tokens(tokenizer.tokenize(text))
        )


class WordnetLemmatizer(BaseLemmatizer):
    _instance = None
    _lemmatizer = WordNetLemmatizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def lemmatize(self, text: BaseString, pos: PosTag = "n") -> Sequence[BaseString]:
        return self._lemmatizer.lemmatize(text, pos)


class NLTKLemmatizer(BaseLemmatizer):
    _instance = None
    _lemmatizer = WordNetLemmatizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def lemmatize(
        self,
        text: BaseString,
        tokenizer: Optional[BaseTokenizer] = None,
        tagger: Optional[BaseTagger] = None,
    ) -> Sequence[str]:
        tokenizer = tokenizer if tokenizer is not None else WordTokenizer()
        tagger = tagger if tagger is not None else NLTKTagger()
        tagged_tokens = tagger.tag(text, tokenizer=tokenizer)
        return [
            self._lemmatizer.lemmatize(word, pos=nltk_tag_to_wordnet_tag(tag))
            for word, tag in tagged_tokens
        ]


class BaseStemmer(StemmerI, metaclass=ABCMeta):
    @abstractmethod
    def stem(self, text: BaseString) -> Sequence[BaseString]:
        pass

    def stem_tokens(self, tokens: Sequence[BaseString]) -> Sequence[BaseString]:
        return [self.stem(token) for token in tokens]

    def stem_text(
        self, text: BaseString, tokenizer: BaseTokenizer = None
    ) -> Sequence[BaseString]:
        return (
            self.stem_tokens(text.split())
            if tokenizer is None
            else self.stem_tokens(tokenizer.tokenize(text))
        )


class PorterStemmer(BaseStemmer):
    _instance = None
    _stemmer = PorterStemmer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stem(self, text: BaseString) -> Sequence[BaseString]:
        return self._stemmer.stem(text)


class SpanishStemmer(BaseStemmer):
    _instance = None
    _stemmer = SpanishStemmer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stem(self, text: BaseString) -> Sequence[BaseString]:
        return self._stemmer.stem(text)


class EnglishStemmer(BaseStemmer):
    _instance = None
    _stemmer = EnglishStemmer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def stem(self, text: BaseString) -> Sequence[BaseString]:
        return self._stemmer.stem(text)
