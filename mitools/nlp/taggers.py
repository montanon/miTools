from abc import ABC, abstractmethod
from typing import Iterator, Literal, Sequence, Tuple, Union

import nltk

from mitools.nlp.en import tag as pattern_tag
from mitools.nlp.nlp_typing import PosTag
from mitools.nlp.tags_translator import translate_tags


class BaseTagger(ABC):
    @abstractmethod
    def tag_tokens(
        self, tokens: Union[str, Sequence[str]]
    ) -> Sequence[Tuple[str, PosTag]]:
        pass

    def itag_tokens(
        self, tokens: Union[str, Sequence[str]]
    ) -> Iterator[Tuple[str, PosTag]]:
        return (t for t in self.tag_tokens(tokens))


class NLTKTagger(BaseTagger):
    _instance = None
    _tagger = staticmethod(nltk.tag.pos_tag)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def tag_tokens(
        self,
        tokens: Union[str, Sequence[str]],
        tags_format: Literal["penn", "universal", "wordnet"] = "penn",
    ) -> Sequence[Tuple[str, PosTag]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        tagged_tokens = self._tagger(tokens)
        if tags_format != "penn":
            tagged_tokens = translate_tags(
                tagged_tokens, source_format="penn", target_format=tags_format
            )
        return tagged_tokens
