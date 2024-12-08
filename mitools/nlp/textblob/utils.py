import re
import string
from typing import Sequence, Tuple

from nltk.tree import Tree

PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")


def strip_punctuation(s: str, all: bool = False) -> str:
    if all:
        return PUNCTUATION_REGEX.sub("", s.strip())
    else:
        return s.strip().strip(string.punctuation)


def lowerstrip(s: str, all: bool = False) -> str:
    return strip_punctuation(s.lower().strip(), all=all)


def tree_to_string(tree: Tree, concat: str = " ") -> str:
    return concat.join([word for (word, tag) in tree])


def filter_insignificant(
    chunk: Sequence[str], tag_suffixes: Tuple[str] = ("DT", "CC", "PRP$", "PRP")
) -> bool:
    good = []
    for word, tag in chunk:
        ok = True
        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                ok = False
                break
        if ok:
            good.append((word, tag))
    return good


def is_filelike(obj: object) -> bool:
    return hasattr(obj, "read")


class CachedProperty:
    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
