import re
import string

PUNCTUATION_REGEX = re.compile(
    "[" + "".join(re.escape(p) for p in string.punctuation) + "]"
)


def strip_punctuation(s: str, all: bool = False) -> str:
    if all:
        return PUNCTUATION_REGEX.sub("", s.strip())
    else:
        return s.strip().strip(string.punctuation)
