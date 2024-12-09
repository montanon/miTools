from typing import List, Literal, Tuple, Union

from nltk.corpus import wordnet

PosTag = Literal["tags"]
PennTag = Literal["tags"]
WordNetTag = Literal["tags"]
BaseString = Union[str, bytes]
SentimentType = Union[Tuple[float, float], Tuple[float, float, List[str]]]
VERB, NOUN, ADJ, ADV = wordnet.VERB, wordnet.NOUN, wordnet.ADJ, wordnet.ADV
