from typing import List, Literal, Tuple, Union

PosTag = Literal["tags"]
PennTag = Literal["tags"]
WordNetTag = Literal["tags"]
BaseString = Union[str, bytes]
SentimentType = Union[Tuple[float, float], Tuple[float, float, List[str]]]
