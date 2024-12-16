from typing import List, Literal, Tuple, Union

PosTag = Literal["tags"]
PennTag = Literal["tags"]
WordNetTag = Literal["tags"]
NLTKTag = Literal["tags"]
UniversalTag = Literal["tags"]
SentimentType = Union[Tuple[float, float], Tuple[float, float, List[str]]]
