from abc import ABCMeta, abstractmethod
from typing import Sequence


class BaseNPExtractor(ABCMeta):
    @abstractmethod
    def extract(self, text: str) -> Sequence[str]:
        pass
