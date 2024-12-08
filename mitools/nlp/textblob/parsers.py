from abc import ABCMeta, abstractmethod

from nltk.tree import Tree


class BaseParser(ABCMeta):
    @abstractmethod
    def parse(self, text: str) -> Tree:
        pass
