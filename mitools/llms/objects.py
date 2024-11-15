from abc import ABC, abstractmethod
from typing import Dict


class LMMModel(ABC):
    @abstractmethod
    def request(self, prompt: str, **kwargs) -> Dict:
        pass

    @abstractmethod
    def parse_response(self, response: Dict) -> str:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        pass

    @abstractmethod
    def model_name(self) -> str:
        pass
