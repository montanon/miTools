from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError


class Prompt:
    def __init__(self, text: str, metadata: Optional[Dict[str, str]] = None):
        if not isinstance(text, str) or not text.strip():
            raise ArgumentValueError("Prompt must be a non-empty string.")
        self.text = text.strip()
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Prompt(\ntext={self.text},\n metadata={self.metadata}\n)"

    def __add__(self, other: Union["Prompt", str]) -> "Prompt":
        if isinstance(other, Prompt):
            combined_text = f"{self.text}\n{other.text}"
            combined_metadata = {
                **other.metadata,
                **self.metadata,
            }  # self.metadata has priority over other.metadata
            return Prompt(combined_text, combined_metadata)
        elif isinstance(other, str):
            combined_text = f"{self.text}\n{other.strip()}"
            return Prompt(combined_text, self.metadata)
        else:
            raise ArgumentTypeError("Can only concatenate with a Prompt or a string.")

    def __iadd__(self, other: Union["Prompt", str]) -> "Prompt":
        concatenated = self + other
        self.text = concatenated.text
        self.metadata = concatenated.metadata
        return self

    def format(self, **kwargs) -> "Prompt":
        try:
            formatted_text = self.text.format(**kwargs)
            return Prompt(text=formatted_text, metadata=self.metadata)
        except KeyError as e:
            raise ArgumentKeyError(f"String missing formatting key: {e}")

    def update_metadata(self, key: str, value: str) -> None:
        if not isinstance(key, str) or not isinstance(value, str):
            raise ArgumentValueError("Metadata keys and values must be strings.")
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[str]:
        return self.metadata.get(key)

    def to_dict(self) -> Dict[str, Union[str, Dict[str, str]]]:
        return {"text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, Dict[str, str]]]) -> "Prompt":
        if "text" not in data:
            raise ArgumentValueError("Dictionary must contain a 'text' key.")
        return cls(text=data["text"], metadata=data.get("metadata", {}))

    @staticmethod
    def concatenate(
        prompts: List[Union["Prompt", str]], separator: str = "\n"
    ) -> "Prompt":
        combined_texts = []
        combined_metadata = {}
        for prompt in prompts:
            if isinstance(prompt, Prompt):
                combined_texts.append(prompt.text)
                combined_metadata.update(prompt.metadata)
            elif isinstance(prompt, str):
                combined_texts.append(prompt.strip())
            else:
                raise ArgumentTypeError(
                    "List must contain only Prompt instances or strings."
                )
        return Prompt(text=separator.join(combined_texts), metadata=combined_metadata)


class LLMModel(ABC):
    @abstractmethod
    def request(self, request: Prompt, **kwargs) -> Dict:
        pass

    @abstractmethod
    def parse_request(self, prompt: Prompt, **kwargs) -> Dict:
        pass

    @abstractmethod
    def get_response(self, prompt) -> Dict:
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


class LLMFactory:
    def __init__(self):
        self.registry = {}

    def register_client(self, name, client_class):
        self.registry[name] = client_class

    def get_client(self, name, **kwargs):
        if name not in self.registry:
            raise ValueError(f"Model '{name}' not supported.")
        return self.registry[name](**kwargs)
