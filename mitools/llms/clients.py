import os
from typing import Dict

import openai
from openai import OpenAI

from mitools.llms.objects import LLMModel, Prompt


class OpenAIClient(LLMModel):
    _roles = ["system", "user", "assistant", "tool", "function"]

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(
            api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )

    def parse_request(self, prompt: Prompt) -> Dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
        }

    def request(self, request: Prompt, **kwargs) -> Dict:
        request = self.parse_request(request)
        response = self.get_response(request, **kwargs)
        return self.parse_response(response)

    def parse_response(self, response: Dict) -> str:
        return response.choices[0].message

    def get_model_info(self) -> Dict:
        return {"name": "OpenAI", "model": self.model}

    def get_response(self, request: Dict, **kwargs) -> Dict:
        return self.client.chat.completions.create(**request, **kwargs)

    def model_name(self) -> str:
        return self.model
