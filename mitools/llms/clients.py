import os
from datetime import datetime
from typing import Dict

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from mitools.llms.objects import LLMModel, Prompt, TokensCounter, TokenUsageStats


class OpenAIClient(LLMModel):
    _roles = ["system", "user", "assistant", "tool", "function"]

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(
            api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        self.raw_responses = []

    def parse_request(self, prompt: Prompt) -> Dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
        }

    def request(self, request: Prompt, **kwargs) -> Dict:
        request = self.parse_request(request)
        response = self.get_response(request, **kwargs)
        self.raw_responses.append(response)
        return self.parse_response(response)

    def parse_response(self, response: Dict) -> str:
        return response.choices[0].message

    def get_model_info(self) -> Dict:
        return {"name": "OpenAI", "model": self.model}

    def get_response(self, request: Dict, **kwargs) -> Dict:
        return self.client.chat.completions.create(**request, **kwargs)

    def model_name(self) -> str:
        return self.model


class OpenAITokensCounter(TokensCounter):
    def get_usage_stats(self, response: ChatCompletion) -> TokenUsageStats:
        total_tokens = response.usage.total_tokens
        return TokenUsageStats(
            total_tokens=total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=total_tokens * (self.cost_per_1k_tokens / 1000),
            timestamp=datetime.fromtimestamp(response.created),
        )
