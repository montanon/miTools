import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from mitools.llms.objects import (
    LLMModel,
    PersistentTokensCounter,
    Prompt,
    TokensCounter,
    TokenUsageStats,
)
from mitools.utils import ArgumentValueError


class OpenAIClient(LLMModel):
    _roles = ["system", "user", "assistant", "tool", "function"]

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        counter: "OpenAITokensCounter" = None,
        beta: bool = False,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        self.raw_responses = []
        self.counter = counter
        self.beta = beta

    def parse_request(self, prompt: Prompt) -> Dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.text}],
        }

    def request(self, request: Prompt, **kwargs) -> Dict:
        request = self.parse_request(request)
        response = self._get_response(request, **kwargs)
        self.raw_responses.append(response)
        if self.counter is not None:
            usage = self.counter.get_usage_stats(response)
            self.counter.update(usage)
        return self.parse_response(response)

    def parse_response(self, response: Dict) -> str:
        return response.choices[0].message

    def get_model_info(self) -> Dict:
        return {"name": "OpenAI", "model": self.model}

    def _get_response(self, request: Dict, **kwargs) -> Dict:
        if not self.beta:
            return self.client.chat.completions.create(**request, **kwargs)
        else:
            return self.client.beta.chat.completions.parse(**request, **kwargs)

    def model_name(self) -> str:
        return self.model


class OpenAITokensCounter(PersistentTokensCounter):
    COST_PER_1M_TOKENS = {
        "gpt-3.5-turbo": {"input": 3.0, "output": 8.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "o1-preview": {"input": 15.0, "output": 60.0},
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
    }

    def __init__(self, file_path: Path, model: str = "gpt-4o-mini"):
        if model not in self.COST_PER_1M_TOKENS.keys():
            raise ArgumentValueError(
                f"Model {model} not supported, must be one of {self.COST_PER_1M_TOKENS.keys()}"
            )
        self.model = model
        super().__init__(
            file_path=file_path,
            cost_per_1M_input_tokens=self.COST_PER_1M_TOKENS[model]["input"],
            cost_per_1M_output_tokens=self.COST_PER_1M_TOKENS[model]["output"],
        )

    def get_usage_stats(self, response: ChatCompletion) -> TokenUsageStats:
        total_tokens = response.usage.total_tokens
        return TokenUsageStats(
            total_tokens=total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=response.usage.prompt_tokens
            * (self.cost_per_1M_input_tokens / 1_000_000)
            + response.usage.completion_tokens
            * (self.cost_per_1M_output_tokens / 1_000_000),
            timestamp=datetime.fromtimestamp(response.created),
        )

    def count_tokens(self, text):
        raise NotImplementedError
