import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict
from unittest import TestCase

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError
from mitools.llms.objects import (
    PersistentTokensCounter,
    Prompt,
    TokensCounter,
    TokenUsageStats,
)


class TestPrompt(TestCase):
    def test_initialization(self):
        prompt = Prompt("Translate to French.", {"task": "translation"})
        self.assertEqual(prompt.text, "Translate to French.")
        self.assertEqual(prompt.metadata, {"task": "translation"})
        prompt = Prompt("Summarize this text.")
        self.assertEqual(prompt.metadata, {})
        with self.assertRaises(ArgumentValueError):
            Prompt("")
        with self.assertRaises(ArgumentValueError):
            Prompt(123)  # Non-string text

    def test_representation(self):
        prompt = Prompt("Translate to French.")
        print(prompt)
        self.assertTrue(repr(prompt).startswith("Prompt(\ntext"))

    def test_format(self):
        prompt = Prompt("Translate to French: {text}")
        formatted_prompt = prompt.format(text="Hello")
        self.assertEqual(formatted_prompt.text, "Translate to French: Hello")
        with self.assertRaises(ArgumentKeyError):
            prompt.format(language="English")

    def test_update_metadata(self):
        prompt = Prompt("Translate to French.")
        prompt.update_metadata("task", "translation")
        self.assertEqual(prompt.metadata, {"task": "translation"})
        with self.assertRaises(ArgumentValueError):
            prompt.update_metadata(123, "translation")
        with self.assertRaises(ArgumentValueError):
            prompt.update_metadata("task", 456)

    def test_get_metadata(self):
        prompt = Prompt("Translate to French.", {"task": "translation"})
        self.assertEqual(prompt.get_metadata("task"), "translation")
        self.assertIsNone(prompt.get_metadata("language"))

    def test_to_dict(self):
        prompt = Prompt("Translate to French.", {"task": "translation"})
        expected = {"text": "Translate to French.", "metadata": {"task": "translation"}}
        self.assertEqual(prompt.to_dict(), expected)

    def test_from_dict(self):
        data = {"text": "Translate to French.", "metadata": {"task": "translation"}}
        prompt = Prompt.from_dict(data)
        self.assertEqual(prompt.text, "Translate to French.")
        self.assertEqual(prompt.metadata, {"task": "translation"})
        with self.assertRaises(ArgumentValueError):
            Prompt.from_dict({"metadata": {"task": "translation"}})

    def test_concatenation_with_prompt(self):
        prompt1 = Prompt("Translate to French.", {"task": "translation"})
        prompt2 = Prompt("Summarize this text.", {"task": "summarization"})
        combined = prompt1 + prompt2
        self.assertEqual(combined.text, "Translate to French.\nSummarize this text.")
        self.assertEqual(
            combined.metadata, {"task": "translation"}
        )  # Metadata from frist prompt

    def test_concatenation_with_string(self):
        prompt = Prompt("Translate to French.")
        combined = prompt + "Provide a summary."
        self.assertEqual(combined.text, "Translate to French.\nProvide a summary.")
        self.assertEqual(combined.metadata, {})
        prompt += "Provide a detailed summary."
        self.assertEqual(
            prompt.text, "Translate to French.\nProvide a detailed summary."
        )

    def test_invalid_concatenation(self):
        prompt = Prompt("Translate to French.")
        with self.assertRaises(ArgumentTypeError):
            prompt + 123  # Invalid type

    def test_static_concatenate(self):
        prompt1 = Prompt("Translate to French.", {"task": "translation"})
        prompt2 = Prompt("Summarize this text.", {"task": "summarization"})
        text = "Explain this in detail."
        combined = Prompt.concatenate([prompt1, prompt2, text], separator="\n---\n")
        self.assertEqual(
            combined.text,
            "Translate to French.\n---\nSummarize this text.\n---\nExplain this in detail.",
        )
        self.assertEqual(combined.metadata, {"task": "summarization"})
        with self.assertRaises(ArgumentValueError):
            Prompt.concatenate([])
        with self.assertRaises(ArgumentTypeError):
            Prompt.concatenate([prompt1, 123])

    def test_edge_cases(self):
        long_text = "A" * 10_000
        prompt = Prompt(long_text)
        self.assertEqual(prompt.text, long_text)
        prompt1 = Prompt("Task 1", {"task": "t1"})
        prompt2 = Prompt("Task 2", {"task": "t2"})
        combined = prompt1 + prompt2
        self.assertEqual(combined.metadata, {"task": "t1"})
        prompts = [Prompt("One"), Prompt("Two"), "Three"]
        combined = Prompt.concatenate(prompts, separator=" | ")
        self.assertEqual(combined.text, "One | Two | Three")


class TestTokensCounter(TokensCounter):
    def get_usage_stats(self, response: Dict) -> TokenUsageStats:
        total_tokens = len(response.get("text", "").split())
        prompt_tokens = total_tokens // 2
        completion_tokens = total_tokens - prompt_tokens
        cost = self._calculate_cost(total_tokens)
        return TokenUsageStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            timestamp=datetime.now(),
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class TokensCounterTests(TestCase):
    def setUp(self):
        self.counter = TestTokensCounter(cost_per_1k_tokens=0.02)
        self.usage_sample = TokenUsageStats(
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            cost=0.002,
            timestamp=datetime.now(),
        )

    def test_initialization(self):
        self.assertEqual(self.counter.cost_per_1k_tokens, 0.02)
        self.assertEqual(self.counter.prompt_tokens_count, 0)
        self.assertEqual(self.counter.completion_tokens_count, 0)
        self.assertEqual(self.counter.total_tokens_count, 0)
        self.assertEqual(self.counter.usage_history, [])

    def test_update_usage(self):
        self.counter.update(self.usage_sample)
        self.assertEqual(len(self.counter.usage_history), 1)
        self.assertEqual(self.counter.prompt_tokens_count, 60)
        self.assertEqual(self.counter.completion_tokens_count, 40)
        self.assertEqual(self.counter.total_tokens_count, 100)

    def test_count_tokens(self):
        text = "This is a sample text with nine tokens total."
        self.assertEqual(self.counter.count_tokens(text), 9)

    def test_would_exceed_context(self):
        self.counter.set_max_context_length(50)
        self.assertTrue(self.counter.would_exceed_context("word " * 51))
        self.assertFalse(self.counter.would_exceed_context("word " * 49))

    def test_cost_calculation(self):
        self.assertEqual(self.counter._calculate_cost(1000), 0.02)
        self.assertEqual(self.counter._calculate_cost(500), 0.01)

    def test_cost_detail(self):
        self.counter.update(self.usage_sample)
        cost_detail = self.counter.cost_detail
        self.assertAlmostEqual(cost_detail["cost"]["total_tokens"], 0.002)
        self.assertAlmostEqual(cost_detail["cost"]["prompt_tokens"], 0.0012)
        self.assertAlmostEqual(cost_detail["cost"]["completion_tokens"], 0.0008)

    def test_json_serialization(self):
        self.counter.update(self.usage_sample)
        json_data = self.counter.json()
        data = json.loads(json_data)
        self.assertEqual(data["prompt_tokens_count"], 60)
        self.assertEqual(data["completion_tokens_count"], 40)
        self.assertEqual(data["total_tokens_count"], 100)
        self.assertEqual(data["cost_per_1k_tokens"], 0.02)

    def test_save_to_json(self):
        file_path = Path("test_tokens_counter.json")
        self.counter.update(self.usage_sample)
        self.counter.save(file_path)

        with open(file_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["prompt_tokens_count"], 60)
        self.assertEqual(data["completion_tokens_count"], 40)
        self.assertEqual(data["total_tokens_count"], 100)
        self.assertEqual(data["cost_per_1k_tokens"], 0.02)
        file_path.unlink()  # Cleanup

    def test_usage_dataframe(self):
        self.counter.update(self.usage_sample)
        self.counter.update(self.usage_sample)
        self.counter.update(self.usage_sample)
        self.counter.update(self.usage_sample)
        self.counter.update(self.usage_sample)
        df = self.counter.usage()
        self.assertEqual(df.shape, (5, 5))
        self.assertEqual(df.loc[0, "total_tokens"], 100)
        self.assertEqual(df.loc[0, "prompt_tokens"], 60)
        self.assertEqual(df.loc[0, "completion_tokens"], 40)
        self.assertAlmostEqual(df.loc[0, "cost"], 0.002)

    def test_load_from_json(self):
        file_path = Path("test_tokens_counter.json")
        self.counter.update(self.usage_sample)
        self.counter.save(file_path)

        loaded_counter = TestTokensCounter.load(file_path)
        self.assertEqual(loaded_counter.prompt_tokens_count, 60)
        self.assertEqual(loaded_counter.completion_tokens_count, 40)
        self.assertEqual(loaded_counter.total_tokens_count, 100)
        self.assertEqual(len(loaded_counter.usage_history), 1)
        self.assertAlmostEqual(loaded_counter.cost, 0.002)
        file_path.unlink()  # Cleanup

    def test_invalid_file_extension(self):
        with self.assertRaises(ArgumentValueError):
            self.counter.save("invalid_file.txt")

    def test_load_from_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            TestTokensCounter.load("nonexistent.json")


class TestPersistentTokensCounter(PersistentTokensCounter):
    def get_usage_stats(self, response: Dict) -> TokenUsageStats:
        total_tokens = len(response.get("text", "").split())
        prompt_tokens = total_tokens // 2
        completion_tokens = total_tokens - prompt_tokens
        cost = self._calculate_cost(total_tokens)
        return TokenUsageStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            timestamp=datetime.now(),
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class PersistentTokensCounterTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.test_dir.name) / "token_counter.json"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_singleton_behavior(self):
        counter1 = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        counter2 = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.05)
        self.assertIs(counter1, counter2)  # Both should refer to the same instance
        self.assertEqual(
            counter1.cost_per_1k_tokens, 0.02
        )  # The first initialization value is retained

    def test_file_based_initialization(self):
        counter1 = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        usage_stats = TokenUsageStats(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            cost=0.02,
            timestamp=datetime.now(),
        )
        counter1.update(usage_stats)
        counter2 = TestPersistentTokensCounter(self.file_path)
        self.assertEqual(len(counter2.usage_history), 1)
        self.assertEqual(counter2.count, 1000)

    def test_update_and_auto_save(self):
        counter = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        usage_stats = TokenUsageStats(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            cost=0.02,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats)
        new_counter = TestPersistentTokensCounter(self.file_path)
        self.assertEqual(new_counter.count, 1000)
        self.assertEqual(new_counter.prompt_tokens_count, 700)
        self.assertEqual(new_counter.completion_tokens_count, 300)

    def test_usage_dataframe(self):
        counter = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        usage_stats1 = TokenUsageStats(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            cost=0.02,
            timestamp=datetime.now(),
        )
        usage_stats2 = TokenUsageStats(
            total_tokens=500,
            prompt_tokens=300,
            completion_tokens=200,
            cost=0.01,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats1)
        counter.update(usage_stats2)
        df = counter.usage()
        self.assertEqual(len(df), 2)  # Two rows in the DataFrame
        self.assertEqual(df["total_tokens"].sum(), 1500)
        self.assertEqual(df["cost"].sum(), 0.03)

    def test_multiple_instances_with_different_paths(self):
        file_path2 = Path(self.test_dir.name) / "token_counter_2.json"
        counter1 = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        counter2 = TestPersistentTokensCounter(file_path2, cost_per_1k_tokens=0.05)
        self.assertIsNot(counter1, counter2)  # Different instances
        self.assertNotEqual(counter1.cost_per_1k_tokens, counter2.cost_per_1k_tokens)

    def test_empty_file_initialization(self):
        self.assertFalse(self.file_path.exists())

        counter = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        self.assertTrue(self.file_path.exists())
        self.assertEqual(len(counter.usage_history), 0)

    def test_data_integrity_after_reload(self):
        counter = TestPersistentTokensCounter(self.file_path, cost_per_1k_tokens=0.02)
        usage_stats = TokenUsageStats(
            total_tokens=2000,
            prompt_tokens=1500,
            completion_tokens=500,
            cost=0.04,
            timestamp=datetime.now(),
        )
        counter.update(usage_stats)
        new_counter = TestPersistentTokensCounter(self.file_path)
        self.assertEqual(new_counter.total_tokens_count, 2000)
        self.assertEqual(new_counter.cost, 0.04)

    def test_invalid_file_path(self):
        with self.assertRaises(FileNotFoundError):
            TestPersistentTokensCounter(
                "/invalid/path/token_counter.json", cost_per_1k_tokens=0.02
            )


if __name__ == "__main__":
    unittest.main()
