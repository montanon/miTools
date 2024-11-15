import unittest
from unittest import TestCase

from mitools.exceptions import ArgumentKeyError, ArgumentTypeError, ArgumentValueError
from mitools.llms.objects import Prompt


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


if __name__ == "__main__":
    unittest.main()
