import unittest
from unittest import TestCase

from mitools.nlp.embeddings import get_device
from mitools.nlp.sentiments import HuggingFaceAnalyzer, SentimentResult


class TestHuggingFaceAnalyzer(TestCase):
    def setUp(self):
        self.analyzer = HuggingFaceAnalyzer()

    def test_positive_sentiment(self):
        text = "I love this movie! It's fantastic and makes me so happy."
        result = self.analyzer.analyze(text)
        self.assertIsInstance(
            result, SentimentResult, "Should return a SentimentResult object."
        )
        self.assertEqual(
            result.label,
            "POSITIVE",
            "Label should be POSITIVE for clearly positive text.",
        )
        self.assertGreater(
            result.polarity, 0, "Polarity should be positive for positive text."
        )
        self.assertGreater(
            result.confidence,
            0.5,
            "Confidence should be reasonably high for positive text.",
        )

    def test_negative_sentiment(self):
        text = "I absolutely hate this. It's the worst experience ever."
        result = self.analyzer.analyze(text)
        self.assertIsInstance(result, SentimentResult)
        self.assertEqual(
            result.label, "NEGATIVE", "Label should be NEGATIVE for negative text."
        )
        self.assertLess(
            result.polarity, 0, "Polarity should be negative for negative text."
        )
        self.assertGreater(
            result.confidence,
            0.5,
            "Confidence should be reasonably high for negative text.",
        )

    def test_nonsense_input(self):
        text = "Blah bleh bloh, what is even this?"
        result = self.analyzer.analyze(text)
        self.assertIsInstance(result, SentimentResult)
        self.assertIn(
            result.label,
            ["POSITIVE", "NEGATIVE"],
            "Should return one of the two possible labels.",
        )
        self.assertGreaterEqual(result.confidence, 0, "Confidence should be >= 0.")
        self.assertLessEqual(result.confidence, 1, "Confidence should be <= 1.")
        self.assertGreaterEqual(
            result.polarity, -1, "Polarity should be between -1 and 1."
        )
        self.assertLessEqual(result.polarity, 1, "Polarity should be between -1 and 1.")

    def test_analyze_sentences_with_single_string(self):
        text = "I love this movie! But sometimes it can be a bit boring. Overall, it's great."
        results = self.analyzer.analyze_sentences(text)
        self.assertIsInstance(results, list, "Should return a list of results.")
        self.assertTrue(
            all(isinstance(r, SentimentResult) for r in results),
            "All elements should be SentimentResult.",
        )
        self.assertGreaterEqual(
            len(results), 2, "Should have multiple sentences analyzed."
        )
        for r in results:
            self.assertIn(
                r.label,
                ["POSITIVE", "NEGATIVE"],
                "Each sentence should be labeled either POSITIVE or NEGATIVE.",
            )
            self.assertGreaterEqual(r.confidence, 0)
            self.assertLessEqual(r.confidence, 1)

    def test_analyze_sentences_with_list_of_strings(self):
        texts = [
            "This is wonderful.",
            "This is terrible.",
            "Not sure how I feel about this.",
        ]
        results = self.analyzer.analyze_sentences(texts)
        self.assertEqual(
            len(results), 3, "Should return a result for each input string."
        )
        for r in results:
            self.assertIsInstance(r, SentimentResult)
            self.assertIn(r.label, ["POSITIVE", "NEGATIVE"])

    def test_repeated_calls(self):
        pos_text = "I absolutely love this product."
        neg_text = "This is the worst thing I've ever bought."
        pos_result = self.analyzer.analyze(pos_text)
        neg_result = self.analyzer.analyze(neg_text)
        self.assertEqual(
            pos_result.label,
            "POSITIVE",
            "Should handle repeated calls correctly for positive text.",
        )
        self.assertEqual(
            neg_result.label,
            "NEGATIVE",
            "Should handle repeated calls correctly for negative text.",
        )

    def test_device_parameter_cpu(self):
        analyzer_cpu = HuggingFaceAnalyzer(device=-1)
        text = "This is quite nice."
        result = analyzer_cpu.analyze(text)
        self.assertIsInstance(
            result, SentimentResult, "Should return a valid result on CPU."
        )

    def test_device_parameter_gpu(self):
        device = get_device()
        analyzer_cpu = HuggingFaceAnalyzer(device=device)
        text = "This is quite nice."
        result = analyzer_cpu.analyze(text)
        self.assertIsInstance(
            result, SentimentResult, "Should return a valid result on CPU."
        )

    def test_train_called_implicitly(self):
        text = "I like this."
        result = self.analyzer.analyze(text)
        self.assertIsInstance(
            result,
            SentimentResult,
            "Should work fine even if train() not explicitly called.",
        )

    def test_raw_output(self):
        text = "I love this!"
        result = self.analyzer.analyze(text)
        self.assertIsInstance(
            result.raw_output,
            list,
            "raw_output should store the original pipeline output.",
        )
        self.assertTrue(len(result.raw_output) > 0, "raw_output should not be empty.")
        self.assertIn(
            "score", result.raw_output[0], "raw_output item should contain a score."
        )
        self.assertIn(
            "label", result.raw_output[0], "raw_output item should contain a label."
        )


if __name__ == "__main__":
    unittest.main()
