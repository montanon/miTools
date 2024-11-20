import os
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from mitools.exceptions import ArgumentValueError
from mitools.nlp import (
    embeddings_col_to_frame,
    huggingface_embed_texts,
    huggingface_specter_embed_chunk,
    umap_embeddings,
)


class TestHuggingfaceEmbedTexts(TestCase):
    def setUp(self):
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.texts_single = "This is a test sentence."
        self.texts_multiple = [
            "This is the first sentence.",
            "This is the second sentence.",
        ]

    def test_single_text_input(self):
        result = huggingface_embed_texts(
            texts=self.texts_single,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_multiple_text_input(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(self.texts_multiple))
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_cls_pooling(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            pooling="cls",
        )
        self.assertIsInstance(result, np.ndarray)

    def test_mean_pooling(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            pooling="mean",
        )
        self.assertIsInstance(result, np.ndarray)

    def test_output_tensor(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            output_type="tensor",
        )
        self.assertIsInstance(result, torch.Tensor)

    def test_output_numpy(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            output_type="numpy",
        )
        self.assertIsInstance(result, np.ndarray)

    def test_output_list(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            output_type="list",
        )
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)

    def test_invalid_pooling(self):
        with self.assertRaises(ArgumentValueError):
            huggingface_embed_texts(
                texts=self.texts_multiple,
                tokenizer=self.tokenizer,
                model=self.model,
                pooling="invalid",
            )

    def test_invalid_output_type(self):
        with self.assertRaises(ArgumentValueError):
            huggingface_embed_texts(
                texts=self.texts_multiple,
                tokenizer=self.tokenizer,
                model=self.model,
                output_type="invalid",
            )

    def test_long_text_truncation(self):
        long_text = " ".join(["word"] * 1000)  # Generate a very long text
        result = huggingface_embed_texts(
            texts=long_text,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_device_gpu(self):
        if torch.cuda.is_available():
            result = huggingface_embed_texts(
                texts=self.texts_multiple,
                tokenizer=self.tokenizer,
                model=self.model,
                device="cuda",
            )
            self.assertIsInstance(result, np.ndarray)

    def test_device_cpu(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            device="cpu",
        )
        self.assertIsInstance(result, np.ndarray)

    def test_custom_tokenizer_length(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            tokenizer_length=128,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_empty_text_list(self):
        with self.assertRaises(ArgumentValueError):
            huggingface_embed_texts(
                texts=[],
                tokenizer=self.tokenizer,
                model=self.model,
            )

    def test_invalid_model(self):
        with self.assertRaises(OSError):
            huggingface_embed_texts(
                texts=self.texts_multiple,
                tokenizer=self.tokenizer,
                model="non-existent-model",
            )

    def test_invalid_tokenizer(self):
        with self.assertRaises(OSError):
            huggingface_embed_texts(
                texts=self.texts_multiple,
                tokenizer="non-existent-tokenizer",
                model=self.model,
            )


class TestHuggingfaceSpecterEmbedTexts(unittest.TestCase):
    def setUp(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def tearDown(self):
        del os.environ["TOKENIZERS_PARALLELISM"]

    def test_single_thread(self):
        texts = ["sample text 1", "sample text 2"]
        embeddings = huggingface_specter_embed_texts(texts, batch_size=1)
        # Assuming embeddings are list of lists/tensors, checking basic structure
        self.assertEqual(len(embeddings), 2)

    def test_multiple_batch(self):
        _batch_size = 4
        texts = [f"sample text {n}" for n in range(1, _batch_size + 1)]
        embeddings = huggingface_specter_embed_texts(texts, batch_size=_batch_size)
        # Assuming embeddings are list of lists/tensors, checking basic structure
        self.assertEqual(len(embeddings), _batch_size)


class TestHuggingfaceSpecterEmbedChunk(unittest.TestCase):
    def setUp(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.model = AutoModel.from_pretrained("allenai/specter").to(device)

    def test_embed_chunk(self):
        _batch_size = 4
        chunk = [f"sample text {n}" for n in range(1, _batch_size + 1)]
        embeddings = huggingface_specter_embed_chunk(chunk, self.tokenizer, self.model)
        self.assertEqual(len(embeddings), _batch_size)


class TestEmbeddingsColToFrame(unittest.TestCase):
    def test_embed_strings(self):
        # Given
        embeddings_series = pd.Series(["[0.1, 0.2]", "[0.3, 0.4]", "[0.5, 0.6]"])
        # When
        embeddings_df = embeddings_col_to_frame(embeddings_series)
        # Then
        expected_df = pd.DataFrame({0: [0.1, 0.3, 0.5], 1: [0.2, 0.4, 0.6]})
        pd.testing.assert_frame_equal(embeddings_df, expected_df)

    def test_embed_lists(self):
        # Given
        embeddings_series = pd.Series([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        # When
        embeddings_df = embeddings_col_to_frame(embeddings_series)
        # Then
        expected_df = pd.DataFrame({0: [0.1, 0.3, 0.5], 1: [0.2, 0.4, 0.6]})
        pd.testing.assert_frame_equal(embeddings_df, expected_df)


class TestUmapEmbeddings(unittest.TestCase):
    def test_umap_dimension_reduction(self):
        # Given: A simple embeddings DataFrame
        embeddings_df = pd.DataFrame(
            {
                "dim1": [v for v in range(0, 100)],
                "dim2": [0.77 * v for v in range(0, 100)],
                "dim3": [0.2 * v for v in range(0, 100)],
            }
        )
        # When: UMAP is applied
        reduced_embeddings = umap_embeddings(embeddings_df)
        # Then: Ensure reduced dimension is as expected (default UMAP is 2D)
        self.assertEqual(reduced_embeddings.shape, (100, 2))


if __name__ == "__main__":
    unittest.main()
