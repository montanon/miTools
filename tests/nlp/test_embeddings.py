import os
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch
from pandas import DataFrame
from sklearn.manifold import TSNE
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

from mitools.exceptions import ArgumentValueError
from mitools.nlp import (
    huggingface_embed_texts,
    specter_embed_texts,
    tsne_embeddings,
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
            "Here is the third one.",
            "Another one for the batch test.",
        ]

    def test_single_text_input(self):
        result = huggingface_embed_texts(
            texts=self.texts_single,
            tokenizer=self.tokenizer,
            model=self.model,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_batch_size_single_batch(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=len(self.texts_multiple),  # Single batch
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(self.texts_multiple))

    def test_batch_size_multiple_batches(self):
        batch_size = 2
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=batch_size,  # Multiple batches
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(self.texts_multiple))  # Total items
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_batch_size_equal_to_one(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=1,  # Each text in its own batch
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(self.texts_multiple))
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_batch_size_with_large_input(self):
        large_texts = ["Sentence " + str(i) for i in range(100)]  # Large input
        batch_size = 10
        result = huggingface_embed_texts(
            texts=large_texts,
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=batch_size,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(large_texts))
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_no_batch_size(self):
        result = huggingface_embed_texts(
            texts=self.texts_multiple,
            tokenizer=self.tokenizer,
            model=self.model,
            batch_size=None,  # No batching
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(self.texts_multiple))
        self.assertEqual(result.shape[1], self.model.config.hidden_size)

    def test_invalid_batch_size(self):
        with self.assertRaises(ArgumentValueError):
            huggingface_embed_texts(
                texts=self.texts_multiple,
                tokenizer=self.tokenizer,
                model=self.model,
                batch_size=0,  # Invalid batch size
            )

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


class TestSpecterEmbedTexts(TestCase):
    def setUp(self):
        self.texts_single = "This is a test sentence."
        self.texts_multiple = [
            "This is the first test sentence.",
            "Here is another one.",
            "Testing Specter embedding function.",
        ]

    def test_single_text(self):
        result = specter_embed_texts(
            texts=self.texts_single,
            output_type="numpy",
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)  # Should return 2D array
        self.assertEqual(result.shape[0], 1)  # Single input

    def test_multiple_texts(self):
        result = specter_embed_texts(
            texts=self.texts_multiple,
            output_type="numpy",
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], len(self.texts_multiple))

    def test_output_type_tensor(self):
        result = specter_embed_texts(
            texts=self.texts_multiple,
            output_type="tensor",
        )
        self.assertIsInstance(result, Tensor)

    def test_batch_size(self):
        result = specter_embed_texts(
            texts=self.texts_multiple,
            batch_size=2,
            output_type="numpy",
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], len(self.texts_multiple))

    def test_invalid_pooling(self):
        with self.assertRaises(ArgumentValueError):
            specter_embed_texts(
                texts=self.texts_multiple,
                pooling="invalid",
                output_type="numpy",
            )


class TestTSNEEmbeddings(TestCase):
    def setUp(self):
        self.array_data = np.random.rand(100, 50)  # 100 samples, 50 features
        self.df_data = DataFrame(self.array_data)

    def test_embeddings_with_ndarray(self):
        result = tsne_embeddings(self.array_data, n_components=2, random_state=42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 2))  # 100 samples reduced to 2 dimensions

    def test_embeddings_with_dataframe(self):
        result = tsne_embeddings(self.df_data, n_components=3, random_state=42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 3))  # 100 samples reduced to 3 dimensions

    def test_return_reducer(self):
        reducer = tsne_embeddings(
            self.array_data, n_components=2, random_state=42, return_reducer=True
        )
        self.assertIsInstance(reducer.embedding_, np.ndarray)
        self.assertIsInstance(reducer, TSNE)
        self.assertEqual(
            reducer.embedding_.shape, (100, 2)
        )  # 100 samples reduced to 2 dimensions

    def test_custom_perplexity(self):
        result = tsne_embeddings(self.array_data, perplexity=50.0, random_state=42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 2))  # Default 2 dimensions


class TestUMAPEmbeddings(unittest.TestCase):
    def setUp(self):
        self.array_data = np.random.rand(100, 50)  # 100 samples, 50 features
        self.df_data = DataFrame(self.array_data)

    def test_embeddings_with_ndarray(self):
        result = umap_embeddings(self.array_data, n_components=2, random_state=42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 2))  # 100 samples reduced to 2 dimensions

    def test_embeddings_with_dataframe(self):
        result = umap_embeddings(self.df_data, n_components=3, random_state=42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 3))  # 100 samples reduced to 3 dimensions

    def test_return_reducer(self):
        reducer = umap_embeddings(
            self.array_data, n_components=2, random_state=42, return_reducer=True
        )
        self.assertIsInstance(reducer.embedding_, np.ndarray)
        self.assertIsInstance(reducer, UMAP)
        self.assertEqual(
            reducer.embedding_.shape, (100, 2)
        )  # 100 samples reduced to 2 dimensions

    def test_custom_neighbors(self):
        result = umap_embeddings(self.array_data, n_neighbors=10, random_state=42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 2))  # Default 2 dimensions


if __name__ == "__main__":
    unittest.main()
