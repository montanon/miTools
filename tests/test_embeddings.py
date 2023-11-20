import os
import unittest
from unittest.mock import Mock, patch

from mitools.nlp import *


class TestHuggingfaceSpecterEmbedTexts(unittest.TestCase):

    def setUp(self):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    def tearDown(self):
        del os.environ['TOKENIZERS_PARALLELISM']

    def test_single_thread(self):
        texts = ["sample text 1", "sample text 2"]
        embeddings = huggingface_specter_embed_texts(texts, batch_size=1)
        # Assuming embeddings are list of lists/tensors, checking basic structure
        self.assertEqual(len(embeddings), 2)

    def test_multiple_batch(self):
        _batch_size = 4
        texts = [f"sample text {n}" for n in range(1,_batch_size+1)] 
        embeddings = huggingface_specter_embed_texts(texts, batch_size=_batch_size)
        # Assuming embeddings are list of lists/tensors, checking basic structure
        self.assertEqual(len(embeddings), _batch_size)


class TestHuggingfaceSpecterEmbedChunk(unittest.TestCase):

    def setUp(self):
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter').to(device)

    def test_embed_chunk(self):
        _batch_size = 4
        chunk = [f"sample text {n}" for n in range(1,_batch_size+1)]
        embeddings = huggingface_specter_embed_chunk(chunk, self.tokenizer, self.model)
        self.assertEqual(len(embeddings), _batch_size)


class TestEmbeddingsColToFrame(unittest.TestCase):

    def test_embed_strings(self):
        # Given
        embeddings_series = pd.Series(["[0.1, 0.2]", "[0.3, 0.4]", "[0.5, 0.6]"])
        # When
        embeddings_df = embeddings_col_to_frame(embeddings_series)
        # Then
        expected_df = pd.DataFrame({
            0: [0.1, 0.3, 0.5],
            1: [0.2, 0.4, 0.6]
        })
        pd.testing.assert_frame_equal(embeddings_df, expected_df)

    def test_embed_lists(self):
        # Given
        embeddings_series = pd.Series([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        # When
        embeddings_df = embeddings_col_to_frame(embeddings_series)
        # Then
        expected_df = pd.DataFrame({
            0: [0.1, 0.3, 0.5],
            1: [0.2, 0.4, 0.6]
        })
        pd.testing.assert_frame_equal(embeddings_df, expected_df)


class TestUmapEmbeddings(unittest.TestCase):

    def test_umap_dimension_reduction(self):
        # Given: A simple embeddings DataFrame
        embeddings_df = pd.DataFrame({
            'dim1': [v for v in range(0,100)],
            'dim2': [0.77*v for v in range(0,100)],
            'dim3': [0.2*v for v in range(0,100)]
        })
        # When: UMAP is applied
        reduced_embeddings = umap_embeddings(embeddings_df)
        # Then: Ensure reduced dimension is as expected (default UMAP is 2D)
        self.assertEqual(reduced_embeddings.shape, (100, 2))


if __name__ == '__main__':
    unittest.main()
