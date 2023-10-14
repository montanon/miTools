from tqdm import tqdm
from ..utils import iterable_chunks
from typing import Union, List, Optional
from .semantic_scholar_api_key import SEMANTIC_SCHOLAR_API_KEY
from pandas import Series, DataFrame
from numpy import float64
from ast import literal_eval

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', NumbaDeprecationWarning)
warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)
from umap import UMAP
from transformers import AutoTokenizer, AutoModel

SPECTER_EMBEDDINGS_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16


def huggingface_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    embeddings = []
    for chunk in tqdm(iterable_chunks(texts, batch_size), total=len(texts)/batch_size):
        inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512)
        result = model(**inputs)
        embeddings.extend(
            result.last_hidden_state[:, 0, :]
            .detach()
            .numpy()
            )
    return embeddings


def semantic_scholar_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE):
    embeddings = []
    for chunk in tqdm(iterable_chunks(texts, batch_size), total=len(texts)/batch_size):
        pass
    return embeddings


def embeddings_col_to_frame(embeddings: Series):
    if all(isinstance(value, str) for value in embeddings):
        embeddings = embeddings.apply(literal_eval)
    embeddings = (embeddings.apply(Series)
                  .astype(float64))
    return embeddings


def umap_embeddings(embeddings: DataFrame):
    reducer = UMAP()
    return reducer.fit_transform(embeddings)
