import warnings
from ast import literal_eval
from typing import Callable, Iterable, List, Optional, Union

import torch
from nltk.tokenize.api import StringTokenizer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from numpy import float64, ndarray
from pandas import DataFrame, Series
from tqdm import tqdm

from ..utils import iterable_chunks, parallel

warnings.simplefilter('ignore', NumbaDeprecationWarning)
warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

SPECTER_EMBEDDINGS_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 4


def huggingface_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE, 
                                    n_threads: Optional[int]=1) -> List[ndarray]:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter').to(device)
    
    if n_threads > 1:
        parallel_function = parallel(n_threads, batch_size)(huggingface_specter_embed_texts_parallel)
        return parallel_function(texts, tokenizer, model)
    else:
        embeddings = []
        for chunk in iterable_chunks(texts, batch_size):
            embeddings.extend(huggingface_specter_embed_chunk(chunk, tokenizer, model))
        return embeddings

def huggingface_specter_embed_texts_parallel(chunk: Iterable, tokenizer: StringTokenizer, model: AutoModel) -> Callable:
    return huggingface_specter_embed_chunk(chunk, tokenizer, model)

def huggingface_specter_embed_chunk(chunk: Iterable, tokenizer: StringTokenizer, model: AutoModel) -> List[ndarray]:
    inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    inputs = inputs.to(device)
    result = model(**inputs)
    return result.last_hidden_state[:, 0, :].detach().to('cpu').numpy().tolist()

def semantic_scholar_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE
                                         ) -> List[ndarray]:
    embeddings = []
    for chunk in tqdm(iterable_chunks(texts, batch_size), total=len(texts)/batch_size):
        pass
    return embeddings

def embeddings_col_to_frame(embeddings: Series) -> DataFrame:
    if all(isinstance(value, str) for value in embeddings):
        embeddings = embeddings.apply(literal_eval)
    embeddings = (embeddings.apply(Series)
                  .astype(float64))
    return embeddings

def umap_embeddings(embeddings: DataFrame) -> ndarray:
    reducer = UMAP()
    return reducer.fit_transform(embeddings)
