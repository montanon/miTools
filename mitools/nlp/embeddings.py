from tqdm import tqdm
from ..utils import iterable_chunks, parallel
from typing import Union, List, Optional
from .semantic_scholar_api_key import SEMANTIC_SCHOLAR_API_KEY
from pandas import Series, DataFrame
from numpy import float64
from ast import literal_eval
from multiprocessing import cpu_count

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', NumbaDeprecationWarning)
warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)
from umap import UMAP
from transformers import AutoTokenizer, AutoModel

SPECTER_EMBEDDINGS_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16


def huggingface_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE, n_threads: Optional[int]=1):
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    
    if n_threads > 1:
        parallel_function = parallel(n_threads, batch_size)(huggingface_specter_embed_chunk)
        return parallel_function(texts, tokenizer, model)
    else:
        embeddings = []
        for chunk in iterable_chunks(texts, batch_size):
            embeddings.extend(huggingface_specter_embed_chunk(chunk, tokenizer, model))
        return embeddings

def huggingface_specter_embed_texts_parallel(chunk, tokenizer, model):
    return huggingface_specter_embed_chunk(chunk, tokenizer, model)

def huggingface_specter_embed_chunk(chunk, tokenizer, model):
    inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    return result.last_hidden_state[:, 0, :].detach().numpy().tolist()

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
