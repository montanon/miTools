import sqlite3
import warnings
from ast import literal_eval
from os import PathLike
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from nltk.tokenize.api import StringTokenizer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from numpy import float64, ndarray
from pandas import DataFrame, Series
from tqdm import tqdm

from ..utils import iterable_chunks

warnings.simplefilter('ignore', NumbaDeprecationWarning)
warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

SPECTER_EMBEDDINGS_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16


def huggingface_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE
                                    ) -> List[ndarray]:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter').to(device)
    embeddings = []
    for chunk in iterable_chunks(texts, batch_size):
        embeddings.extend(huggingface_specter_embed_chunk(chunk, tokenizer, model))
    return embeddings

def huggingface_specter_embed_chunk(chunk: Iterable, tokenizer: StringTokenizer, model: AutoModel) -> List[ndarray]:
    inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    inputs = inputs.to(device)
    result = model(**inputs)
    return result.last_hidden_state[:, 0, :].detach().to('cpu').numpy().tolist()

def create_embeddings_data_table(db_path: PathLike, embeddings_tablename: str, id_col: Optional[str]='id') -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    sql_query = f'''CREATE TABLE IF NOT EXISTS {embeddings_tablename} ({id_col} TEXT PRIMARY KEY,'''
    sql_query += ', '.join([f'col_{i} REAL' for i in range(768)])
    sql_query += ''')'''
    cursor.execute(sql_query)
    conn.commit()
    conn.close()

def read_embedding_indexes(db_path: PathLike, embeddings_tablename: str, id_col: Optional[str]='id') -> List[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f'SELECT {id_col} FROM {embeddings_tablename}')
    indexes = cursor.fetchall()
    conn.close()
    return [index[0] for index in indexes]

def add_embedding_to_table(db_path: PathLike, embedding: Tuple[str,ndarray], embeddings_tablename: str,
                           id_col: Optional[str]='id') -> None:
    embedding_id, embedding_vector = embedding
    assert isinstance(embedding_id, str), "ID must be an string"
    assert all(isinstance(x, (int, float)) for x in embedding_vector), "All embedding values must be numeric"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    col_names = [f'col_{i}' for i in range(768)]
    sql_query = f'''INSERT OR REPLACE INTO {embeddings_tablename} ({id_col}, '''
    sql_query += ', '.join(col_names)
    sql_query += ''') VALUES (?, '''
    sql_query += ', '.join(['?'] * 768)
    sql_query += ''')'''
    cursor.execute(sql_query, [embedding_id] + embedding_vector)
    conn.commit()
    conn.close()

def embeddings_col_to_frame(embeddings: Series) -> DataFrame:
    if all(isinstance(value, str) for value in embeddings):
        embeddings = embeddings.apply(literal_eval)
    embeddings = (embeddings.apply(Series)
                  .astype(float64))
    return embeddings

def umap_embeddings(embeddings: DataFrame) -> ndarray:
    reducer = UMAP()
    return reducer.fit_transform(embeddings)

def semantic_scholar_specter_embed_texts(texts: Union[List[str],str], batch_size: Optional[int]=MAX_BATCH_SIZE
                                         ) -> List[ndarray]:
    embeddings = []
    for chunk in tqdm(iterable_chunks(texts, batch_size), total=len(texts)/batch_size):
        pass
    return embeddings
