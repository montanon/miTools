from typing import Iterable, Callable
from functools import wraps
#from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessPool as Pool
from .helper_functions import iterable_chunks
from tqdm import tqdm


def parallel(n_threads: int, chunk_size: int):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(iterable: Iterable, *args, **kwargs):
            print('Parallelizing!')
            chunks = list(iterable_chunks(iterable, chunk_size))
            results = []
            # Using pathos's ProcessPool
            with Pool(n_threads) as pool:
                results = pool.map(lambda chunk: func(chunk, *args, **kwargs), chunks)
            # Flattening the results
            return [item for sublist in results for item in sublist]
        return wrapper
    return decorator