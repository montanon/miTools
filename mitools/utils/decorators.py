import warnings
from functools import wraps
from multiprocessing import Pool, cpu_count
from typing import Callable, Iterable

from tqdm import tqdm

from .helper_functions import iterable_chunks


def parallel(n_threads: int, chunk_size: int):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(iterable: Iterable, *args, **kwargs):
            print('Parallelizing!')
            chunks = list(iterable_chunks(iterable, chunk_size))
            results = []
            with Pool(processes=n_threads) as pool:
                async_results = [pool.apply_async(func, (chunk, *args)) for chunk in chunks]
                for async_result in tqdm(async_results, total=len(chunks)):
                    results.append(async_result.get())
            return [item for sublist in results for item in sublist]
        return wrapper
    return decorator

def suppress_user_warning(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return func(*args, **kwargs)
    return wrapper