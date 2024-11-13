import inspect
import warnings
from functools import wraps
from multiprocessing import Pool
from typing import Any, Callable, Iterable

from pandas import DataFrame
from tqdm import tqdm

from mitools.exceptions import ArgumentTypeError, ArgumentValueError

from .helper_functions import iterable_chunks


def parallel(n_threads: int, chunk_size: int):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(iterable: Iterable, *args, **kwargs):
            print("Parallelizing!")
            chunks = list(iterable_chunks(iterable, chunk_size))
            results = []
            with Pool(processes=n_threads) as pool:
                async_results = [
                    pool.apply_async(func, (chunk, *args)) for chunk in chunks
                ]
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


def validate_args_types(**type_hints):
    def decorator(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Applies default values to missing arguments

            for name, expected_type in type_hints.items():
                if name in bound_args.arguments:
                    value = bound_args.arguments[name]
                else:
                    raise ArgumentValueError(
                        f"Argument '{name}' not found in function signature"
                    )

                if not isinstance(value, expected_type):
                    raise ArgumentTypeError(
                        f"Argument '{name}' must be of type {expected_type.__name__}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_dataframe_structure(
    dataframe_name: str,
    validation: Callable[[DataFrame, Any], None],
    *validation_args,
    **validation_kwargs,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            dataframe = kwargs.get(dataframe_name, None)
            if dataframe is None:
                raise ArgumentValueError(
                    f"Dataframe argument '{dataframe}' is missing."
                )
            if not isinstance(dataframe, DataFrame):
                raise ArgumentTypeError(f"Argument '{dataframe}' must be a DataFrame.")
            validation(
                dataframe,
                *validation_args,
                **validation_kwargs,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
