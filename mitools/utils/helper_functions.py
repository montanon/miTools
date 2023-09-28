from typing import Iterable

def iterable_chunks(iterable: Iterable, chunk_size: int):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

def str_is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False
