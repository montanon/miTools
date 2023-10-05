import re
from typing import Iterable
from icecream import ic

def iterable_chunks(iterable: Iterable, chunk_size: int):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

def str_is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_numbers_from_str(string: str) -> list:
    pattern = r'(-?\d*\.?\d*(?:[eE][-+]?\d+)?)'
    values = [s for s in re.findall(pattern, string.strip()) if s and s != '-']
    return [float(s) if s != '.' else 0 for s in values]

def remove_multiple_spaces(string: str) -> str:
    return re.sub(r'\s+', ' ', string)

def find_str_line_number_in_text(text: str, substring: str):
    lines = text.split('\n')
    for idx, line in enumerate(lines): 
        if substring in line:
            return idx

def read_log_file(log_path):
    with open(log_path, 'r') as f:
        return f.read() 
    
def dict_from_kwargs(**kwargs):
    return {k:v for k, v in kwargs}