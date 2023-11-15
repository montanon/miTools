import itertools
import re
from os import PathLike
from typing import (Any, Dict, Generator, Iterable, List, Optional, Pattern,
                    Type, TypeVar, Union)

import numpy as np
from fuzzywuzzy import fuzz
from numpy import ndarray
from pandas import DataFrame, Series

T = TypeVar('T')
COLOR_CODES = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'reset': '\033[0m'  # Reset to default color
}
# Define a cycle of colors
color_cycler = itertools.cycle(COLOR_CODES.keys() - {'reset'})


def iterable_chunks(iterable: Iterable[T], chunk_size: int) -> Generator[Iterable[T], None, None]:
    if not isinstance(iterable, (str, list, tuple, bytes)):
        raise TypeError(f"Provided iterable of type {type(iterable).__name__} doesn't support slicing.")
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i: i + chunk_size]

def str_is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_numbers_from_str(string: str, n: Optional[int]=None) -> List:
    pattern = r'(-?\d*\.?\d*(?:[eE][-+]?\d+)?)'
    values = [s for s in re.findall(pattern, string.strip()) if s and s != '-']
    numbers = [float(s) if s != '.' else 0 for s in values]
    return numbers[n] if n else numbers

def remove_multiple_spaces(string: str) -> str:
    return re.sub(r'\s+', ' ', string)

def find_str_line_number_in_text(text: str, substring: str) -> int:
    lines = text.split('\n')
    for idx, line in enumerate(lines): 
        if substring in line:
            return idx

def read_text_file(text_path: PathLike) -> str:
    with open(text_path, 'r') as f:
        return f.read() 
    
def dict_from_kwargs(**kwargs: Dict[str, Any]) -> Dict:
    return {k:v for k, v in kwargs}

def lcs_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len_s1, len_s2 = len(s1), len(s2)
    prev_row = [0] * (len_s2 + 1)
    curr_row = [0] * (len_s2 + 1)
    for i in range(len_s1):
        for j in range(len_s2):
            if s1[i] == s2[j]:
                curr_row[j+1] = prev_row[j] + 1
            else:
                curr_row[j+1] = max(curr_row[j], prev_row[j+1])
        prev_row, curr_row = curr_row, prev_row

    lcs_length = prev_row[-1]
    return lcs_length / max(len_s1, len_s2)

def fuzz_string_in_string(src_string: str, dst_string: str, threshold: Optional[int]=90) -> bool:
    similarity_score = fuzz_ratio(src_string, dst_string)
    return similarity_score > threshold

def fuzz_ratio(src_string: str, dst_string: str) -> float:
    similarity_score = fuzz.partial_ratio(src_string, dst_string)
    return similarity_score

def replace_prefix(string: str, prefix: Pattern, replacement: str) -> str:
    return re.sub(r'^' + re.escape(prefix), replacement, string)

def split_strings(str_list: List[str]) -> List[str]:
    new_list = []
    for s in str_list:
        new_list += re.split('(?=[A-Z])', s)
    return new_list

def add_significance(row: Series) -> Series:
    p_value = float(row.split(' ')[1].replace('(','').replace(')',''))  
    if p_value < 0.01:
        return row + "***"
    elif p_value < 0.05:
        return row + "**"
    elif p_value < 0.1:
        return row + "*"
    else:
        return row
    
def remove_dataframe_duplicates(dfs: List[DataFrame]) -> List[DataFrame]:
    unique_dfs = []
    for i in range(len(dfs)):
        if not any(dfs[i].equals(dfs[j]) for j in range(i+1, len(dfs))):
            unique_dfs.append(dfs[i])
    return unique_dfs

def can_convert_to(items: Iterable, type: Type) -> bool:
    try:
        return all(isinstance(type(item), type) for item in items)
    except ValueError:
        return False
    
def invert_dict(dictionary: Dict) -> Dict:
    return {value: key for key, value in dictionary.items()}

def iprint(iterable: Union[Iterable,str], splitter: Optional[str] = '', c: Optional[str] = ''):
    if not hasattr(iprint, 'color_cycler'):
        iprint.color_cycler = itertools.cycle(COLOR_CODES.keys() - {'reset'})
    color_code = COLOR_CODES.get(c, '')  # Get the ANSI escape code for the specified color
    if c == 'cycler':
        color_code = COLOR_CODES[next(iprint.color_cycler)]
    else:
        color_code = COLOR_CODES.get(c, '')  # Get the ANSI escape code for the specified color
    if isinstance(iterable, str):
        iterable = [iterable]
    elif not isinstance(iterable, Iterable):
        iterable = [repr(iterable)]
    for item in iterable:
        if splitter:
            print(splitter * 40)
        if color_code:
            print(f"{color_code}{item}{COLOR_CODES['reset']}")
        else:
            print(item)
        if splitter:
            print(splitter * 40)
            
def check_symmetrical_matrix(a: ndarray, rtol: Optional[float]=1e-05, atol: Optional[float]=1e-08) -> bool:
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def remove_chars(input_string: str, chars_to_remove: str) -> str:
    remove_set = set(chars_to_remove)
    return ''.join(char for char in input_string if char not in remove_set)
