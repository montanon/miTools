import re
from typing import Iterable, Pattern
from icecream import ic
from os import PathLike
from typing import Dict, Any, List, Optional
from fuzzywuzzy import fuzz
from pandas import DataFrame

def iterable_chunks(iterable: Iterable, chunk_size: int):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

def str_is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_numbers_from_str(string: str, n: Optional[int]=None) -> list:
    pattern = r'(-?\d*\.?\d*(?:[eE][-+]?\d+)?)'
    values = [s for s in re.findall(pattern, string.strip()) if s and s != '-']
    numbers = [float(s) if s != '.' else 0 for s in values]
    return numbers[n] if n else numbers

def remove_multiple_spaces(string: str) -> str:
    return re.sub(r'\s+', ' ', string)

def find_str_line_number_in_text(text: str, substring: str):
    lines = text.split('\n')
    for idx, line in enumerate(lines): 
        if substring in line:
            return idx

def read_log_file(log_path: PathLike):
    with open(log_path, 'r') as f:
        return f.read() 
    
def dict_from_kwargs(**kwargs: Dict[str, Any]):
    return {k:v for k, v in kwargs}

def lcs_similarity(s1: str, s2: str):
    matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                matrix[i+1][j+1] = matrix[i][j] + 1
            else:
                matrix[i+1][j+1] = max(matrix[i+1][j], matrix[i][j+1])
    lcs_length = matrix[-1][-1]
    return lcs_length / max(len(s1), len(s2))

def fuzz_string_in_string(src_string: str , dst_string: str, threshold: Optional[int]=90):
    similarity_score = fuzz.partial_ratio(src_string, dst_string)
    return similarity_score > threshold

def replace_prefix(string: str, prefix: Pattern, replacement: str):
    return re.sub(r'^' + re.escape(prefix), replacement, string)

def split_strings(str_list: List[str]):
    new_list = []
    for s in str_list:
        new_list += re.split('(?=[A-Z])', s)
    return new_list

def add_significance(row):
    p_value = float(row.split(' ')[1].replace('(','').replace(')',''))  
    if p_value < 0.01:
        return row + "***"
    elif p_value < 0.05:
        return row + "**"
    elif p_value < 0.1:
        return row + "*"
    else:
        return row
    
def remove_dataframe_duplicates(dfs: List[DataFrame]):
    unique_dfs = []
    for i in range(len(dfs)):
        if not any(dfs[i].equals(dfs[j]) for j in range(i+1, len(dfs))):
            unique_dfs.append(dfs[i])
    return unique_dfs
