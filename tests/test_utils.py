import pandas as pd
import numpy as np

from typing import Optional, List

def generate_mock_dataframe(n_rows: Optional[int]=10, n_cols: Optional[int]=5, 
                              col_names: Optional[List[str]]=None, data_type: Optional[str]='int',
                              low: Optional[int]=0, high: Optional[int]=100):
    if col_names is None or len(col_names) != n_cols:
        col_names = [f"Col_{i}" for i in range(n_cols)]
    if data_type == 'int':
        data = np.random.randint(low, high, size=(n_rows, n_cols))
    elif data_type == 'float':
        data = np.random.uniform(low, high, size=(n_rows, n_cols))
    else:
        raise ValueError("Invalid data_type. Choose either 'int' or 'float'.")
    df = pd.DataFrame(data, columns=col_names)
    return df
