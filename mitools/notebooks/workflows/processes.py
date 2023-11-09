import os
import re
from abc import ABC
from typing import List, Optional, Union

import pandas as pd
from pandas import DataFrame

from ..context import DISPLAY
from .configurations.urban_climate_papers import articles_cols, main_words, sub_words


class Process(ABC):
    def __init__(self):
        pass





ARTICLES_COLS = articles_cols
ARTICLES_CSV = 'all_articles.csv'
CSVS_FOLDER = './final_articles'
TEXTS_COLUMNS = ['Title', 'Abstract']
FULL_TEXT_COLUMN = 'text'

class ETL(Process):

    def __init__(self, *inputs):
        pass

    def etl(self, ):

        create_articles_csv = not os.path.exists(ARTICLES_CSV)

        if create_articles_csv:
            csvs = [os.path.join(CSVS_FOLDER, path) for path in os.listdir(CSVS_FOLDER)]
            csvs = [pd.read_csv(csv, index_col=0) for csv in csvs]
            df = pd.concat(csvs, axis=0).drop_duplicates().reset_index(drop=True)

            inv_articles_cols = {v: k for k, v in ARTICLES_COLS.items() if v is not None}
            df = df.rename(columns=inv_articles_cols)

            df[FULL_TEXT_COLUMN] = df[TEXTS_COLUMNS].apply(
                lambda x: ' '.join(
                    [str(v) for v in x.values]
                    ), axis=1)

            sub_words_pattern = '|'.join([f"(?=.*{w.lower()})" for w in sub_words])
            
            pattern = f"(((?=.*urban)|(?=.*city))({sub_words_pattern}))"
            df['relevant'] = df['text'].apply(lambda x: bool(re.search(pattern, x, re.IGNORECASE)))

            if DISPLAY:
                print('Amount of papers before removing irrelevants: ', df.shape[0])
                df = df.loc[df['relevant'] == True].reset_index()
                print('Amount of papers after removing irrelevants: ', df.shape[0])
                df.to_csv(ARTICLES_CSV)
        else:
            df = pd.read_parquet(ARTICLES_CSV)

        return df

        



