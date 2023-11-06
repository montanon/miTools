import re
from typing import AnyStr, Dict, Iterable, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from nltk import FreqDist, sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.api import StemmerI
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.api import StringTokenizer
from nltk.util import ngrams
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from tqdm import tqdm
from unidecode import unidecode

from ..utils import lcs_similarity


def tag_tokens(tokens: List[str]) -> List[Tuple[str,str]]:
    nltk_tags = nltk.pos_tag([token.lower() for token in tokens])
    wordnet_tags = nltk_tags_to_wordnet_tags(nltk_tags)
    return wordnet_tags

def tag_token(token: str) -> List[Tuple[str,str]]:
    return tag_tokens([token])

def nltk_tags_to_wordnet_tags(nltk_tags: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    return list(map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tags))

def nltk_tag_to_wordnet_tag(nltk_tag: str) -> str:
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return wordnet.NOUN
    
def lemmatize_text(text: str, lemmatizer: Optional[Type[StemmerI]]=None) -> str:
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = lemmatize_tokens(tokens, lemmatizer)
    return ' '.join(lemmatized_tokens)

def lemmatize_tokens(tokens: Iterable[str], lemmatizer: Optional[Type[StemmerI]]=None) -> List[str]:
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tags = tag_tokens(tokens)
    return [lemmatizer.lemmatize(token, tag) for token, tag in tags]

def lemmatize_token(token: str, lemmatizer: Optional[Type[StemmerI]]=None) -> str:
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tag = tag_token(token)
    return [lemmatizer.lemmatize(token, tag) for token, tag in tag][0]

def preprocess_texts(texts: List[str], stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    tokenizer: Optional[Type[StringTokenizer]]=None) -> List[str]:
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    return [preprocess_text(text, stopwords, lemmatize, tokenizer) for text in texts]

def preprocess_text(text: str, stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None
                    ) -> List[str]:
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    tokens = tokenizer.tokenize(text)
    return preprocess_tokens(tokens, stopwords, lemmatize, lemmatizer)

def preprocess_tokens(tokens: List[str], stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    lemmatizer: Optional[Type[StemmerI]]=None) -> List[str]:
    if lemmatize:
        tokens = lemmatize_tokens(tokens, lemmatizer)
    if stopwords:
        tokens = [token for token in tokens if token.lower() not in stopwords]
    return tokens

def preprocess_token(token: str, stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    lemmatizer: Optional[Type[StemmerI]]=None) -> str:
    if lemmatize:
        token = lemmatize_token(token, lemmatizer)
    if stopwords and token.lower() in stopwords:
        return ''
    return token

def get_tfidf(words_count: DataFrame) -> DataFrame:
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf = transformer.fit_transform(words_count.values)
    df_tfidf = DataFrame(tfidf.toarray(), columns=words_count.columns, index=words_count.index)
    return df_tfidf
    
def get_bow_of_tokens(tokens: List[str], preprocess: Optional[bool]=False, 
                      stopwords: Optional[List[str]]=None) -> Dict[str,int]:
    tokens = tokens if not preprocess else preprocess_tokens(tokens, stopwords)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokens)
    feature_names = vectorizer.get_feature_names_out()
    bow = dict(zip(feature_names, X.sum(axis=0).A1))
    bow = dict(sorted(bow.items(), key=lambda item: item[1], reverse=True))
    return bow

def get_dataframe_bow(dataframe: DataFrame, text_col: str, preprocess: Optional[bool]=False,
                     stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                     tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None
                    ) -> DataFrame:
    return dataframe[[text_col]].apply(get_bow_of_text, axis=1,
                                       args=(preprocess, stopwords, lemmatize, tokenizer, lemmatizer)
                                       ).apply(Series).fillna(0)

def get_bow_of_text(text: Union[str,Series], preprocess: Optional[bool]=False, stopwords: Optional[List[str]]=None, 
                    lemmatize: Optional[bool]=False, tokenizer: Optional[Type[StringTokenizer]]=None, 
                    lemmatizer: Optional[Type[StemmerI]]=None) -> Dict[str,int]:
    text = list(text) if isinstance(text, str) else text
    text = text if not preprocess else preprocess_text(text[0], stopwords, lemmatize, tokenizer, lemmatizer)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    feature_names = vectorizer.get_feature_names_out()
    bow = dict(zip(feature_names, X.sum(axis=0).A1))
    bow = dict(sorted(bow.items(), key=lambda item: item[1], reverse=True))
    return bow

def preprocess_country_name(country_name: str) -> str:
    country_name = unidecode(country_name) 
    country_name = country_name.lower() 
    country_name = re.sub(r'[^a-z\s]', ' ', country_name).strip()
    return country_name

def find_countries_in_dataframe(dataframe: DataFrame, countries: List[str], demonyms: Dict[str,str]) -> DataFrame:
    tqdm.pandas()
    return dataframe.progress_applymap(
        lambda x: find_country_in_token(x, countries, demonyms) if pd.notna(x) else np.nan
        )

def find_country_in_token(token: str, countries: List[str], demonyms: Dict[str, str], 
                            similarity_threshold: Optional[int]=0.9) -> Tuple[str, str]:
    special_cases = {
        'uk': 'united kingdom'
    }
    original_token = token
    token = demonyms.get(token, token)
    token = special_cases.get(token, token)
    highest_similarity = 0
    mentioned_country = (None, None)
    for country in countries:
        dist = lcs_similarity(token, country)
        if dist >= highest_similarity:
            highest_similarity = dist
            mentioned_country = (country, original_token)
            if dist == 1.0:
                break
    if highest_similarity < similarity_threshold:
        mentioned_country = (None, None)
    return mentioned_country

def sort_multiindex_dataframe(df: DataFrame, selected_cols: List[str], sorting_col: Optional[str]=None, 
                              top_level: int=0, bot_level=-1, ascending: Optional[bool]=False
                              ) -> DataFrame:
    top_level_values = df.columns.get_level_values(top_level).unique()
    sorted_dfs = []
    for top_value in top_level_values:
        selected_cluster_df = df.loc[:, [(top_value, c) for c in selected_cols]]
        if sorting_col:
            sort_key = tuple((top_value, sorting_col))
            selected_cluster_df = selected_cluster_df.sort_values(by=sort_key, ascending=ascending).reset_index(drop=True)
        sorted_dfs.append(selected_cluster_df)
    return pd.concat(sorted_dfs, axis=1)

def plot_token_features(df: DataFrame, columns: List[str], 
                        hue: Optional[str]=None, log: Optional[bool]=True, 
                        ncols: Optional[int]=2,
                        figsize: Optional[Tuple]=(4,4)) -> Axes:
    nrows = (len(columns) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*nrows, figsize[1]*ncols))

    for n, var in enumerate(columns):
        ax = axes[n//ncols, n%ncols]
        sns.histplot(data=df, x=var, hue=None, bins=30, kde=False, ax=ax)
        if log:
            ax.set_yscale("log")
        ax.set_title(f"Histogram for {var} in Papers' Text")
    plt.tight_layout()
    plt.show()

    return axes
