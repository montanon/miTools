import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.api import StringTokenizer
from nltk.stem.api import StemmerI

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

import re
import pandas as pd
from typing import List, Optional, Tuple, Dict, Type, Iterable, Union, Literal
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode

from ..utils import lcs_similarity


def tag_tokens(tokens: List[str]):
    nltk_tags = nltk.pos_tag([token.lower() for token in tokens])
    wordnet_tags = nltk_tags_to_wordnet_tags(nltk_tags)
    return wordnet_tags

def tag_token(token: str):
    return tag_tokens([token])

def nltk_tags_to_wordnet_tags(nltk_tags):
    return map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tags)

def nltk_tag_to_wordnet_tag(nltk_tag: str):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmatize_text(text: str, lemmatizer: Optional[Type[StemmerI]]=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = lemmatize_tokens(tokens, lemmatizer)
    return ' '.join(lemmatized_tokens)

def lemmatize_tokens(tokens: Iterable[str], lemmatizer: Optional[Type[StemmerI]]=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tags = tag_tokens(tokens)
    return [lemmatizer.lemmatize(token, tag) if tag is not None else token for token, tag in tags]

def lemmatize_token(token: str, lemmatizer: Optional[Type[StemmerI]]=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tag = tag_token(token)
    return [lemmatizer.lemmatize(token, tag) if tag is not None else token for token, tag in tag][0]

def preprocess_texts(texts: List[str], stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    tokenizer: Optional[Type[StringTokenizer]]=None):
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    return [preprocess_text(text, stopwords, lemmatize, tokenizer) for text in texts]

def preprocess_text(text: str, stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None):
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    tokens = tokenizer.tokenize(text)
    return preprocess_tokens(tokens, stopwords, lemmatize, lemmatizer)

def preprocess_tokens(tokens: List[str], stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    lemmatizer: Optional[Type[StemmerI]]=None):
    if lemmatize:
        tokens = lemmatize_tokens(tokens, lemmatizer)
    if stopwords:
        tokens = [token for token in tokens if token.lower() not in stopwords]
    return tokens

def preprocess_token(token: str, stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    lemmatizer: Optional[Type[StemmerI]]=None):
    if lemmatize:
        token = lemmatize_token(token, lemmatizer)
    if stopwords and token.lower() in stopwords:
        return ''
    return token

def get_tfidf(words_count: DataFrame):
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf = transformer.fit_transform(words_count.values)
    df_tfidf = DataFrame(tfidf.toarray(), columns=words_count.columns, index=words_count.index)
    return df_tfidf
    
def get_bow_of_tokens(tokens: List[str], preprocess: Optional[bool]=False, stopwords: Optional[List[str]]=None):
    tokens = tokens if not preprocess else preprocess_text(tokens, stopwords)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokens)
    feature_names = vectorizer.get_feature_names_out()
    bow = dict(zip(feature_names, X.sum(axis=0).A1))
    bow = dict(sorted(bow.items(), key=lambda item: item[1], reverse=True))
    return bow

def get_dataframe_bow(dataframe: DataFrame, text_col: str, preprocess: Optional[bool]=False,
                     stopwords: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                     tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None):
    return dataframe[[text_col]].apply(get_bow_of_text, axis=1,
                                       args=(preprocess, stopwords, lemmatize, tokenizer, lemmatizer)
                                       ).apply(Series).fillna(0)

def get_bow_of_text(text: Union[str,Series], preprocess: Optional[bool]=False, stopwords: Optional[List[str]]=None, 
                    lemmatize: Optional[bool]=False, tokenizer: Optional[Type[StringTokenizer]]=None, 
                    lemmatizer: Optional[Type[StemmerI]]=None):
    text = list(text) if isinstance(text, str) else text
    text = text if not preprocess else preprocess_text(text[0], stopwords, lemmatize, tokenizer, lemmatizer)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text)
    feature_names = vectorizer.get_feature_names_out()
    bow = dict(zip(feature_names, X.sum(axis=0).A1))
    bow = dict(sorted(bow.items(), key=lambda item: item[1], reverse=True))
    return bow

def preprocess_country_name(name):
    name = unidecode(name) 
    name = name.lower() 
    name = re.sub(r'[^a-z\s]', '', name)  
    return name

def find_countries_in_paper(tokens: List[str], countries: List[str], demonyms: Dict[str, str],
                            similarity_threshold: Optional[int]=0.9):
    mentioned_countries = []
    for token in tokens:
        _token = token
        if token in list(demonyms.keys()):
            token = demonyms[token]
        elif token == 'uk':
            token = 'united kingdom'
        for country in countries:
            dist = lcs_similarity(token, country)
            if dist >= similarity_threshold:
                mentioned_countries.append((country, _token))
    return mentioned_countries

def sort_multiindex_dataframe(df: DataFrame, bottom_col: str, top_level: int=0, ascending: Optional[bool]=False):
    top_level_values = df.columns.get_level_values(top_level).unique()
    sorted_dfs = []
    for top_value in top_level_values:
        sort_key = tuple((top_value, bottom_col))
        sorted_cluster_df = df.sort_values(by=sort_key, ascending=ascending).reset_index(drop=True)
        sorted_dfs.append(sorted_cluster_df)
    return pd.concat(sorted_dfs, axis=1)

def plot_token_features(df: DataFrame, columns: List[str], 
                        hue: Optional[str]=None, log: Optional[bool]=True, 
                        ncols: Optional[int]=2,
                        figsize: Optional[Tuple]=(4,4)):
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
