import re
from typing import AnyStr, Dict, Iterable, List, Optional, Tuple, Type, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes
from nltk import FreqDist, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.api import StemmerI
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.api import StringTokenizer
from nltk.util import ngrams
from pandas import DataFrame, Series
from plotly.graph_objects import Sankey
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
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
    tag_dict = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_dict.get(nltk_tag[0], wordnet.NOUN)
    
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

def preprocess_texts(texts: List[str], stop_words: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    tokenizer: Optional[Type[StringTokenizer]]=None) -> List[str]:
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    return [preprocess_text(text, stop_words, lemmatize, tokenizer) for text in texts]

def preprocess_text(text: str, stop_words: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None
                    ) -> List[str]:
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    tokens = tokenizer.tokenize(text)
    return preprocess_tokens(tokens, stop_words, lemmatize, lemmatizer)

def preprocess_tokens(tokens: List[str], stop_words: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    lemmatizer: Optional[Type[StemmerI]]=None) -> List[str]:
    if lemmatize:
        tokens = lemmatize_tokens(tokens, lemmatizer)
    if stop_words:
        tokens = [token for token in tokens if token.lower() not in stop_words]
    return tokens

def preprocess_token(token: str, stop_words: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                    lemmatizer: Optional[Type[StemmerI]]=None) -> str:
    if lemmatize:
        token = lemmatize_token(token, lemmatizer)
    if stop_words and token.lower() in stop_words:
        return ''
    return token

def get_tfidf(words_count: DataFrame) -> DataFrame:
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf = transformer.fit_transform(words_count.values)
    df_tfidf = DataFrame(tfidf.toarray(), columns=words_count.columns, index=words_count.index)
    return df_tfidf
    
def get_bow_of_tokens(tokens: List[str], preprocess: Optional[bool]=False, 
                      stop_words: Optional[List[str]]=None) -> Dict[str,int]:
    tokens = tokens if not preprocess else preprocess_tokens(tokens, stop_words)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokens)
    feature_names = vectorizer.get_feature_names_out()
    bow = dict(zip(feature_names, X.sum(axis=0).A1))
    bow = dict(sorted(bow.items(), key=lambda item: item[1], reverse=True))
    return bow

def get_dataframe_bow(dataframe: DataFrame, text_col: str, preprocess: Optional[bool]=False,
                     stop_words: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                     tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None
                      ) -> DataFrame:
    return dataframe[[text_col]].apply(get_bow_of_text, axis=1,
                                       args=(preprocess, stop_words, lemmatize, tokenizer, lemmatizer)
                                       ).apply(Series).fillna(0.0)

def get_dataframe_bow_chunks(dataframe: DataFrame, text_col: str, preprocess: Optional[bool]=False,
                      stop_words: Optional[List[str]]=None, lemmatize: Optional[bool]=False,
                      tokenizer: Optional[Type[StringTokenizer]]=None, lemmatizer: Optional[Type[StemmerI]]=None,
                      chunk_size: Optional[int]=2_000,
                      words_per_paper: Optional[int]=None
                             ) -> DataFrame:
    def process_chunk(chunk: DataFrame) -> DataFrame:
        return chunk[[text_col]].apply(get_bow_of_text, axis=1,
                           args=(preprocess, stop_words, lemmatize, tokenizer, lemmatizer)
                           ).apply(Series)
    num_chunks = (dataframe.shape[0] + chunk_size - 1) // chunk_size
    chunks = (dataframe.iloc[i:i+chunk_size, :] for i in range(0, dataframe.shape[0], chunk_size))
    processed_chunks = [process_chunk(chunk) for chunk in tqdm(chunks, total=num_chunks, desc="Processing Chunks")]
    return pd.concat(processed_chunks, ignore_index=True).fillna(0.0)

def get_ngram_count(df: DataFrame, text_col: str, id_col: str, tokenizer: Optional[Type[StringTokenizer]]=None, 
                    stop_words: Optional[List[str]]=None, ngram_range: Optional[Tuple[int, int]]=(1,1),
                    max_features: Optional[int]=None, 
                    max_df: Optional[Union[int, float]]=1.0, 
                    min_df: Optional[Union[int, float]]=1,
                    lowercase: Optional[bool]=True
                    ) -> DataFrame:
    if tokenizer is None: 
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    ngrams_counter = CountVectorizer(ngram_range=ngram_range, 
                                     stop_words=stop_words,
                                     max_features=max_features,
                                     max_df=max_df,
                                     min_df=min_df,
                                     lowercase=lowercase,
                                     tokenizer=lambda x: tokenizer.tokenize(x),
                                     token_pattern=None
                                     )
    ngrams_count = ngrams_counter.fit_transform(df[text_col].values).toarray()
    ngrams_count = pd.DataFrame(ngrams_count, 
                                columns=ngrams_counter.get_feature_names_out(),
                                index=df[id_col]
                                )
    ngrams_count = ngrams_count[ngrams_count.sum(axis=0).sort_values(ascending=False).index]
    return ngrams_count

def plot_clusters_ngrams(clusters_ngrams: DataFrame, n_gram: int, ncols: int, n_grams: Optional[int]=20,
                         figsize: Optional[Tuple[int, int]]=(6, 6), ax: Optional[Axes]=None
                         ) -> Axes:
    n_gram = clusters_ngrams.columns.get_level_values(1).unique()[n_gram-1]
    clusters = clusters_ngrams.columns.get_level_values(0).unique()
    nrows = len(clusters) // ncols
    if len(clusters) % ncols != 0: 
        nrows += 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    for cluster, ax in zip(clusters, axes.flat):
        cluster_ngram_frequency = clusters_ngrams[pd.IndexSlice[cluster, n_gram]].set_index('Gram')
        ax = plot_ngrams_count(cluster_ngram_frequency, n_grams=n_grams, ax=ax)
        ax.set_title(cluster)
    fig.tight_layout()
    return ax

def plot_ngrams(grams: DataFrame, n_grams: Optional[Union[int, float]]=20, 
                ax: Optional[Axes]=None) -> Axes:
    grams_count = grams.sum()
    return plot_ngrams_count(grams_count=grams_count, n_grams=n_grams, ax=ax)

def plot_ngrams_count(grams_count: DataFrame, n_grams: Optional[Union[int, float]]=20, 
                      ax: Optional[Axes]=None) -> Axes:
    if isinstance(grams_count, Series):
        grams_count = grams_count.sort_values(ascending=False)
    elif isinstance(grams_count, DataFrame) and len(grams_count.columns) == 1:
        grams_count = grams_count.iloc[:, 0].sort_values(ascending=False)
    if n_grams is not None:
        if isinstance(n_grams, float):
            n_grams = int(len(grams_count)*n_grams)
        grams_count = grams_count.iloc[:n_grams]
    grams_n = len(grams_count.index[0].split(' '))
    _mapping = {1: "Uni", 2:"Bi", 3:"Tri", 4:"Four"}
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(x=grams_count.values, y=grams_count.index, ax=ax)
    ax.set_title(f"{_mapping[grams_n]}-Grams Frequency")
    ax.set_ylabel(f"{_mapping[grams_n]}-Grams")
    ax.set_xlabel('Frequency')
    return ax

def get_dataframe_tokens(df: DataFrame, text_col: str, id_col: str, stop_words: Optional[List[str]]=None, 
                         tokenizer: Optional[Type[StringTokenizer]]=None, lowercase: Optional[bool]=True) -> DataFrame:
    if tokenizer is None:
        tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    df_tokens = df[text_col].apply(tokenizer.tokenize)
    df_tokens.index = df[id_col]
    if lowercase:
        df_tokens = df_tokens.apply(lambda tokens: [t.lower() for t in tokens])
    if stop_words is not None:
        df_tokens = df_tokens.apply(lambda tokens: [t for t in tokens if t not in stop_words])
    max_len = df_tokens.apply(len).max()
    tokens_df = pd.DataFrame({
        id_col: pd.Series(tokens).reindex(range(max_len))
        for id_col, tokens in df_tokens.items()
    })
    return tokens_df

def get_clustered_dataframe_tokens(df: DataFrame, text_col: str, id_col: str, cluster_col: str) -> DataFrame:
    df_tokens = get_dataframe_tokens(df, text_col, id_col)
    papers_clusters = df.set_index(id_col)[cluster_col]
    cluster_columns = {cl: f"Cluster {cl}" for cl in papers_clusters.unique()}
    mapped_clusters = papers_clusters.map(cluster_columns)
    cluster_columns = df_tokens.columns.map(mapped_clusters)
    df_tokens.columns = pd.MultiIndex.from_tuples(list(zip(cluster_columns, df_tokens.columns)))
    return df_tokens

def get_clusters_ngrams(df: DataFrame, text_col: str, id_col: str, 
                        cluster_col: str, max_features: int, 
                        stop_words: Optional[List[str]]=None, 
                        ngram_range: Optional[Tuple[int, int]]=(1,5),
                        frequency: Optional[bool]=True,
                        lowercase: Optional[bool]=False) -> DataFrame:
    clusters_ngrams = []
    for cluster in tqdm(df[cluster_col].unique(), desc="Processing clusters"):
        cluster_texts_ids = df.query(f"{cluster_col} == @cluster")[id_col]
        cluster_texts = df[df[id_col].isin(cluster_texts_ids)]
        for gram_n in range(*ngram_range):
            cluster_ngrams = get_cluster_ngrams(cluster_texts=cluster_texts, 
                                                    text_col=text_col, 
                                                    id_col=id_col, 
                                                    cluster=cluster, 
                                                    stop_words=stop_words, 
                                                    max_features=max_features, 
                                                    gram_n=gram_n,
                                                    frequency=frequency,
                                                    lowercase=lowercase
                                                )
            clusters_ngrams.append(cluster_ngrams)
    clusters_ngrams_frequency_df = pd.concat(clusters_ngrams, axis=1)
    clusters_ngrams_frequency_df.columns = pd.MultiIndex.from_tuples(clusters_ngrams_frequency_df.columns)
    return clusters_ngrams_frequency_df

def get_cluster_ngrams(cluster_texts: pd.DataFrame, text_col: str, id_col: str, cluster: Union[str,int], 
                       gram_n: int, max_features: Optional[int]=None,
                       stop_words: Optional[List[str]]=None,
                       frequency: Optional[bool]=True,
                       lowercase: Optional[bool]=False) -> pd.DataFrame:
        cluster_grams = get_ngram_count(cluster_texts, 
                                        text_col=text_col, 
                                        id_col=id_col, 
                                        stop_words=stop_words, 
                                        ngram_range=(gram_n, gram_n),
                                        max_features=max_features
                                        )
        cluster_grams = cluster_grams.sum()
        if frequency:
            cluster_grams /= cluster_grams.sum()
        cluster_grams = cluster_grams.sort_values(ascending=False).to_frame().reset_index()
        sub_cols = ['Gram', 'Frequency'] if frequency else ['Gram', 'Count']
        cluster_grams.columns = [(f"Cluster {cluster}", f"{gram_n}-Gram", col) for col in sub_cols]
        if not lowercase:
            cluster_grams.iloc[:, 0] = cluster_grams.iloc[:, 0].apply(lambda x: x.title())
        return cluster_grams

def get_bow_of_text(text: Union[str,Series], preprocess: Optional[bool]=False, stop_words: Optional[List[str]]=None, 
                    lemmatize: Optional[bool]=False, tokenizer: Optional[Type[StringTokenizer]]=None, 
                    lemmatizer: Optional[Type[StemmerI]]=None) -> Dict[str,int]:
    text = list(text) if isinstance(text, str) else text
    text = text if not preprocess else preprocess_text(text[0], stop_words, lemmatize, tokenizer, lemmatizer)
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

def find_cities_in_texts(df: DataFrame, pattern: List[str], text_col: str) -> DataFrame:
    matches = (df[text_col].str
               .extractall(pattern))
    mentioned_cities = pd.DataFrame(index=df.index.get_level_values(0).unique(), columns=['cities'])
    for idx, g in matches.reset_index()[['match', 0]].groupby('match'):
        mentioned_cities.loc[idx, 'cities'] = g[0].values.tolist()
    mentioned_cities = mentioned_cities['cities'].apply(pd.Series)
    mentioned_cities = mentioned_cities.sort_index()
    return mentioned_cities

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

def strings_to_group_patterns(strings: List[str], union: str) -> str:
    return union.join([f"(?=.*{s.lower()})" for s in strings])

def plot_token_features(df: DataFrame, columns: List[str], 
                        hue: Optional[str]=None, log: Optional[bool]=True, 
                        ncols: Optional[int]=2, bins: Optional[int]=30,
                        figsize: Optional[Tuple]=(4,4)) -> Axes:
    nrows = (len(columns) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*nrows, figsize[1]*ncols))
    for n, var in enumerate(columns):
        ax = axes[n//ncols, n%ncols]
        sns.histplot(data=df, x=var, hue=None, bins=bins, kde=False, ax=ax)
        if log:
            ax.set_yscale("log")
        ax.set_title(f"Histogram for {var} in Papers' Text")
    plt.tight_layout()
    plt.show()
    return axes

def sankey_plot_clusters_ngrams(clusters_ngrams: DataFrame, n_gram: int, min_ngram: Optional[int]=0, 
                                max_ngram: Optional[int]=20) -> Sankey:
    n_gram = clusters_ngrams.columns.get_level_values(1).unique()[n_gram-1]
    clusters_ngram = clusters_ngrams.loc[:, pd.IndexSlice[:,n_gram,:]]
    common_ngrams = [g.droplevel([0, 1], axis=1) for n, g in clusters_ngram.groupby(level=[0, 1], axis=1)]
    common_ngrams = pd.concat(common_ngrams, axis=0)
    common_ngrams = common_ngrams.groupby('Gram').sum().sort_values(by='Frequency', ascending=False)
    sources_labels = clusters_ngrams.columns.get_level_values(0).unique()
    targets_labels = common_ngrams.index[min_ngram:max_ngram].tolist()
    labels = [*sources_labels, *targets_labels]
    labels_ids = {label: n for n, label in enumerate(labels)}
    x_pos, y_pos = gen_clusters_ngrams_sankey_positions(labels, sources_labels)
    sankey_nodes = {
        'label': labels,
        'x': x_pos,
        'y': y_pos,
        'pad': 5,
        'thickness': 20
    }
    labels_colors = gen_clusters_ngrams_sankey_colors(sources_labels, targets_labels)
    nodes_colors = gen_clusters_ngrams_sankey_nodes_colors(labels, labels_colors)
    sources, targets, values = gen_clusters_ngrams_sankey_links(clusters_ngram, 
                                                                labels_ids, 
                                                                sources_labels, 
                                                                targets_labels
                                                                )
    sankey_links = {
        'source': sources,
        'target': targets,
        'value': values,
    }   
    links_colors = gen_clusters_ngrams_sankey_links_colors(labels_ids, targets, labels_colors)
    sankey_data = go.Sankey(link=sankey_links, node=sankey_nodes, arrangement='snap')
    fig = go.Figure(sankey_data)
    fig.update_layout(width=1000, height=900, font_size=16)
    fig.update_traces(node_color=nodes_colors, link_color=links_colors)
    fig = go.Figure(sankey_data)
    fig.update_layout(width=1000, height=900, font_size=16)
    fig.update_traces(node_color=nodes_colors, link_color=links_colors)
    return fig

def gen_clusters_ngrams_sankey_nodes_colors(labels: List[str], labels_colors: Dict[str, Tuple[int, int, int, int]]
                                            ) -> List[str]:
    nodes_colors = [labels_colors[l] for l in labels]
    nodes_colors = [f"rgba({c[0]},{c[1]},{c[2]},{c[3]})" for c in nodes_colors]
    return nodes_colors

def gen_clusters_ngrams_sankey_links_colors(labels_ids: Dict[str, str], targets: List[str], 
                                            labels_colors: Dict[str, Tuple[int, int, int]]
                                            ) -> List[Tuple[int, int, int, int]]:
    reverse_labels_ids = {value: key for key, value in labels_ids.items()}       
    links_colors = [labels_colors[reverse_labels_ids.get(label)] for label in targets]
    links_colors = [f"rgba({c[0]},{c[1]},{c[2]},{0.5})" for c in links_colors]
    return links_colors

def gen_clusters_ngrams_sankey_positions(labels: List[str], sources_labels: List[str]
                                         ) -> Tuple[List[float], List[float]]:
    x_pos = [0.0 for _ in sources_labels] + [1.0 for _ in labels[len(sources_labels):]]
    y_pos = list(np.linspace(0.0, 1.0, len(sources_labels))
                 ) + list(np.linspace(0.0, 1.0, len(labels) - len(sources_labels)))
    return [max(min(v, 0.999), 0.001) for v in x_pos], [max(min(v, 0.999), 0.001) for v in y_pos]

def gen_clusters_ngrams_sankey_colors(sources_labels: List[str], targets_labels: List[str]
                                      ) -> Dict[str, Tuple[int, int, int, int]]:
    sources_colors = sns.color_palette("Paired")
    sorted_colors = {cluster: sources_colors[i] for i, cluster in zip([1, 0, 7, 6, 3, 2, 5], sources_labels)}
    sources_colors = {w: [*c, 1.0] for w, c in sorted_colors.items()}
    spectral_colors = mpl.colormaps.get_cmap("Spectral_r")
    targets_colors = spectral_colors(np.linspace(0, 1, len(targets_labels)))
    targets_colors = {w: [*c, 0.5] for w, c in zip(targets_labels, targets_colors)}
    return {**targets_colors, **sources_colors}

def gen_clusters_ngrams_sankey_links(clusters_ngram: DataFrame, labels_ids: Dict[str, str], sources_labels: List[str], 
                                     targets_labels: List[str]) -> Tuple[List[str], List[str], List[float]]:
    sources, targets, values = [], [], []
    for label, source_id in labels_ids.items():
        if label in sources_labels:
            flows = clusters_ngram.loc[:, pd.IndexSlice[label, :]]
            for index, flow in flows.iterrows():
                gram, value = flow.values
                if gram in targets_labels and value > 0.0:
                    sources.append(source_id)
                    targets.append(labels_ids[gram])
                    values.append(value * 100)
    return sources, targets, values
