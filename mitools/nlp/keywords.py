import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

import re
from typing import List, Optional, Tuple, Dict
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode

from ..utils import lcs_similarity

def split_strings(str_list: List[str]):
    new_list = []
    for s in str_list:
        new_list += re.split('(?=[A-Z])', s)
    return new_list

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
    
def lemmatize_sentence(sentence: str):
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence.lower()))  
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

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

def preprocess_text(text: str, stop_words: List[str]):
    tokenizer = RegexpTokenizer("[A-Za-z]{2,}[0-9]{,1}")
    tokens = tokenizer.tokenize(text)
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatize_sentence(token) for token in tokens]
    keywords = [lemma for lemma in lemmas if lemma not in stop_words]
    return keywords

def get_bow_of_text(tokens: List[str]):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokens)
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