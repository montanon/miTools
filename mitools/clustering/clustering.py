import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from pandas import DataFrame
from matplotlib.axes import Axes

def kmeans_ncluster_search(dataframe: DataFrame, max_clusters: Optional[int]=25, 
                           random_state: Optional[int]=0, n_init: Optional[str]='auto'):
    silhouette_scores = []
    inertia = []
    for n_clusters in tqdm(range(2, max_clusters)):
        kmeans_clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        kmeans_clustering.fit_predict(dataframe)
        score = silhouette_score(dataframe, kmeans_clustering.labels_)
        silhouette_scores.append(score)
        inertia.append(kmeans_clustering.inertia_)
    return silhouette_scores, inertia

def kmeans_clustering(dataframe: DataFrame, n_clusters: int,
                      random_state: Optional[int]=0, n_init: Optional[str]='auto'):
    kmeans_clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return kmeans_clustering.fit_predict(dataframe)

def plot_kmeans_ncluster_search(silhouette_scores: List[float], inertia: List[float]):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

    x_values_silhouette = range(0, len(silhouette_scores))
    ax[0].plot(range(0, len(silhouette_scores)), silhouette_scores, 'bx-')
    ax[0].set_title('K-Means Clustering Algorithm Silhouette Score')
    ax[0].set_xlabel('N° of Clusters')
    ax[0].set_ylabel('Silhouette Score')
    ax[0].set_xticks(x_values_silhouette)

    min_y_silhouette = ax[0].get_ylim()[0]
    for x, y in zip(x_values_silhouette, silhouette_scores):
        ax[0].vlines(x, min_y_silhouette, y, linestyles='dotted', colors='grey', linewidth=0.5)
    
    diff = np.diff(inertia)
    diff_r = diff[1:] / diff[:-1]
    elbow = np.where(diff_r < np.mean(diff_r))[0][0]

    x_values_inertia = range(2, 25)
    ax[1].plot(x_values_inertia, inertia, 'bx-', label='Inertia')
    ax[1].vlines(x_values_inertia[elbow], ax[1].get_ylim()[0], ax[1].get_ylim()[1], linestyles='dashed', colors='r', label='Elbow')
    ax[1].set_title('K-Means Clustering Algorithm Inertia')
    ax[1].set_xticks(x_values_inertia)
    ax[1].legend()
    
    min_y_inertia = ax[1].get_ylim()[0]
    for x, y in zip(x_values_inertia, inertia):
        ax[1].vlines(x, min_y_inertia, y, linestyles='dotted', colors='grey', linewidth=0.5)

    plt.tight_layout()

    plt.show()

def agglomerative_ncluster_search(dataframe: DataFrame, max_clusters: Optional[int]=25):
    silhouette_scores = []
    for n_clusters in tqdm(range(2, max_clusters)):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        agg_clustering.fit_predict(dataframe)
        score = silhouette_score(dataframe, agg_clustering.labels_)
        silhouette_scores.append(score)
    return silhouette_scores

def agglomerative_clustering(dataframe: DataFrame, n_clusters: int):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return agg_clustering.fit_predict(dataframe)

def plot_agglomerative_ncluster_search(silhouette_scores: List[float]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))

    x_values_silhouette = range(0, len(silhouette_scores))
    ax.plot(range(0, len(silhouette_scores)), silhouette_scores, 'bx-')
    ax.set_title('Agglomerative Clustering Algorithm Silhouette Score')
    ax.set_xlabel('N° of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_xticks(x_values_silhouette)

    min_y_silhouette = ax.get_ylim()[0]
    for x, y in zip(x_values_silhouette, silhouette_scores):
        ax.vlines(x, min_y_silhouette, y, linestyles='dotted', colors='grey', linewidth=0.5)
    
    plt.show()

def plot_clusters(dataframe: DataFrame, cluster_col: str, x_col: str, y_col: str,
                  ax: Optional[Axes]=None, labels: Optional[bool]=True, **kwargs: Dict[str, Any]):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 10))

    if kwargs is None:
        kwargs = dict(alpha=0.75, marker='o', size=5)
        
    classes = dataframe[cluster_col].sort_values().unique()
    colors = sns.color_palette("cubehelix", len(classes))
    
    for i, cls in enumerate(classes):
        ax.scatter(
            dataframe[dataframe[cluster_col] == cls][x_col], 
            dataframe[dataframe[cluster_col] == cls][y_col],
            color=colors[i],
            label=cls if labels else None,
            **kwargs)
        
    ax.set_xticks([])
    ax.set_yticks([])

    return ax