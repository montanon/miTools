from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from numpy import ndarray
from pandas import DataFrame, IndexSlice
from scipy.spatial.distance import euclidean
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

from ..exceptions import (ArgumentStructureError, ArgumentTypeError,
                          ArgumentValueError)

N_ELEMENTS_COL = 'N Elements'

MAX_CLUSTERS_TYPE_ERROR = 'max_clusters provided must be a positive int larger than 2.'
MAX_CLUSTERS_VALUE_ERROR = 'max_clusters must be a number larger or equal than 2.'
X_Y_SIZE_ERROR = "x values and y values must be the same size."
CLUSTER_COL_NOT_IN_INDEX_ERROR = 'DataFrame provided does not have the {cluster_col} index level!'
SINGLE_GROUP_DF_ERROR = 'DataFrame provided has a single group!'
EMPTY_DF_ERROR = 'DataFrame provided is empty!'


def kmeans_ncluster_search(data: DataFrame, max_clusters: Optional[int]=25, 
                           random_state: Optional[int]=0, n_init: Optional[str]='auto'
                           ) -> Tuple[List[float], List[float]]:
    if not isinstance(max_clusters, int):
        raise ArgumentTypeError(MAX_CLUSTERS_TYPE_ERROR)
    if max_clusters < 2:
        raise ArgumentValueError(MAX_CLUSTERS_VALUE_ERROR)
    silhouette_scores = []
    inertia = []
    for n_clusters in tqdm(range(2, max_clusters)):
        kmeans_clusters = kmeans_clustering(data, n_clusters, random_state, n_init)
        score = silhouette_score(data, kmeans_clusters.labels_)
        silhouette_scores.append(score)
        inertia.append(kmeans_clusters.inertia_)
    return silhouette_scores, inertia

def kmeans_clustering(data: DataFrame, n_clusters: int, random_state: Optional[int]=0, 
                      n_init: Optional[str]='auto') -> ndarray:
    kmeans_clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans_clustering.fit_predict(data)
    return kmeans_clustering

def agglomerative_ncluster_search(data: DataFrame, max_clusters: Optional[int]=25) -> List[float]:
    if not isinstance(max_clusters, int):
        raise ArgumentTypeError(MAX_CLUSTERS_TYPE_ERROR)
    if max_clusters < 2:
        raise ArgumentValueError(MAX_CLUSTERS_VALUE_ERROR)
    silhouette_scores = []
    for n_clusters in tqdm(range(2, max_clusters)):
        agg_clustering = agglomerative_clustering(data, n_clusters)
        score = silhouette_score(data, agg_clustering.labels_)
        silhouette_scores.append(score)
    return silhouette_scores

def agglomerative_clustering(data: DataFrame, n_clusters: int) -> ndarray:
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    agg_clustering.fit_predict(data)
    return agg_clustering

def get_clusters_centroids(data: DataFrame, cluster_col: str) -> DataFrame:
    if cluster_col not in data.index.names:
        raise KeyError(f"{CLUSTER_COL_NOT_IN_INDEX_ERROR}")
    if data.index.get_level_values(cluster_col).nunique() == 1:
        raise ArgumentStructureError(SINGLE_GROUP_DF_ERROR)
    clf = NearestCentroid()
    clf.fit(data.values, data.index.get_level_values(cluster_col).values)
    return DataFrame(clf.centroids_, columns=data.columns, 
                     index=np.unique(data.index.get_level_values(cluster_col)))

def get_clusters_centroids_distances(centroids: DataFrame) -> DataFrame:
    if centroids.empty:
        raise ArgumentStructureError(EMPTY_DF_ERROR)
    return DataFrame(
        pairwise_distances(centroids)
        )

def get_cosine_similarities(data: DataFrame, cluster_col: str) -> ndarray:
    cosine_similarities = (data.groupby(level=cluster_col)
                           .apply(cosine_similarity)
                           )
    return cosine_similarities

def get_distances_to_centroids(data: DataFrame, centroids: DataFrame, cluster_col: str) -> DataFrame:
    distances = []
    label_pos = data.index.names.index(cluster_col)
    for idx, values in data.iterrows():
        cluster = idx[label_pos]
        centroid = centroids.loc[cluster]
        distance = euclidean(values, centroid)
        distances.append(distance)
    distances = DataFrame(distances)
    distances.index = data.index.get_level_values(1)
    return distances.sort_index()

def plot_kmeans_ncluster_search(silhouette_scores: List[float], inertia: List[float]) -> Axes:
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

    return ax

def plot_agglomerative_ncluster_search(silhouette_scores: List[float]) -> Axes:
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
    
    return ax

def plot_clusters_evolution(dataframe: DataFrame, cluster_col: str, x_col: str, y_col: str, 
                            time_col: str, time_values: Tuple,
                            labels: Optional[List[Union[str,int]]]=None,
                            colors: Optional[List[Tuple]]=None,
                            plots_kwargs:  Optional[Dict[str, Dict]]=None
                            ) -> Axes:
    fig, axes = plt.subplot_mosaic([['a', 'a'], ['a', 'a'], ['b', 'c']],
                              layout='constrained',
                              figsize=(14, 14))

    if labels is None:
        labels = dataframe[cluster_col].unique()

    plot_clusters(dataframe, cluster_col, x_col, y_col, labels=labels, 
                  ax=axes['a'], colors=colors,
                  **plots_kwargs.get('a', {}) if plots_kwargs is not None else {}
                  )
    axes['a'].set_xlabel('')
    axes['a'].set_ylabel('')
    axes['a'].set_xticks([])
    axes['a'].set_yticks([])
    axes['a'].set_title("Historical Record of Clusters' Embeddings")

    first_period_df = dataframe.loc[
            (dataframe[time_col] < time_values[1]) & (dataframe[time_col] > time_values[0])
        ].copy()

    plot_clusters(
        first_period_df, 
        cluster_col,
        x_col,
        y_col,
        labels=labels,
        colors=colors,
        ax=axes['b'], 
        **plots_kwargs.get('b', {}) if plots_kwargs is not None else {}
        )
    years = (dataframe.loc[(dataframe[time_col] < time_values[1]) & (dataframe[time_col] > time_values[0]), time_col]
             .sort_values()
             .unique()
             .astype(np.int16))
    axes['b'].set_xlabel(f'Before Paris Agreement, {time_values[0]} to {years[-1]}')
    axes['b'].set_ylabel('')
    axes['b'].set_xticks([])
    axes['b'].set_yticks([])
    
    second_period_df = dataframe.loc[
        dataframe[time_col] >= time_values[1]
    ]
    plot_clusters(
        second_period_df, 
        cluster_col,
        x_col, 
        y_col,
        labels=labels,
        colors=colors,
        ax=axes['c'], 
        **plots_kwargs.get('c', {}) if plots_kwargs is not None else {}
        )
    years = dataframe.loc[dataframe[time_col] >= time_values[1], time_col].sort_values().unique().astype(np.int16)
    axes['c'].set_xlabel(f'After Paris Agreement, {years[0]} to {years[-1]}')
    axes['c'].set_ylabel('')
    axes['c'].set_xticks([])
    axes['c'].set_yticks([])

    handles, labels = axes['c'].get_legend_handles_labels()
    
    lgnd = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.2525, 0.52))
    for handle in lgnd.legend_handles:
        handle.set_sizes([100.0])

    return axes

def plot_clusters(data: DataFrame, cluster_col: str, x_col: str, y_col: str,
                  ax: Optional[Axes]=None, labels: Optional[List]=None,
                   colors: Optional[List[Tuple]]=None, **kwargs: Dict[str, Any]) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 10))
    if kwargs is None:
        kwargs = dict(alpha=0.75, marker='o', size=5)
    if labels is None:
        labels = data[cluster_col].unique()
    if colors is None:
        colors = sns.color_palette("husl", len(labels))[::1]
    
    for i, cls in enumerate(labels):
        ax.scatter(
            data[data[cluster_col] == cls][x_col], 
            data[data[cluster_col] == cls][y_col],
            color=colors[i],
            label=cls if labels is not None else None,
            zorder=99,
            **kwargs)
        
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def add_clusters_centroids(ax: Axes, data: DataFrame, cluster_col: str,
                           x_col: str, y_col: str, colors: Optional[List[Tuple]]=None,
                           labels: Optional[List[Tuple]]=None,
                           **kwargs: Dict[str, Any]
                           ) -> Axes:
    if labels is None:
        labels = data[cluster_col].unique()
    if colors is None:
        colors = sns.color_palette("husl", len(labels))[::1]

    for i, cls in enumerate(labels):
        ax.plot(
            data[data[cluster_col] == cls][x_col],
            data[data[cluster_col] == cls][y_col],
            color=colors[i],
            zorder=0,
            **kwargs
        )
    return ax

def add_clusters_confidence_ellipse(ax: Axes, data: DataFrame, cluster_col: str,
                                    x_col: str, y_col: str,
                                    colors: Optional[List[Tuple]]=None,
                                    labels: Optional[List[Tuple]]=None,
                                    **kwargs: Dict[str, Any]) -> Axes:
    if labels is None:
        labels = data[cluster_col].unique()
    if colors is None:
        colors = sns.color_palette("husl", len(labels))[::1]
        
    for i, cls in enumerate(labels):
        ax = confidence_ellipse(data[data[cluster_col] == cls][x_col],
                                data[data[cluster_col] == cls][y_col],
                                ax,
                                edgecolor=colors[i],
                                **kwargs
                                )
    return ax

def confidence_ellipse(xvalues, yvalues, ax, n_std=1.96, facecolor='none', **kwargs) -> Axes:
    if xvalues.size != yvalues.size:
        raise ArgumentStructureError(X_Y_SIZE_ERROR)
  
    cov = np.cov(xvalues.astype(float), yvalues.astype(float), rowvar=False)
    pearson_corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse_radius_x = np.sqrt(1 + pearson_corr)
    ellipse_radius_y = np.sqrt(1 - pearson_corr)
    ellipse = Ellipse(
        (0, 0), 
        width=ellipse_radius_x * 2, 
        height=ellipse_radius_y * 2, 
        facecolor=facecolor, 
        **kwargs
        )
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(xvalues)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(yvalues)

    transformation = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transformation + ax.transData)
    ax.add_patch(ellipse)

    return ax

def display_clusters_size(data: DataFrame, cluster_col: str) -> DataFrame:
    cluster_count = data[[cluster_col]].value_counts().sort_index().to_frame()
    cluster_count.columns = [N_ELEMENTS_COL]
    return cluster_count

def plot_clusters_growth(data: DataFrame, time_col: str, cluster_col: str) -> Axes:
    clusters_count = data.groupby(time_col)[cluster_col].value_counts().to_frame().sort_index(axis=1, level=1)

    fig, ax = plt.subplots(1, 1, figsize=(21,7))

    clusters = clusters_count.index.get_level_values(1).unique().sort_values()
    times = clusters_count.index.get_level_values(0).unique().sort_values()

    colors = sns.color_palette("husl", len(clusters))

    for cl in clusters:
        cluster_papers = clusters_count.loc[IndexSlice[:, cl], :]
        cluster_papers.index = cluster_papers.index.droplevel(1)
        cluster_papers = cluster_papers.reindex(times, fill_value=0)
        ax.plot(times[:-1], cluster_papers['count'][:-1], c=colors[cl])

    ax.set_title('Cluster Size Evolution')
    ax.set_ylabel('N° Elements')
    ax.set_xlabel('Year')

    return ax

def plot_cosine_similarities(cosine_similarities: Dict[int,DataFrame],
                             normed: Optional[bool]=False) -> Axes:
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    palette = sns.color_palette('husl', len(cosine_similarities))[::-1]
    for cl, similarities in cosine_similarities.items():
        upper_tri_vals = similarities[np.triu_indices(similarities.shape[0], k=1)]
        if not normed:
            ax = sns.histplot(upper_tri_vals, bins=30, ax=ax, alpha=0.05, stat="density", 
                              color=palette[cl], legend=False)
            ax = sns.kdeplot(upper_tri_vals, ax=ax, color=palette[cl], label=f"Cluster {cl}")
        else:
            kde = gaussian_kde(upper_tri_vals)
            x_vals = np.linspace(min(upper_tri_vals), max(upper_tri_vals), 1000)
            y_vals = kde(x_vals) / max(kde(x_vals))
            ax.plot(x_vals, y_vals, alpha=1.0, label=f"Cluster {cl}", color=palette[cl])

    ax.set_title('Distributions of Cosine Similarities per Cluster')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    return ax

def plot_distances_to_centroids(distances: DataFrame, cluster_col: str) -> Axes:
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    palette = sns.color_palette('husl', len(distances.index.unique()))[::1]
    for cl, distances in distances.groupby(cluster_col):
        distances = distances[0].values
        kde = gaussian_kde(distances)
        x_vals = np.linspace(min(distances), max(distances), 1000)
        y_vals = kde(x_vals) / max(kde(x_vals))
        ax.plot(x_vals, y_vals, alpha=1.0, label=f"Cluster {cl}", color=palette[cl])

    ax.set_title('Standardized Distribution of Distances to Centroid of Embeddings by Cluster')
    ax.set_xlabel('Distance to Centroid')
    ax.set_ylabel('Density')
    ax.legend()

    return ax
