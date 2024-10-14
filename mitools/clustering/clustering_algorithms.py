from os import PathLike
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame, IndexSlice
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from ..exceptions import ArgumentStructureError, ArgumentTypeError, ArgumentValueError

EMPTY_DATA_ERROR = "Input DataFrame cannot be empty."
MAX_CLUSTERS_TYPE_ERROR = "n_clusters provided must be a positive int larger than 2."
MIN_CLUSTERS_VALUE_ERROR = "n_clusters must be a number larger or equal than 2."
MAX_CLUSTERS_VALUE_ERROR = "n_clusters must be a number larger or equal than 2."


WARD_AFFINITY_ERROR = "Ward linkage only allows for euclidean affinity!"
DISTANCE_THRESHOLD_ERROR = "If distance_threshold is not None, n_clusters must be None and compute_full_tree must be True!"


def kmeans_clustering(
    data: DataFrame,
    n_clusters: int,
    random_state: Optional[int] = 0,
    n_init: Optional[int] = 10,
    max_iter: Optional[int] = 300,
    algorithm: Optional[Literal["auto", "full", "elkan"]] = "auto",
    tol: Optional[float] = 1e-4,
    verbose: Optional[bool] = False,
) -> Tuple[KMeans, ndarray]:
    if data.empty:
        raise ArgumentStructureError(EMPTY_DATA_ERROR)
    if not isinstance(n_clusters, int):
        raise ArgumentTypeError(MAX_CLUSTERS_TYPE_ERROR)
    if n_clusters < 2:
        raise ArgumentValueError(MIN_CLUSTERS_VALUE_ERROR)
    if n_clusters > len(data):
        raise ArgumentValueError(MAX_CLUSTERS_VALUE_ERROR)
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
        tol=tol,
        verbose=verbose,
    )
    cluster_labels = kmeans_model.fit_predict(data)
    return kmeans_model, cluster_labels


def agglomerative_clustering(
    data: DataFrame,
    n_clusters: int,
    metric: Optional[
        Union[
            Literal["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
            Callable,
        ]
    ] = "euclidean",
    memory: Optional[PathLike] = None,
    connectivity: Optional[Union[ndarray, Callable]] = None,
    compute_full_tree: Optional[Union[Literal["auto"], bool]] = "auto",
    linkage: Optional[Literal["ward", "complete", "average", "single"]] = "ward",
    distance_threshold: Optional[float] = None,
    compute_distances: Optional[bool] = False,
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    if linkage == "ward" and metric != "euclidean":
        raise ArgumentValueError(WARD_AFFINITY_ERROR)
    if distance_threshold is not None and (
        n_clusters is not None or not compute_full_tree
    ):
        raise ArgumentValueError(DISTANCE_THRESHOLD_ERROR)
    if data.empty:
        raise ArgumentStructureError(EMPTY_DATA_ERROR)
    if not isinstance(n_clusters, int):
        raise ArgumentTypeError(MAX_CLUSTERS_TYPE_ERROR)
    if n_clusters < 2:
        raise ArgumentValueError(MIN_CLUSTERS_VALUE_ERROR)
    if n_clusters > len(data):
        raise ArgumentValueError(MAX_CLUSTERS_VALUE_ERROR)
    agg_clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        memory=memory,
        connectivity=connectivity,
        compute_full_tree=compute_full_tree,
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=compute_distances,
    )
    cluster_labels = agg_clustering_model.fit_predict(data)
    return agg_clustering_model, cluster_labels


def clustering_ncluster_search(
    data: DataFrame,
    max_clusters: Optional[int] = 25,
    clustering_method: Optional[
        Union[kmeans_clustering, agglomerative_clustering]
    ] = kmeans_clustering,
    score_metric: Optional[Callable] = silhouette_score,
    **kwargs,
) -> Tuple[List[float], Union[List[float], None]]:
    if not isinstance(max_clusters, int):
        raise ArgumentTypeError(MAX_CLUSTERS_TYPE_ERROR)
    if max_clusters < 2:
        raise ArgumentValueError(MAX_CLUSTERS_VALUE_ERROR)
    silhouette_scores = []
    inertia = [] if clustering_method == kmeans_clustering else None
    for n_clusters in tqdm(range(2, max_clusters)):
        model, labels = clustering_method(data, n_clusters, **kwargs)
        score = score_metric(data, labels)
        silhouette_scores.append(score)
        if inertia is not None:
            inertia.append(model.inertia_)
    return silhouette_scores, inertia


def plot_clusters_growth(
    data: DataFrame,
    time_col: str,
    cluster_col: str,
    colors: Optional[List[Tuple]] = None,
) -> Axes:
    clusters_count = (
        data.groupby(time_col)[cluster_col]
        .value_counts()
        .to_frame()
        .sort_index(axis=1, level=1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(21, 7))

    clusters = clusters_count.index.get_level_values(1).unique().sort_values()
    times = clusters_count.index.get_level_values(0).unique().sort_values()

    if colors is None:
        colors = sns.color_palette("husl", len(clusters))

    for cl in clusters:
        cluster_papers = clusters_count.loc[IndexSlice[:, cl], :]
        cluster_papers.index = cluster_papers.index.droplevel(1)
        cluster_papers = cluster_papers.reindex(times, fill_value=0)
        ax.plot(
            cluster_papers.index[:-1], cluster_papers["count"].values[:-1], c=colors[cl]
        )

    ax.set_title("Cluster Size Evolution")
    ax.set_ylabel("N° Elements")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left")

    return ax


def plot_clusters_growth_stacked(
    data: DataFrame,
    time_col: str,
    cluster_col: str,
    colors: Optional[Dict[str, Tuple]] = None,
    filtered_clusters: Optional[List[str]] = None,
    share_pct: Optional[bool] = False,
) -> Axes:
    topics = data[cluster_col].unique()
    clusters_count = data.groupby([time_col, cluster_col]).size().unstack(fill_value=0)
    clusters_count = clusters_count[topics]

    if filtered_clusters:
        clusters_count = clusters_count[
            [c for c in clusters_count.columns if c not in filtered_clusters]
        ]
    if share_pct:
        clusters_count = clusters_count.div(clusters_count.sum(axis=1), axis=0) * 100

    _, ax = plt.subplots(figsize=(21, 7))

    if colors is None:
        colors = sns.color_palette("husl", len(clusters_count.columns))
    else:
        colors = [colors[c] for c in clusters_count.columns]

    times = clusters_count.index
    cluster_values = [clusters_count[cluster].values for cluster in clusters_count]

    ax.stackplot(times, cluster_values, labels=clusters_count.columns, colors=colors)

    ax.set_title("Stacked Cluster Size Evolution")
    ax.set_ylabel("N° Elements")
    ax.set_xlabel("Year")
    if not share_pct:
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return ax
