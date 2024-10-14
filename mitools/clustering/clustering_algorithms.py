from os import PathLike
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame
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
