from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestCentroid

from ..exceptions import ArgumentStructureError, ArgumentTypeError, ArgumentValueError

CLUSTER_COL_NOT_IN_INDEX_ERROR = (
    "DataFrame provided does not have the {cluster_level} index level!"
)
SINGLE_GROUP_DF_ERROR = "DataFrame provided has a single group!"
EMPTY_DATA_ERROR = "Input DataFrame cannot be empty."
EMPTY_CENTROIDS_ERROR = "Centroids DataFrame cannot be empty."
N_ELEMENTS_COL = "N Elements"


def get_clusters_centroids(
    data: DataFrame,
    cluster_level: Union[str, int],
    metric: Optional[Literal["euclidean", "manhattan"]] = "euclidean",
) -> DataFrame:
    if cluster_level not in data.index.names:
        raise KeyError(f"{CLUSTER_COL_NOT_IN_INDEX_ERROR}")
    if data.empty:
        raise ArgumentStructureError(EMPTY_DATA_ERROR)
    if data.index.get_level_values(cluster_level).nunique() == 1:
        raise ArgumentStructureError(SINGLE_GROUP_DF_ERROR)
    clf = NearestCentroid(metric=metric)
    clf.fit(data.values, data.index.get_level_values(cluster_level).values)
    return DataFrame(
        clf.centroids_,
        columns=data.columns,
        index=np.unique(data.index.get_level_values(cluster_level)),
    )


def get_distances_between_centroids(centroids: DataFrame) -> DataFrame:
    if centroids.empty:
        raise ArgumentStructureError(EMPTY_DATA_ERROR)
    distance_matrix = pairwise_distances(centroids.values)
    return DataFrame(distance_matrix, index=centroids.index, columns=centroids.index)


def get_distances_to_centroids(
    data: DataFrame,
    centroids: DataFrame,
    cluster_level: str,
    metric: Optional[
        Literal[
            "braycurtis",
            "canberra",
            "chebyshev",
            "cityblock",
            "correlation",
            "cosine",
            "dice",
            "euclidean",
            "hamming",
            "jaccard",
            "jensenshannon",
            "kulczynski1",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        ]
    ] = "euclidean",
) -> DataFrame:
    if cluster_level not in data.index.names:
        raise KeyError(f"{CLUSTER_COL_NOT_IN_INDEX_ERROR}")
    if data.empty:
        raise ArgumentStructureError(EMPTY_DATA_ERROR)
    if centroids.empty:
        raise ArgumentStructureError(EMPTY_CENTROIDS_ERROR)
    cluster_labels = data.index.get_level_values(cluster_level)
    corresponding_centroids = centroids.loc[cluster_labels]
    distances = cdist(
        data.values, corresponding_centroids.values, metric="euclidean"
    ).diagonal()
    return DataFrame(
        distances, index=data.index, columns=[f"distance_to_{cluster_level}_centroid"]
    )


def get_clusters_size(data: DataFrame, cluster_level: str) -> DataFrame:
    if cluster_level not in data.index.names:
        raise KeyError(f"{CLUSTER_COL_NOT_IN_INDEX_ERROR}")
    cluster_count = (
        data.index.get_level_values(cluster_level)
        .value_counts()
        .sort_index()
        .to_frame(name=N_ELEMENTS_COL)
    )
    return cluster_count


def get_cosine_similarities(data: DataFrame) -> DataFrame:
    return get_similarities_metric(data, cosine_similarity)


def get_similarities_metric(data: DataFrame, metric: Callable) -> DataFrame:
    if data.empty:
        raise ArgumentStructureError(EMPTY_DATA_ERROR)
    if data.shape[0] == 1:
        raise ArgumentStructureError(SINGLE_GROUP_DF_ERROR)
    similarity_matrix = metric(data.values)
    similarity_df = DataFrame(
        similarity_matrix,
        index=data.index,
        columns=data.index,
    )
    return similarity_df


def get_cosine_similarities_vector(
    data: DataFrame, id_level: Union[str, int]
) -> DataFrame:
    return get_similarities_metric_vector(data, cosine_similarity, id_level)


def get_similarities_metric_vector(
    data: DataFrame, metric: Callable, id_level: Union[str, int]
) -> DataFrame:
    if id_level not in data.index.names and not (
        isinstance(id_level, int) and id_level < data.index.nlevels
    ):
        raise KeyError(f"{CLUSTER_COL_NOT_IN_INDEX_ERROR}")
    similarity_df = get_similarities_metric(data, metric)
    upper_tri_indices = np.triu_indices_from(similarity_df, k=1)
    sample_pairs = [
        (
            similarity_df.index.get_level_values(id_level)[i],
            similarity_df.index.get_level_values(id_level)[j],
        )
        for i, j in zip(*upper_tri_indices)
    ]
    similarity_vector_df = DataFrame(
        similarity_df.values[upper_tri_indices],
        index=pd.MultiIndex.from_tuples(sample_pairs),
        columns=[metric.__name__ if hasattr(metric, "__name__") else "similarity"],
    )
    return similarity_vector_df
