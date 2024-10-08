from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestCentroid

from ..exceptions import ArgumentStructureError, ArgumentTypeError, ArgumentValueError

CLUSTER_COL_NOT_IN_INDEX_ERROR = (
    "DataFrame provided does not have the {cluster_col} index level!"
)
SINGLE_GROUP_DF_ERROR = "DataFrame provided has a single group!"
EMPTY_DATA_ERROR = "Input DataFrame cannot be empty."
EMPTY_CENTROIDS_ERROR = "Centroids DataFrame cannot be empty."


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
