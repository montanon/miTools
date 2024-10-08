from os import PathLike
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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
from tqdm import tqdm

from ..exceptions import ArgumentStructureError, ArgumentTypeError, ArgumentValueError

EMPTY_DATA_ERROR = "Input DataFrame cannot be empty."
MAX_CLUSTERS_TYPE_ERROR = "n_clusters provided must be a positive int larger than 2."
MIN_CLUSTERS_VALUE_ERROR = "n_clusters must be a number larger or equal than 2."
MAX_CLUSTERS_VALUE_ERROR = "n_clusters must be a number larger or equal than 2."
X_Y_SIZE_ERROR = "x values and y values must be the same size."

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


def get_cosine_similarities(data: DataFrame, cluster_col: str) -> ndarray:
    cosine_similarities = data.groupby(level=cluster_col).apply(cosine_similarity)
    return cosine_similarities


def plot_clusters_evolution(
    dataframe: DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    time_col: str,
    time_values: Tuple,
    labels: Optional[List[Union[str, int]]] = None,
    colors: Optional[List[Tuple]] = None,
    plots_kwargs: Optional[Dict[str, Dict]] = None,
) -> Axes:
    fig, axes = plt.subplot_mosaic(
        [["a", "a"], ["a", "a"], ["b", "c"]], layout="constrained", figsize=(14, 14)
    )

    if labels is None:
        labels = dataframe[cluster_col].unique()

    plot_clusters(
        dataframe,
        cluster_col,
        x_col,
        y_col,
        labels=labels,
        ax=axes["a"],
        colors=colors,
        **plots_kwargs.get("a", {}) if plots_kwargs is not None else {},
    )
    axes["a"].set_xlabel("")
    axes["a"].set_ylabel("")
    axes["a"].set_xticks([])
    axes["a"].set_yticks([])
    axes["a"].set_title("Historical Record of Clusters' Embeddings")

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
        ax=axes["b"],
        **plots_kwargs.get("b", {}) if plots_kwargs is not None else {},
    )
    years = (
        dataframe.loc[
            (dataframe[time_col] < time_values[1])
            & (dataframe[time_col] > time_values[0]),
            time_col,
        ]
        .sort_values()
        .unique()
        .astype(np.int16)
    )
    axes["b"].set_xlabel(f"Before Paris Agreement, {time_values[0]} to {years[-1]}")
    axes["b"].set_ylabel("")
    axes["b"].set_xticks([])
    axes["b"].set_yticks([])

    second_period_df = dataframe.loc[dataframe[time_col] >= time_values[1]]
    plot_clusters(
        second_period_df,
        cluster_col,
        x_col,
        y_col,
        labels=labels,
        colors=colors,
        ax=axes["c"],
        **plots_kwargs.get("c", {}) if plots_kwargs is not None else {},
    )
    years = (
        dataframe.loc[dataframe[time_col] >= time_values[1], time_col]
        .sort_values()
        .unique()
        .astype(np.int16)
    )
    axes["c"].set_xlabel(f"After Paris Agreement, {years[0]} to {years[-1]}")
    axes["c"].set_ylabel("")
    axes["c"].set_xticks([])
    axes["c"].set_yticks([])

    handles, labels = axes["c"].get_legend_handles_labels()

    lgnd = fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(1.2525, 0.52)
    )
    for handle in lgnd.legend_handles:
        handle.set_sizes([100.0])

    return axes


def plot_clusters(
    data: DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    ax: Optional[Axes] = None,
    labels: Optional[List] = None,
    colors: Optional[List[Tuple]] = None,
    **kwargs: Dict[str, Any],
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(14, 10))
    if kwargs is None:
        kwargs = dict(alpha=0.75, marker="o", size=5)
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
            **kwargs,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def add_clusters_centroids(
    ax: Axes,
    data: DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    colors: Optional[List[Tuple]] = None,
    labels: Optional[List[Tuple]] = None,
    **kwargs: Dict[str, Any],
) -> Axes:
    if labels is None:
        labels = data[cluster_col].unique()
    if colors is None:
        colors = sns.color_palette("husl", len(labels))[::1]
    if kwargs is None or "zorder" not in kwargs:
        kwargs["zorder"] = 0

    for i, cls in enumerate(labels):
        ax.plot(
            data[data[cluster_col] == cls][x_col],
            data[data[cluster_col] == cls][y_col],
            color=colors[i],
            **kwargs,
        )
    return ax


def add_clusters_ellipse(
    ax: Axes,
    data: DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    colors: Optional[List[Tuple]] = None,
    labels: Optional[List[Tuple]] = None,
    **kwargs: Dict[str, Any],
) -> Axes:
    if labels is None:
        labels = data[cluster_col].unique()
    if colors is None:
        colors = sns.color_palette("husl", len(labels))[::1]

    for i, cls in enumerate(labels):
        ax = confidence_ellipse(
            data[data[cluster_col] == cls][x_col],
            data[data[cluster_col] == cls][y_col],
            ax,
            edgecolor=colors[i],
            **kwargs,
        )
    return ax


def confidence_ellipse(
    xvalues, yvalues, ax, n_std=1.96, facecolor="none", **kwargs
) -> Axes:
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
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(xvalues)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(yvalues)

    transformation = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transformation + ax.transData)
    ax.add_patch(ellipse)

    return ax


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


def plot_cosine_similarities(
    cosine_similarities: Dict[int, DataFrame],
    normed: Optional[bool] = False,
    colors: Optional[List[Tuple]] = None,
    bins: Optional[bool] = False,
) -> Axes:
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    if colors is None:
        colors = sns.color_palette("husl", len(cosine_similarities))[::-1]
    for cl, similarities in tqdm(cosine_similarities.items()):
        upper_tri_vals = similarities[np.triu_indices(similarities.shape[0], k=1)]
        if not normed:
            if bins:
                ax = sns.histplot(
                    upper_tri_vals,
                    bins=30,
                    ax=ax,
                    alpha=0.05,
                    stat="density",
                    color=colors[cl],
                    legend=False,
                )
            ax = sns.kdeplot(
                upper_tri_vals, ax=ax, color=colors[cl], label=f"Cluster {cl}"
            )
        else:
            kde = gaussian_kde(upper_tri_vals)
            x_vals = np.linspace(min(upper_tri_vals), max(upper_tri_vals), 1000)
            y_vals = kde(x_vals) / max(kde(x_vals))
            ax.plot(x_vals, y_vals, alpha=1.0, label=f"Cluster {cl}", color=colors[cl])

    ax.set_title("Distributions of Cosine Similarities per Cluster")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.legend()

    return ax


def plot_distances_to_centroids(
    distances: DataFrame, cluster_col: str, colors: Optional[List[Tuple]] = None
) -> Axes:
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    if colors is None:
        colors = sns.color_palette("husl", len(distances.index.unique()))[::1]
    for cl, distances in tqdm(distances.groupby(cluster_col)):
        distances = distances[0].values
        kde = gaussian_kde(distances)
        x_vals = np.linspace(min(distances), max(distances), 1000)
        y_vals = kde(x_vals) / max(kde(x_vals))
        ax.plot(x_vals, y_vals, alpha=1.0, label=cl, color=colors[cl])

    ax.set_title(
        "Standardized Distribution of Distances to Centroid of Embeddings by Cluster"
    )
    ax.set_xlabel("Distance to Centroid")
    ax.set_ylabel("Density")
    ax.legend()

    return ax
