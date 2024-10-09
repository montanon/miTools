from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from pandas import DataFrame
from scipy.stats import gaussian_kde


def create_figure(with_inertia: bool) -> Tuple[plt.Figure, List[Axes]]:
    if with_inertia:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
        return fig, axes
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5))
        return fig, [ax]


def plot_silhouette_scores(
    ax: Axes, silhouette_scores: List[float], algorithm_name: str
) -> None:
    x_values = range(2, len(silhouette_scores) + 2)
    ax.plot(x_values, silhouette_scores, "bx-")
    ax.set_title(f"{algorithm_name} Silhouette Score")
    ax.set_xlabel("N° of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(x_values)
    min_y = ax.get_ylim()[0]
    for x, y in zip(x_values, silhouette_scores):
        ax.vlines(x, min_y, y, linestyles="dotted", colors="grey", linewidth=0.5)


def plot_inertia(
    ax: Axes, inertia: List[float], algorithm_name: str, max_clusters: int
) -> None:
    x_values = range(2, max_clusters)
    ax.plot(x_values, inertia, "gx-", label="Inertia")
    diff = np.diff(inertia)
    diff_r = diff[1:] / diff[:-1]
    elbow = np.where(diff_r < np.mean(diff_r))[0][0]
    ax.vlines(
        x_values[elbow],
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        linestyles="dashed",
        colors="r",
        label="Elbow",
    )
    ax.set_title(f"{algorithm_name} Inertia")
    ax.set_xlabel("N° of Clusters")
    ax.set_xticks(x_values)
    ax.legend()
    min_y = ax.get_ylim()[0]
    for x, y in zip(x_values, inertia):
        ax.vlines(x, min_y, y, linestyles="dotted", colors="grey", linewidth=0.5)


def plot_clustering_ncluster_search(
    silhouette_scores: List[float],
    inertia: Optional[List[float]] = None,
    max_clusters: Optional[int] = 25,
    algorithm_name: Optional[str] = "Clustering Algorithm",
) -> List[Axes]:
    fig, axes = create_figure(inertia is not None)

    plot_silhouette_scores(axes[0], silhouette_scores, algorithm_name)

    if inertia is not None:
        plot_inertia(axes[1], inertia, algorithm_name, max_clusters)

    plt.tight_layout()
    return axes


def plot_df_col_distribution(
    dataframe: DataFrame,
    column: Union[str, int],
    normed: Optional[bool] = False,
    color: Optional[Union[str, Tuple[float], Tuple[int]]] = None,
    bins: Optional[Union[int, None]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    if color is None:
        color = sns.color_palette("husl", 1)
    df_values = dataframe.iloc[:, column].values
    col_name = (
        dataframe.columns[0].title().replace("_", " ")
        if isinstance(column, int)
        else column
    )
    if not normed:
        if bins:
            ax = sns.histplot(
                df_values,
                bins=bins,
                ax=ax,
                alpha=0.5,
                stat="density",
                color=color,
                legend=False,
            )
        ax = sns.kdeplot(df_values, ax=ax, color=color)
    else:
        kde = gaussian_kde(df_values)
        x_vals = np.linspace(min(df_values), max(df_values), 1000)
        y_vals = kde(x_vals) / max(kde(x_vals))
        ax.plot(x_vals, y_vals, alpha=1.0, color=color)
    ax.set_title(f"Distributions of {col_name}")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Frequency")
    ax.legend()
    return ax


def plot_dfs_col_distribution(
    dataframes: Iterable[DataFrame],
    column: Union[str, int],
    normed: Optional[bool] = False,
    colors: Optional[Union[str, Tuple[float], Tuple[int]]] = None,
    bins: Optional[Union[int, None]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    if colors is None:
        colors = sns.color_palette("husl", len(dataframes))
    for dataframe, color in zip(dataframes, colors):
        plot_df_col_distribution(
            dataframe=dataframe,
            column=column,
            normed=normed,
            color=color,
            bins=bins,
            ax=ax,
        )
    return ax


def plot_clusters(
    data: DataFrame,
    cluster_level: Union[int, str],
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
        labels = data.index.get_level_values(cluster_level).unique()
    if colors is None:
        colors = sns.color_palette("husl", len(labels))[::1]
    for i, cls in enumerate(labels):
        ax.scatter(
            data[data.index.get_level_values(cluster_level) == cls][x_col],
            data[data.index.get_level_values(cluster_level) == cls][y_col],
            color=colors[i],
            label=cls if labels is not None else None,
            zorder=99,
            **kwargs,
        )
    return ax
