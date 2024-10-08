from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def create_figure(with_inertia: bool) -> Tuple[plt.Figure, List[Axes]]:
    if with_inertia:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
        return fig, axes if isinstance(axes, np.ndarray) else [axes]
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
