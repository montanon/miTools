import warnings
from ast import literal_eval
from os import PathLike
from typing import Callable, Dict, List, Literal, Sequence, Union

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from matplotlib.axes import Axes
from nltk.tokenize.api import StringTokenizer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from numpy import float64, ndarray
from pandas import DataFrame, Series
from sklearn.manifold import TSNE
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from umap import UMAP

from mitools.etl import CustomConnection
from mitools.exceptions import ArgumentValueError
from mitools.utils import iterable_chunks

warnings.simplefilter("ignore", NumbaDeprecationWarning)
warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)
import umap.plot

SPECTER_EMBEDDINGS_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16

UMAP_METRICS = Literal[
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "canberra",
    "braycurtis",
    "mahalanobis",
    "wminkowski",
    "seuclidean",
    "cosine",
    "correlation",
    "haversine",
    "hamming",
    "jaccard",
    "dice",
    "russelrao",
    "kulsinski",
    "ll_dirichlet",
    "hellinger",
    "rogerstanimoto",
    "sokalmichener",
    "sokalsneath",
    "yule",
]


def get_device() -> str:
    return (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def specter_embed_texts(
    texts: Union[List[str], str],
    tokenizer_length: int = 512,
    batch_size: int = None,
    device: str = None,
    return_device: str = "cpu",
    pooling: Literal["mean", "cls"] = "cls",
    output_type: Literal["tensor", "numpy", "list"] = "numpy",
    task: Literal[
        "proximity", "classification", "regression", "adhoc_query"
    ] = "proximity",
) -> Union[Tensor, ndarray, List[List]]:
    if task not in ["proximity", "classification", "regression", "adhoc_query"]:
        raise ArgumentValueError(
            f"'task'={task} must be one from ['proximity', 'classification', 'regression', 'adhoc_query']"
        )
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as=task, set_active=True)
    return huggingface_embed_texts(
        texts=texts,
        model=model,
        tokenizer="allenai/specter2_base",
        tokenizer_length=tokenizer_length,
        batch_size=batch_size,
        device=device,
        return_device=return_device,
        pooling=pooling,
        output_type=output_type,
    )


def huggingface_embed_texts(
    texts: Union[List[str], str],
    model: Union[AutoModel, str],
    tokenizer: Union[AutoTokenizer, str],
    tokenizer_length: int = 512,
    batch_size: int = None,
    device: str = None,
    return_device: str = "cpu",
    pooling: Literal["mean", "cls"] = "cls",
    output_type: Literal["tensor", "numpy", "list"] = "numpy",
) -> Union[Tensor, ndarray, List[List]]:
    if not texts:
        raise ArgumentValueError(
            "'texts' must be a non-empty list of strings or single string."
        )
    if pooling not in ["mean", "cls"]:
        raise ArgumentValueError(
            f"'pooling'={pooling} must be one from ['mean', 'cls']"
        )
    if output_type not in ["tensor", "numpy", "list"]:
        raise ArgumentValueError(
            f"'output_type'={output_type} must be one from ['tensor', 'numpy', 'list']"
        )
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
        raise ArgumentValueError("'batch_size' must be an integer larger than 0.")
    texts = [texts] if isinstance(texts, str) else texts
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer)
        if isinstance(tokenizer, str)
        else tokenizer
    )
    model = AutoModel.from_pretrained(model) if isinstance(model, str) else model
    device = device or get_device()
    model = model.to(device)
    if batch_size is not None:
        embeddings_chunks = []
        for chunk in iterable_chunks(texts, batch_size):
            embeddings = _generate_embeddings(
                texts=chunk,
                model=model,
                tokenizer=tokenizer,
                tokenizer_length=tokenizer_length,
                device=device,
                return_device=return_device,
                pooling=pooling,
            )
            embeddings_chunks.append(embeddings)
        embeddings = torch.vstack(embeddings_chunks)
    else:
        embeddings = _generate_embeddings(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            tokenizer_length=tokenizer_length,
            device=device,
            return_device=return_device,
            pooling=pooling,
        )
    if output_type == "tensor":
        return embeddings
    elif output_type == "numpy":
        return embeddings.cpu().numpy()
    elif output_type == "list":
        return embeddings.cpu().numpy().tolist()


def _generate_embeddings(
    texts: List[str],
    model: Union[AutoModel, str],
    tokenizer: Union[AutoTokenizer, str],
    tokenizer_length: int = 512,
    device: str = None,
    return_device: str = "cpu",
    pooling: Literal["mean", "cls"] = "cls",
) -> Tensor:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=tokenizer_length,
    ).to(device)
    with torch.no_grad():
        result = model(**inputs)
    if pooling == "cls":
        embeddings = result.last_hidden_state[:, 0, :]
    elif pooling == "mean":
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        sum_embeddings = torch.sum(result.last_hidden_state * attention_mask, dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
        embeddings = sum_embeddings / sum_mask
    return embeddings.detach().to(return_device)


def tsne_embeddings(
    data: Union[DataFrame, ndarray],
    return_reducer: bool = False,
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: Union[float, str] = "auto",
    n_iter: int = 1_000,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-7,
    metric: Union[str, Callable] = "euclidean",
    metric_params: Dict = None,
    init: Union[str, ndarray] = "pca",
    verbose: int = 0,
    random_state: int = 42,
    method: Literal["barnes_hut", "exact"] = "barnes_hut",
    angle: float = 0.5,
    n_jobs: int = None,
):
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
        metric=metric,
        metric_params=metric_params,
        init=init,
        verbose=verbose,
        random_state=random_state,
        method=method,
        angle=angle,
        n_jobs=n_jobs,
    )
    embeddings = tsne.fit_transform(
        data.values if isinstance(data, DataFrame) else data
    )
    return embeddings if not return_reducer else tsne


# https://umap-learn.readthedocs.io/en/latest/api.html
def umap_embeddings(
    data: Union[DataFrame, ndarray],
    return_reducer: bool = False,
    n_neighbors: float = 15,
    n_components: int = 2,
    metric: UMAP_METRICS = "euclidean",
    n_epochs: int = None,
    learning_rate: float = 1.0,
    init: Literal["spectral", "random", "pca", "tswspectral"] = "spectral",
    min_dist: float = 0.1,
    spread: float = 1.0,
    low_memory: bool = False,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 5,
    transform_queue_size: float = 4.0,
    a: float = None,
    b: float = None,
    random_state: int = 42,
    angular_rp_forest: bool = False,
    target_n_neighbors: int = -1,
    target_metric: Literal["categorical", "l1", "l2"] = "categorical",
    target_metric_kwds: Dict = None,
    target_weight: float = 0.5,
    transform_seed: int = 42,
    verbose: bool = False,
    unique: bool = False,
    densmap: bool = False,
    dens_lambda: float = 2.0,
    dens_frac: float = 0.3,
    dens_var_shift: float = 0.1,
    output_dens: bool = False,
    disconnection_distance: float = np.inf,
    precomputed_knn: tuple = (None, None, None),
) -> ndarray:
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        init=init,
        min_dist=min_dist,
        spread=spread,
        low_memory=low_memory,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        transform_queue_size=transform_queue_size,
        a=a,
        b=b,
        random_state=random_state,
        angular_rp_forest=angular_rp_forest,
        target_n_neighbors=target_n_neighbors,
        target_metric=target_metric,
        target_metric_kwds=target_metric_kwds,
        target_weight=target_weight,
        transform_seed=transform_seed,
        verbose=verbose,
        unique=unique,
        densmap=densmap,
        dens_lambda=dens_lambda,
        dens_frac=dens_frac,
        dens_var_shift=dens_var_shift,
        output_dens=output_dens,
        disconnection_distance=disconnection_distance,
        precomputed_knn=precomputed_knn,
    )
    embeddings = reducer.fit_transform(
        data.values if isinstance(data, DataFrame) else data
    )
    return embeddings if not return_reducer else reducer


def plot_umap_points(
    reducer: UMAP,
    labels: Sequence = None,
    color_key_cmap: str = "Paired",
    background: str = "white",
    theme: str = None,
    width: int = 800,
    height: int = 800,
    show_legend=True,
    ax: Axes = None,
    alpha: float = 1.0,
) -> Axes:
    if labels is not None and len(labels) != len(reducer.embedding_):
        raise ArgumentValueError(
            "The number of labels must match the number of embeddings."
        )
    return umap.plot.points(
        umap_object=reducer,
        labels=labels,
        color_key_cmap=color_key_cmap,
        theme=theme,
        background=background,
        width=width,
        height=height,
        show_legend=show_legend,
        ax=ax,
        alpha=alpha,
    )


def plot_umap_interactive(
    reducer: UMAP,
    labels: Sequence = None,
    values: Sequence = None,
    hover_data: DataFrame = None,
    tools: List = None,
    theme: str = None,
    cmap: str = "Blues",
    color_key: Union[Dict, ndarray] = None,
    color_key_cmap: str = "Paired",
    background: str = "white",
    width: int = 800,
    height: int = 800,
    alpha: float = 1.0,
    point_size: int = None,
    subset_points: Sequence[bool] = None,
    interactive_text_search: bool = False,
    interactive_text_search_columns: bool = None,
    interactive_text_search_alpha_contrast: float = 0.95,
):
    if values is not None and len(values) != len(reducer.embedding_):
        raise ArgumentValueError(
            "The number of 'values' must match the number of embeddings."
        )
    if labels is not None and len(labels) != len(reducer.embedding_):
        raise ArgumentValueError(
            "The number of labels must match the number of embeddings."
        )
    umap.plot.output_notebook()
    pl = umap.plot.interactive(
        umap_object=reducer,
        labels=labels,
        values=values,
        hover_data=hover_data,
        tools=tools,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=background,
        width=width,
        height=height,
        alpha=alpha,
        point_size=point_size,
        subset_points=subset_points,
        interactive_text_search=interactive_text_search,
        interactive_text_search_columns=interactive_text_search_columns,
        interactive_text_search_alpha_contrast=interactive_text_search_alpha_contrast,
    )
    umap.plot.show(pl)


def plot_umap_connectivity(
    reducer: UMAP,
    labels: Sequence = None,
    edge_bundling: Union[Literal["hammer"], None] = None,
    edge_cmap: str = "gray_r",
    show_points: bool = False,
    values: Sequence = None,
    theme: str = None,
    cmap: str = "Blues",
    color_key: Union[Dict, ndarray] = None,
    color_key_cmap: str = "Paired",
    background: str = "white",
    width: int = 800,
    height: int = 800,
) -> Axes:
    if values is not None and len(values) != len(reducer.embedding_):
        raise ArgumentValueError(
            "The number of 'values' must match the number of embeddings."
        )
    if labels is not None and len(labels) != len(reducer.embedding_):
        raise ArgumentValueError(
            "The number of labels must match the number of embeddings."
        )
    return umap.plot.connectivity(
        umap_object=reducer,
        labels=labels,
        edge_bundling=edge_bundling,
        edge_cmap=edge_cmap,
        show_points=show_points,
        values=values,
        theme=theme,
        cmap=cmap,
        color_key=color_key,
        color_key_cmap=color_key_cmap,
        background=background,
        width=width,
        height=height,
    )


def plot_umap_diagnostic(
    reducer: UMAP,
    diagnostic_type: Literal[
        "pca", "ica", "vq", "local_dim", "neighborhood", "all"
    ] = "pca",
    nhood_size: int = 15,
    local_variance_threshold: float = 0.8,
    cmap: str = "viridis",
    point_size: int = None,
    background: str = "white",
    width: int = 800,
    height: int = 800,
    ax: Axes = None,
) -> Axes:
    return umap.plot.diagnostic(
        umap_object=reducer,
        diagnostic_type=diagnostic_type,
        nhood_size=nhood_size,
        local_variance_threshold=local_variance_threshold,
        cmap=cmap,
        point_size=point_size,
        background=background,
        width=width,
        height=height,
        ax=ax,
    )
