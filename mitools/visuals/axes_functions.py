from typing import Callable, Iterable, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text
from numpy import ndarray

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)

FONTSIZES = ["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]


def adjust_axes_array_limits(
    axes: Union[ndarray, Iterable[Axes], Axes],
    mode: Literal["all", "rows", "columns"] = "all",
    x: bool = True,
    y: bool = True,
) -> Axes:
    if mode not in {"all", "rows", "columns"}:
        raise ArgumentValueError(
            f'Unknown mode: {mode}, must be one of "all", "rows", "columns"'
        )
    axes = [axes] if isinstance(axes, Axes) else axes
    if isinstance(axes, list) and all(isinstance(ax, Axes) for ax in axes):
        axes = np.array(axes, dtype=object)
    if isinstance(axes, ndarray):
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]
        nrows, ncols = axes.shape
    else:
        axes = np.array([[axes]])
        nrows, ncols = 1, 1
    if axes.size == 0:
        raise ArgumentValueError("The provided axes array is empty.")
    if not (x or y):
        return axes

    def adjust_axes_limits(axes: Iterable[Axes], axis: Literal["x", "y"]) -> None:
        lim_min, lim_max = get_axes_limits(axes, axis)
        set_axes_limits(axes, lim_min, lim_max, axis)

    if mode == "all":
        if x:
            adjust_axes_limits(list(axes.flat), "x")
        if y:
            adjust_axes_limits(list(axes.flat), "y")
    elif mode == "rows":
        for i in range(nrows):
            if x:
                adjust_axes_limits(axes[i, :], "x")
            if y:
                adjust_axes_limits(axes[i, :], "y")
    elif mode == "columns":
        for j in range(ncols):
            if x:
                adjust_axes_limits(axes[:, j], "x")
            if y:
                adjust_axes_limits(axes[:, j], "y")
    else:
        raise ArgumentValueError(
            f"Unknown mode: {mode}, must be one of {'all', 'rows', 'columns'}"
        )

    return axes


def get_axes_limits(
    axes: Union[Iterable[Axes], Axes],
    axis: Literal["x", "y"] = "x",
    get_lim_func: Callable = None,
) -> Tuple[float, float]:
    axes = validate_axes(axes)
    if axis not in {"x", "y"}:
        raise ArgumentValueError(
            f"The 'axis'={axis} parameter must be one of 'x' or 'y'."
        )
    if not isinstance(get_lim_func, Callable):

        def get_lim_func(ax: Axes) -> Tuple[float, float]:
            return ax.get_xlim() if axis == "x" else ax.get_ylim()

    lim_min, lim_max = float("inf"), float("-inf")
    for ax in axes:
        if not is_ax_empty(ax):
            lim1, lim2 = get_lim_func(ax)
            lim_min, lim_max = min(lim1, lim_min), max(lim2, lim_max)
    if lim_min == float("inf") or lim_max == float("-inf"):
        raise ArgumentValueError("No valid limits found across the provided axes.")
    return lim_min, lim_max


def set_axes_limits(
    axes: Union[Iterable[Axes], Axes],
    lim_min: float,
    lim_max: float,
    axis: Literal["x", "y"] = "x",
    set_lim_func: Callable = None,
) -> Iterable[Axes]:
    axes = validate_axes(axes)
    if axis not in {"x", "y"}:
        raise ArgumentValueError(
            f"The 'axis'={axis} parameter must be one of 'x' or 'y'."
        )
    if not isinstance(set_lim_func, Callable):

        def set_lim_func(ax: Axes, limits: Tuple[float, float]) -> Axes:
            return ax.set_xlim(limits) if axis == "x" else ax.set_ylim(limits)

    for ax in axes:
        set_lim_func(ax, (lim_min, lim_max))
    return axes


def adjust_ax_text_limits(
    ax: Axes, text: Text, axis: Literal["x", "y", "both"] = "both"
) -> Axes:
    if not isinstance(ax, Axes):
        raise ArgumentTypeError(
            "The 'ax' parameter must be an instance of matplotlib.axes.Axes."
        )
    if not isinstance(text, Text):
        raise ArgumentTypeError(
            "The 'text' parameter must be an instance of matplotlib.text.Text."
        )
    if axis not in ("x", "y", "both"):
        raise ArgumentValueError(
            "The 'axis' parameter must be one of 'x', 'y', or 'both'."
        )
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer)
    bbox_transformed = bbox.transformed(ax.transData.inverted())
    if axis in ("x", "both"):
        current_xlim = ax.get_xlim()
        if (
            bbox_transformed.x1 > current_xlim[1]
            or bbox_transformed.x0 < current_xlim[0]
        ):
            ax.set_xlim(
                min(bbox_transformed.x0, current_xlim[0]),
                max(bbox_transformed.x1, current_xlim[1]),
            )
    if axis in ("y", "both"):
        current_ylim = ax.get_ylim()
        if (
            bbox_transformed.y1 > current_ylim[1]
            or bbox_transformed.y0 < current_ylim[0]
        ):
            ax.set_ylim(
                min(bbox_transformed.y0, current_ylim[0]),
                max(bbox_transformed.y1, current_ylim[1]),
            )

    return ax


def adjust_axes_text_limits(
    axes: Union[Iterable[Axes], Axes],
    texts: Union[Iterable[Text], Text],
    axis: Literal["x", "y", "both"] = "both",
) -> Iterable[Axes]:
    axes = validate_axes(axes)
    texts = [texts] if isinstance(texts, Text) else texts
    if texts is None or not isinstance(texts, list):
        raise ArgumentTypeError(
            "fontsizes must be an integer or a string or an iterable of such values"
        )
    if len(axes) != len(texts):
        raise ArgumentStructureError(
            f"The number of axes={len(axes)} must be equal to the number of texts={len(texts)}"
        )
    for ax, text in zip(axes, texts):
        adjust_ax_text_limits(ax, text, axis)
    return axes


def adjust_ax_labels_fontsize(
    ax: Axes,
    fontsize: Union[int, str],
) -> Axes:
    if not isinstance(ax, Axes):
        raise ArgumentTypeError("ax must be an instance of matplotlib.axes.Axes")
    if not isinstance(fontsize, int) and fontsize not in FONTSIZES:
        raise ArgumentTypeError(
            f"'fontsize'={fontsize} must be an integer or be one of {FONTSIZES}"
        )
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    return ax


def adjust_axes_labels_fontsize(
    axes: Union[Iterable[Axes], Axes],
    fontsizes: Union[List[int], int, List[str], str],
) -> Iterable[Axes]:
    axes = validate_axes(axes)
    if isinstance(fontsizes, int) or isinstance(fontsizes, str):
        fontsizes = [fontsizes] * len(axes)
    if fontsizes is None or not isinstance(fontsizes, list):
        raise ArgumentTypeError(
            "fontsizes must be an integer or a string or an iterable of such values"
        )
    if len(fontsizes) != len(axes):
        raise ArgumentStructureError(
            f"Length of 'fontsizes'={len(fontsizes)} must be equal to the number of axes={len(axes)}"
        )
    for ax, fontsize in zip(axes, fontsizes):
        adjust_ax_labels_fontsize(ax, fontsize)
    return axes


def is_ax_empty(ax: Axes) -> bool:
    if not isinstance(ax, Axes):
        raise ArgumentTypeError("ax must be an instance of matplotlib.axes.Axes")
    return not (
        ax.lines
        or ax.patches
        or ax.collections
        or ax.texts
        or ax.images
        or ax.get_xlabel()
        or ax.get_ylabel()
        or ax.get_legend()
        or ax.get_title()
    )


def are_axes_empty(axes: Union[Iterable[Axes], Axes]) -> bool:
    axes = validate_axes(axes)
    return all(is_ax_empty(ax) for ax in axes)


def validate_axes(axes: Union[Iterable[Axes], Axes]) -> Iterable[Axes]:
    axes = [axes] if isinstance(axes, Axes) else axes
    if axes is None or not (
        isinstance(axes, Iterable) and all(isinstance(ax, Axes) for ax in axes)
    ):
        raise ArgumentTypeError(
            "axes must be an instance of matplotlib.axes.Axes or an iterable of such instances"
        )
    return axes
