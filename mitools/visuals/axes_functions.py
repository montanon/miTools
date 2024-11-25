from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from mitools.exceptions import ArgumentStructureError, ArgumentTypeError


def adjust_axes_lims(
    axes: Axes,
    mode: Optional[str] = "all",
    x: Optional[bool] = True,
    y: Optional[bool] = True,
) -> Axes:
    if not (x or y):
        return axes

    # Check if axes are 1D or a single axis
    if isinstance(axes, np.ndarray) and len(axes.shape) == 1:
        nrows, ncols = axes.shape[0], 1
        axes = axes[:, np.newaxis]
    elif isinstance(axes, np.ndarray) and len(axes.shape) == 2:
        nrows, ncols = axes.shape
    else:
        nrows, ncols = 1, 1
        axes = np.array([[axes]])

    if mode == "all":
        if x:
            xlim_min, xlim_max = get_axes_limits(axes.flat, lambda ax: ax.get_xlim())
            set_axes_limits(
                axes.flat, lambda ax, lim: ax.set_xlim(*lim), xlim_min, xlim_max
            )
        if y:
            ylim_min, ylim_max = get_axes_limits(axes.flat, lambda ax: ax.get_ylim())
            set_axes_limits(
                axes.flat, lambda ax, lim: ax.set_ylim(*lim), ylim_min, ylim_max
            )
    elif mode == "rows":
        for i in range(nrows):
            if x:
                xlim_min, xlim_max = get_axes_limits(
                    axes[i, :], lambda ax: ax.get_xlim()
                )
                set_axes_limits(
                    axes[i, :], lambda ax, lim: ax.set_xlim(*lim), xlim_min, xlim_max
                )
            if y:
                ylim_min, ylim_max = get_axes_limits(
                    axes[i, :], lambda ax: ax.get_ylim()
                )
                set_axes_limits(
                    axes[i, :], lambda ax, lim: ax.set_ylim(*lim), ylim_min, ylim_max
                )
    elif mode == "columns":
        for j in range(ncols):
            if x:
                xlim_min, xlim_max = get_axes_limits(
                    axes[:, j], lambda ax: ax.get_xlim()
                )
                set_axes_limits(
                    axes[:, j], lambda ax, lim: ax.set_xlim(*lim), xlim_min, xlim_max
                )
            if y:
                ylim_min, ylim_max = get_axes_limits(
                    axes[:, j], lambda ax: ax.get_ylim()
                )
                set_axes_limits(
                    axes[:, j], lambda ax, lim: ax.set_ylim(*lim), ylim_min, ylim_max
                )
    else:
        raise ValueError(
            f'Unknown mode: {mode}, must be one of "all", "rows", "columns"'
        )

    return axes


def get_axes_limits(axes: Axes, get_lim_func: Callable) -> Tuple[float, float]:
    lim_min, lim_max = float("inf"), float("-inf")
    for ax in axes:
        if not is_ax_empty(ax):
            lim1, lim2 = get_lim_func(ax)
            lim_min, lim_max = min(lim1, lim_min), max(lim2, lim_max)
    return lim_min, lim_max


def set_axes_limits(
    axes: Axes, set_lim_func: Callable, lim_min: float, lim_max: float
) -> None:
    for ax in axes:
        set_lim_func(ax, (lim_min, lim_max))


def adjust_text_axes_limits(ax: Axes, text: str) -> None:
    ax.figure.canvas.draw()
    bbox = text.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    bbox_transformed = bbox.transformed(ax.transData.inverted())
    right_x = bbox_transformed.x1
    ax.set_xlim(ax.get_xlim()[0], right_x)
    ax.autoscale(enable=True, axis="both", tight=False)


def adjust_ax_fontsize(ax: Axes, fontsize: int) -> None:
    if not isinstance(ax, Axes):
        raise ArgumentTypeError("ax must be an instance of matplotlib.axes.Axes")
    if not isinstance(fontsize, (int, str)):
        raise ArgumentTypeError(f"'fontsize'={fontsize} must be an integer or str")
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)


def adjust_axes_fontsize(
    axes: Iterable[Axes], fontsizes: Union[List[int], int]
) -> Iterable[Axes]:
    fontsizes = [fontsizes] * len(axes) if isinstance(fontsizes, int) else fontsizes
    if len(fontsizes) != len(axes):
        raise ArgumentStructureError(
            f"Length of 'fontsizes'={len(fontsizes)} must be equal to the number of axes={len(axes)}"
        )
    for ax, fontsize in zip(axes, fontsizes):
        adjust_ax_fontsize(ax, fontsize)
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
    )
