import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Color,
    Scale,
)


class PlotterException(Exception):
    pass


class Plotter(ABC):
    def __init__(self, x_data: Any, y_data: Any, **kwargs):
        self.x_data = self._validate_data(x_data, "x_data")
        self.y_data = self._validate_data(y_data, "y_data")
        if len(self.x_data) != len(self.y_data):
            raise ArgumentStructureError(
                f"'x_data' and 'y_data' must be of the same length, {len(x_data)} != {len(y_data)}."
            )
        self.data_size = len(self.x_data)
        self._init_params = {
            "title": {"default": "", "type": Text},
            "xlabel": {"default": "", "type": Text},
            "ylabel": {"default": "", "type": Text},
            "color": {"default": None, "type": Union[Sequence[Color], Color]},
            "alpha": {"default": 1.0, "type": Union[Sequence[float], float]},
            "label": {"default": None, "type": Union[Sequence[str], str]},
            "legend": {"default": None, "type": Union[Dict, None]},
            "zorder": {"default": None, "type": Union[Sequence[float], float]},
            "figsize": {"default": (21, 14), "type": Tuple[float, float]},
            "style": {"default": "dark_background", "type": str},
            "grid": {"default": None, "type": Dict[str, Any]},
            "tight_layout": {"default": False, "type": bool},
            "texts": {"default": None, "type": Union[Sequence[Text], Text]},
            "xscale": {"default": None, "type": Scale},
            "yscale": {"default": None, "type": Scale},
            "background": {"default": None, "type": Color},
            "figure_background": {"default": None, "type": Color},
            "suptitle": {"default": None, "type": Text},
            "xlim": {"default": None, "type": Union[Tuple[float, float], None]},
            "ylim": {"default": None, "type": Union[Tuple[float, float], None]},
            "x_ticks": {
                "default": None,
                "type": Union[Sequence[Union[float, int]], None],
            },
            "y_ticks": {
                "default": None,
                "type": Union[Sequence[Union[float, int]], None],
            },
            "x_tick_labels": {"default": None, "type": Union[Sequence[str], None]},
            "y_tick_labels": {"default": None, "type": Union[Sequence[str], None]},
        }

        for param, config in self._init_params.items():
            setattr(self, param, config["default"])
            if param in kwargs and kwargs[param] is not None:
                setter_name = f"set_{param}"
                if hasattr(self, setter_name):
                    if isinstance(kwargs[param], dict):
                        getattr(self, setter_name)(**kwargs[param])
                    else:
                        getattr(self, setter_name)(kwargs[param])
                else:
                    if param in ["xscale", "yscale"]:
                        self.set_scales(**{param: kwargs[param]})
                    elif param in ["xlim", "ylim"]:
                        self.set_ax_limits(**{param: kwargs[param]})
                    elif param in ["x_ticks", "y_ticks"]:
                        self.set_ticks(**{param: kwargs[param]})
                    elif param in ["x_tick_labels", "y_tick_labels"]:
                        self.set_tick_labels(**{param: kwargs[param]})
                    else:
                        raise ArgumentValueError(f"Parameter '{param}' is not valid.")
        self.figure: Figure = None
        self.ax: Axes = None

    def _validate_data(
        self, data: Sequence[Union[float, int, integer]], name: str
    ) -> Any:
        if not isinstance(data, (list, tuple, ndarray, Series)):
            raise ArgumentTypeError(
                f"{name} must be a sequence of floats, ints, or integers."
            )
        if not all(isinstance(d, (float, int, integer)) for d in data):
            raise ArgumentTypeError(
                f"All elements in {name} must be floats, ints, or integers."
            )
        return data

    def set_title(self, label: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html"""
        self.title = dict(label=label, **kwargs)
        return self

    def set_xlabel(self, xlabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel"""
        self.xlabel = dict(xlabel=xlabel, **kwargs)
        return self

    def set_ylabel(self, ylabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel"""
        self.ylabel = dict(ylabel=ylabel, **kwargs)
        return self

    def set_axes_labels(self, xlabel: str, ylabel: str, **kwargs):
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)
        return self

    def set_style(self, style: str):
        if style in plt.style.available or style is None:
            self.style = style
        else:
            raise ArgumentValueError(
                f"Style '{style}' is not available in Matplotlib styles: {plt.style.available}."
            )
        return self

    @abstractmethod
    def set_color(
        self, color: Union[Sequence[Color], Color, Sequence[float], Sequence[int]]
    ):
        raise NotImplementedError

    def set_alpha(self, alpha: Union[Sequence[float], float]):
        if isinstance(alpha, (list, tuple, ndarray, Series, float, int)):
            if not isinstance(alpha, (float, int)):
                if len(alpha) != len(self.x_data):
                    raise ArgumentStructureError(
                        "alpha must be of the same length as x_data and y_data, "
                        + f"len(alpha)={len(alpha)} != len(x_data)={self.data_size}."
                    )
            self.alpha = alpha
        else:
            raise ArgumentTypeError(
                "alpha must be a numeric value or an array-like of numeric values."
            )
        return self

    def set_label(self, labels: Union[Sequence[str], str]):
        if isinstance(labels, str):
            self.label = labels
        elif isinstance(labels, (list, tuple, ndarray, Series)) and all(
            isinstance(label, str) for label in labels
        ):
            if len(labels) != self.data_size:
                raise ArgumentStructureError(
                    "labels must be of the same length as x_data and y_data, "
                    + f"len(labels)={len(labels)} != len(x_data)={self.data_size}."
                )
            self.label = labels
        else:
            raise ArgumentTypeError("labels must be a str or a sequence of strs.")
        return self

    def set_legend(
        self,
        show: bool = True,
        labels: Union[Sequence[str], str, None] = None,
        handles: Union[Sequence[Any], None] = None,
        loc: Union[str, int] = "best",
        bbox_to_anchor: Union[Tuple[float, float], None] = None,
        ncol: int = 1,
        fontsize: Union[int, str, None] = None,
        title: Union[str, None] = None,
        title_fontsize: Union[int, str, None] = None,
        frameon: bool = True,
        fancybox: bool = True,
        framealpha: float = 0.8,
        edgecolor: Union[str, None] = None,
        facecolor: Union[str, None] = "inherit",
        **kwargs,
    ):
        if show not in [True, False]:
            raise ArgumentTypeError("'show' must be a boolean")
        if "kwargs" not in kwargs:
            legend_kwargs = {
                "loc": loc,
                "ncol": ncol,
                "frameon": frameon,
                "fancybox": fancybox,
                "framealpha": framealpha,
            }
            if labels is not None:
                if isinstance(labels, str):
                    legend_kwargs["labels"] = [labels]
                elif isinstance(labels, (list, tuple)) and all(
                    isinstance(lbl, str) for lbl in labels
                ):
                    legend_kwargs["labels"] = labels
                else:
                    raise ArgumentTypeError(
                        "'labels' must be a string or sequence of strings"
                    )
            if handles is not None:
                if not isinstance(handles, (list, tuple)):
                    raise ArgumentTypeError(
                        "'handles' must be a sequence of Artist objects"
                    )
                legend_kwargs["handles"] = handles
            if bbox_to_anchor is not None:
                if not isinstance(bbox_to_anchor, tuple) or len(bbox_to_anchor) not in [
                    2,
                    4,
                ]:
                    raise ArgumentTypeError(
                        "'bbox_to_anchor' must be a tuple of 2 or 4 floats"
                    )
                legend_kwargs["bbox_to_anchor"] = bbox_to_anchor

            if fontsize is not None:
                if not isinstance(fontsize, (int, str)):
                    raise ArgumentTypeError("'fontsize' must be an integer or string")
                legend_kwargs["fontsize"] = fontsize

            if title is not None:
                if not isinstance(title, str):
                    raise ArgumentTypeError("'title' must be a string")
                legend_kwargs["title"] = title

            if title_fontsize is not None:
                if not isinstance(title_fontsize, (int, str)):
                    raise ArgumentTypeError(
                        "'title_fontsize' must be an integer or string"
                    )
                legend_kwargs["title_fontsize"] = title_fontsize

            if edgecolor is not None:
                if not isinstance(edgecolor, str):
                    raise ArgumentTypeError("'edgecolor' must be a string")
                legend_kwargs["edgecolor"] = edgecolor

            if facecolor is not None:
                if not isinstance(facecolor, str):
                    raise ArgumentTypeError("'facecolor' must be a string")
                legend_kwargs["facecolor"] = facecolor
            legend_kwargs.update(kwargs)
            legend = {"show": show, "kwargs": legend_kwargs}
        else:
            legend = {"show": show, "kwargs": kwargs["kwargs"]}
        self.legend = legend if show else None
        return self

    def set_zorder(self, zorder: Union[Sequence[float], float]):
        if isinstance(zorder, (float, int)):
            self.zorder = zorder
        elif isinstance(zorder, (list, tuple, ndarray, Series)) and all(
            isinstance(zo, (float, int)) for zo in zorder
        ):
            if len(zorder) != self.data_size:
                raise ArgumentStructureError(
                    "zorder must be of the same length as x_data and y_data, "
                    + f"len(zorder)={len(zorder)} != len(x_data)={self.data_size}."
                )
            self.zorder = zorder
        else:
            raise ArgumentTypeError("zorder must be a float or sequence of floats")
        return self

    def set_figsize(self, figsize: Tuple[float, float]):
        if isinstance(figsize, list):
            figsize = tuple(figsize)
        if isinstance(figsize, tuple) and all(
            isinstance(val, (float, int)) for val in figsize
        ):
            self.figsize = figsize
        else:
            raise ArgumentTypeError("figsize must be a tuple of floats or ints.")
        return self

    def set_scales(
        self,
        xscale: Scale = None,
        yscale: Scale = None,
    ):
        _scales = ["linear", "log", "symlog", "logit"]
        if xscale is not None:
            if xscale not in _scales:
                raise ArgumentValueError(f"'xscale'={xscale} must be one of {_scales}")
            self.xscale = xscale
        if yscale is not None:
            if yscale not in _scales:
                raise ArgumentValueError(
                    f"'x=yscale'={yscale} must be one of {_scales}"
                )
            self.yscale = yscale
        return self

    def set_grid(
        self,
        visible: bool = None,
        which: Literal["major", "minor", "both"] = "major",
        axis: Literal["both", "x", "y"] = "both",
        **kwargs,
    ):
        if visible not in [True, False]:
            raise ArgumentTypeError("visible must be a bool.")
        if which not in ["major", "minor", "both"]:
            raise ArgumentValueError("which must be one of 'major', 'minor', 'both'.")
        if axis not in ["both", "x", "y"]:
            raise ArgumentValueError("axis must be one of 'both', 'x', 'y'.")
        self.grid = dict(visible=visible, which=which, axis=axis, **kwargs)
        return self

    def set_texts(self, texts: Union[Sequence[Dict], Dict]):
        if isinstance(texts, dict):
            self.texts = [texts]
        elif isinstance(texts, (list, tuple, ndarray, Series)) and all(
            isinstance(tx, dict) for tx in texts
        ):
            self.texts = texts
        else:
            raise ArgumentTypeError(
                "texts must a matplotlib Text object or a sequence of Text objects"
            )
        return self

    def set_tight_layout(self, tight_layout: bool = False):
        if tight_layout in [True, False]:
            self.tight_layout = tight_layout
        else:
            raise ArgumentValueError("tight_layout must be a bool.")
        return self

    def set_background(self, background: Color):
        self.background = background
        return self

    def set_figure_background(self, figure_background: Color):
        self.figure_background = figure_background
        return self

    def set_suptitle(self, t: str, **kwargs):
        self.suptitle = dict(t=t, **kwargs)
        return self

    def set_ax_limits(
        self,
        xlim: Union[Tuple[float, float], None] = None,
        ylim: Union[Tuple[float, float], None] = None,
    ):
        if xlim is not None:
            if not isinstance(xlim, (list, tuple)) or len(xlim) != 2:
                raise ArgumentStructureError(
                    "xlim must be a tuple or list of two ints or floats (min, max)."
                )
            if not all(isinstance(x, (int, float, integer)) for x in xlim):
                raise ArgumentTypeError("xlim values must be ints or floats.")
            self.xlim = xlim
        if ylim is not None:
            if not isinstance(ylim, (list, tuple)) or len(ylim) != 2:
                raise ArgumentStructureError(
                    "ylim must be a tuple or list of two ints or floats (min, max)."
                )
            if not all(isinstance(y, (int, float, integer)) for y in ylim):
                raise ArgumentTypeError("ylim values must be ints or floats.")
            self.ylim = ylim
        return self

    def set_ticks(
        self,
        x_ticks: Union[Sequence[Union[float, int]], None] = None,
        y_ticks: Union[Sequence[Union[float, int]], None] = None,
    ):
        if x_ticks is not None:
            if not isinstance(x_ticks, (list, tuple, ndarray)):
                raise ArgumentTypeError("x_ticks must be array-like")
            if not all(isinstance(x, (int, float, integer)) for x in x_ticks):
                raise ArgumentTypeError("x_ticks values must be ints or floats.")
            self.x_ticks = x_ticks
        if y_ticks is not None:
            if not isinstance(y_ticks, (list, tuple, ndarray)):
                raise ArgumentTypeError("y_ticks must be array-like")
            if not all(isinstance(y, (int, float, integer)) for y in y_ticks):
                raise ArgumentTypeError("y_ticks values must be ints or floats.")
            self.y_ticks = y_ticks
        return self

    def set_tick_labels(
        self,
        x_tick_labels: Union[Sequence[Union[str, float, int]], None] = None,
        y_tick_labels: Union[Sequence[Union[str, float, int]], None] = None,
    ):
        if x_tick_labels is not None:
            if not isinstance(x_tick_labels, (list, tuple, ndarray)):
                raise ArgumentTypeError("x_tick_labels must be array-like")
            if not all(isinstance(x, (str, float, int)) for x in x_tick_labels):
                raise ArgumentTypeError(
                    "x_tick_labels values must be strings, floats, or ints"
                )
            self.x_tick_labels = x_tick_labels
        if y_tick_labels is not None:
            if not isinstance(y_tick_labels, (list, tuple, ndarray)):
                raise ArgumentTypeError("y_tick_labels must be array-like")
            if not all(isinstance(y, (str, float, int)) for y in y_tick_labels):
                raise ArgumentTypeError(
                    "y_tick_labels values must be strings, floats, or ints"
                )
            self.y_tick_labels = y_tick_labels
        return self

    def draw(self, show: bool = True):
        if self.figure is not None or self.ax is not None:
            self.clear()
        if self.style is not None:
            default_style = plt.rcParams.copy()
            plt.style.use(self.style)
        if not self.ax and not self.figure:
            self.figure, self.ax = plt.subplots(figsize=self.figsize)

        if self.grid is not None:
            self.ax.grid(**self.grid)

        scatter_kwargs = {
            "x": self.x_data,
            "y": self.y_data,
            "s": self.size,
            "c": self.color,
            "marker": self.marker,
            "cmap": self.colormap,
            "norm": self.normalization,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "alpha": self.alpha,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "label": self.label,
            "zorder": self.zorder,
            "plotnonfinite": self.plot_non_finite,
        }

        scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}

        try:
            self.ax.scatter(**scatter_kwargs)
        except Exception as e:
            raise PlotterException(f"Error while creating scatter plot: {e}")
        if self.title:
            self.ax.set_title(**self.title)
        if self.xlabel:
            self.ax.set_xlabel(**self.xlabel)
            if "color" in self.xlabel:
                self.ax.tick_params(axis="x", colors=self.xlabel["color"])
                self.ax.spines["bottom"].set_color(self.xlabel["color"])
                self.ax.spines["top"].set_color(self.xlabel["color"])
        if self.ylabel:
            self.ax.set_ylabel(**self.ylabel)
            if "color" in self.ylabel:
                self.ax.tick_params(axis="y", colors=self.ylabel["color"])
                self.ax.spines["left"].set_color(self.ylabel["color"])
                self.ax.spines["right"].set_color(self.ylabel["color"])
        if self.xscale:
            self.ax.set_xscale(self.xscale)
        if self.yscale:
            self.ax.set_yscale(self.yscale)
        if self.texts is not None:
            for text in self.texts:
                self.ax.text(**text)
        if self.legend is not None and self.legend["show"]:
            self.ax.legend(**self.legend["kwargs"])
        if self.background:
            self.ax.set_facecolor(self.background)
        if self.figure_background:
            self.figure.set_facecolor(self.figure_background)
        if self.suptitle:
            self.figure.suptitle(**self.suptitle)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        if self.x_ticks is not None:
            self.ax.set_xticks(self.x_ticks)
        if self.y_ticks is not None:
            self.ax.set_yticks(self.y_ticks)
        if self.x_tick_labels is not None:
            self.ax.set_xticklabels(self.x_tick_labels)
        if self.y_tick_labels is not None:
            self.ax.set_yticklabels(self.y_tick_labels)
        if self.hover and self.label is not None:
            pass
        if self.tight_layout:
            plt.tight_layout()
        if show:
            plt.show()
        if self.style is not None:
            plt.rcParams.update(default_style)
        return self.ax

    def save_plot(
        self,
        file_path: Path,
        dpi: int = 300,
        bbox_inches: str = "tight",
        draw: bool = False,
    ):
        if self.figure or draw:
            if self.figure is None and draw:
                self.draw()
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise PlotterException(f"Error while saving scatter plot: {e}")
        else:
            raise PlotterException("Plot not drawn yet. Call draw() before saving.")
        return self

    def clear(self):
        if self.figure or self.ax:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self

    def _to_serializable(self, value: Any) -> Any:
        if value is None:
            return None
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, ndarray):
            return value.tolist()
        elif isinstance(value, Series):
            return value.to_list()
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, Colormap):
            return value.name
        elif isinstance(value, Normalize):
            return value.__class__.__name__.lower()
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, MarkerStyle):
            marker = dict(
                marker=value.get_marker(),
                fillstyle=value.get_fillstyle(),
                capstyle=value.get_capstyle(),
                joinstyle=value.get_joinstyle(),
            )
            return marker

        return value

    def save_plotter(self, file_path: Union[str, Path], data: bool = False) -> None:
        init_params = {}
        for param, config in self._init_params.items():
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        if data:
            init_params["x_data"] = self._to_serializable(self.x_data)
            init_params["y_data"] = self._to_serializable(self.y_data)
        with open(file_path, "w") as f:
            json.dump(init_params, f, indent=4)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Plotter":
        with open(file_path, "r") as f:
            params = json.load(f)
        x_data = params.pop("x_data") if "x_data" in params else []
        y_data = params.pop("y_data") if "y_data" in params else []
        if "xlim" in params:
            params["xlim"] = tuple(params["xlim"])
        if "ylim" in params:
            params["ylim"] = tuple(params["ylim"])
        return cls(x_data=x_data, y_data=y_data, **params)
