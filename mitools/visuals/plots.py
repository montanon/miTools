from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

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

Color = Union[str, Sequence[float]]
Marker = Union[str, int, Path, MarkerStyle]
Markers = Union[Marker, Sequence[Marker]]
Cmap = Union[
    Literal[
        "magma",
        "inferno",
        "plasma",
        "viridis",
        "cividis",
        "twilight",
        "twilight_shifted",
        "turbo",
    ],
    Colormap,
]
Norm = Union[str, Normalize]
EdgeColor = Union[Literal["face", "none", None], Color, Sequence[Color]]
FaceColor = Union[Color, Sequence[Color]]
LineStyle = Literal["solid", "dashed", "dashdot", "dotted", "-", "--", "-.", ":"]
Scale = Literal["linear", "log", "symlog", "logit"]


class ScatterPlotterException(Exception):
    pass


class ScatterPlotter:
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
            "size": {"default": None, "type": Union[Sequence[float], float]},
            "color": {"default": None, "type": Union[Sequence[Color], Color]},
            "marker": {"default": "o", "type": Markers},
            "colormap": {"default": None, "type": Cmap},
            "normalization": {"default": None, "type": Norm},
            "vmin": {"default": None, "type": float},
            "vmax": {"default": None, "type": float},
            "alpha": {"default": 1.0, "type": Union[Sequence[float], float]},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
            "edgecolor": {"default": None, "type": EdgeColor},
            "facecolor": {"default": None, "type": FaceColor},
            "label": {"default": None, "type": Union[Sequence[str], str]},
            "zorder": {"default": None, "type": Union[Sequence[float], float]},
            "plot_non_finite": {"default": False, "type": bool},
            "figsize": {"default": (21, 14), "type": Tuple[float, float]},
            "style": {"default": "dark_background", "type": str},
            "grid": {"default": None, "type": Dict[str, Any]},
            "hover": {"default": False, "type": bool},
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
            if param in kwargs:
                setter_name = f"set_{param}"
                if hasattr(self, setter_name):
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
        self.legend: Union[Dict, None] = None

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

    def set_title(self, title: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html"""
        self.title = dict(label=title, **kwargs)
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

    def set_size(self, size_data: Union[Sequence[float], float]):
        if isinstance(size_data, (list, tuple, ndarray, Series, float, int)):
            if not isinstance(size_data, (float, int)):
                if len(size_data) != self.data_size:
                    raise ArgumentStructureError(
                        "size_data must be of the same length as x_data and y_data,"
                        + f"len(size_data)={len(size_data)} != len(x_data)={self.data_size}."
                    )
                if not all(isinstance(s, (float, int)) for s in size_data):
                    raise ArgumentTypeError(
                        "All elements in size_data must be numeric."
                    )
            self.size = size_data
        else:
            raise ArgumentTypeError(
                "size_data must be array-like or a single numeric value."
            )
        return self

    def set_color(
        self, color: Union[Sequence[Color], Color, Sequence[float], Sequence[int]]
    ):
        if isinstance(color, (list, tuple, ndarray, Series, str)):
            if not isinstance(color, str):
                if len(color) != self.data_size:
                    raise ArgumentStructureError(
                        "color must be of the same length as x_data and y_data, "
                        + f"len(color)={len(color)} != len(x_data)={self.data_size}."
                    )
                if not all(
                    isinstance(c, (str, tuple, list, ndarray, float, int))
                    for c in color
                ):
                    raise ArgumentTypeError(
                        "All elements in color must be strings, tuples, lists, or ndarrays."
                    )
            self.color = color
        else:
            raise ArgumentTypeError(
                "color must be a string, array-like of strings, or array-like of RGB/RGBA values."
            )
        return self

    def set_marker(
        self,
        marker: Markers,
        fillstyle: Literal["full", "left", "right", "bottom", "top", "none"] = None,
    ):
        if fillstyle is not None and fillstyle not in MarkerStyle.fillstyles:
            raise ArgumentValueError(
                f"'fillstyle'={fillstyle} must be a valid Matplotlib fillstyle string {MarkerStyle.fillstyles}."
            )
        if isinstance(marker, str):
            if marker not in MarkerStyle.markers:
                raise ArgumentValueError(
                    f"'marker'={marker} must be a valid Matplotlib marker string"
                    + f" {MarkerStyle.markers} or a MarkerStyle object."
                )
            self.marker = MarkerStyle(marker, fillstyle=fillstyle)
        elif isinstance(marker, MarkerStyle):
            self.marker = MarkerStyle(marker.get_marker(), fillstyle=fillstyle)
        elif isinstance(marker, Sequence):
            if not all(isinstance(m, (MarkerStyle, str)) for m in marker):
                raise ArgumentTypeError(
                    "All elements in marker must be MarkerStyle objects or valid Matplotlib marker strings."
                )
            if len(marker) != self.data_size:
                raise ArgumentStructureError(
                    "marker must be of the same length as x_data and y_data, "
                    + f"len(marker)={len(marker)} != len(x_data)={self.data_size}."
                )
            self.marker = [
                MarkerStyle(m.get_marker(), fillstyle=fillstyle)
                if isinstance(m, MarkerStyle)
                else MarkerStyle(m, fillstyle=fillstyle)
                for m in marker
            ]
        else:
            raise ArgumentTypeError(
                "marker must be a string, a MarkerStyle object, or a sequence of MarkerStyle objects."
            )
        return self

    def set_colormap(self, cmap: Cmap):
        _cmaps = [
            "magma",
            "inferno",
            "plasma",
            "viridis",
            "cividis",
            "twilight",
            "twilight_shifted",
            "turbo",
        ]
        if isinstance(cmap, str) and cmap in _cmaps or isinstance(cmap, Colormap):
            self.color_map = cmap
        else:
            raise ArgumentTypeError(
                f"cmap must be a Colormap object or a valid Colormap string from {_cmaps}."
            )
        return self

    def set_normalization(self, normalization: Norm):
        _normalizations = [
            "linear",
            "log",
            "symlog",
            "asinh",
            "logit",
            "function",
            "functionlog",
        ]
        if (
            isinstance(normalization, Normalize)
            or isinstance(normalization, str)
            and normalization in _normalizations
        ):
            self.normalization = normalization
        else:
            raise ArgumentTypeError(
                "normalize must be a valid Normalize object or a valid Matplotlib normalization string"
                + f" of {_normalizations}."
            )
        return self

    def set_vmin(self, vmin: Union[float, int]):
        if self.normalization is not None and not isinstance(self.normalization, str):
            raise ArgumentValueError(
                f"Normalization {self.normalization} has been set. vmin only work when 'self.normalization' is a str."
            )
        if isinstance(vmin, (float, int)):
            self.vmin = vmin
        return self

    def set_vmax(self, vmax: Union[float, int]):
        if self.normalization is not None and not isinstance(self.normalization, str):
            raise ArgumentValueError(
                f"Normalization {self.normalization} has been set. vmax only work when 'self.normalization' is a str."
            )
        if isinstance(vmax, (float, int)):
            self.vmax = vmax
        return self

    def set_normalization_range(self, vmin: float, vmax: float):
        self.set_vmin(vmin)
        self.set_vmax(vmax)
        return self

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

    def set_linewidth(self, linewidth: Union[Sequence[float], float]):
        if isinstance(linewidth, (float, int)):
            self.linewidth = linewidth
        elif isinstance(linewidth, (list, tuple, ndarray, Series)) and all(
            isinstance(lw, (float, int)) for lw in linewidth
        ):
            if len(linewidth) != self.data_size:
                raise ArgumentStructureError(
                    "linewidth must be of the same length as x_data and y_data, "
                    + f"len(linewdith)={len(linewidth)} != len(x_data)={self.data_size}."
                )
            self.linewidth = linewidth
        else:
            raise ArgumentTypeError(
                "linewidth must be a float or an array-like of floats or ints."
            )
        return self

    def set_edgecolor(self, edgecolor: EdgeColor):
        if edgecolor in ["face", "none", None]:
            self.edgecolor = edgecolor
        elif isinstance(edgecolor, str) or (
            isinstance(edgecolor, (list, tuple))
            and len(edgecolor) in [3, 4]
            and all(isinstance(x, (int, float, integer)) for x in edgecolor)
        ):
            self.edgecolor = edgecolor
        elif isinstance(edgecolor, (list, tuple, ndarray, Series)):
            if len(edgecolor) != self.data_size:
                raise ArgumentStructureError(
                    "edgecolor must be of the same length as x_data and y_data, "
                    + f"len(edgecolor)={len(edgecolor)} != len(x_data)={self.data_size}."
                )
            for ec in edgecolor:
                if not (
                    isinstance(ec, str)
                    or (
                        isinstance(ec, (list, tuple))
                        and len(ec) in [3, 4]
                        and all(isinstance(x, (int, float, integer)) for x in ec)
                    )
                ):
                    raise ArgumentTypeError(
                        "Each edgecolor must be a string or RGB/RGBA values."
                    )
            self.edgecolor = edgecolor
        else:
            raise ArgumentTypeError(
                "edgecolor must be 'face', 'none', None, a color string, RGB/RGBA values, "
                + "or an array-like of color strings/RGB/RGBA values."
            )
        return self

    def set_facecolor(self, facecolor: FaceColor):
        if isinstance(facecolor, str) or (
            isinstance(facecolor, (list, tuple))
            and len(facecolor) in [3, 4]
            and all(isinstance(x, (int, float, integer)) for x in facecolor)
        ):
            self.facecolor = facecolor
        elif isinstance(facecolor, (list, tuple, ndarray, Series)):
            if len(facecolor) != self.data_size:
                raise ArgumentStructureError(
                    "facecolor must be of the same length as x_data and y_data, "
                    + f"len(facecolor)={len(facecolor)} != len(x_data)={self.data_size}."
                )
            for fc in facecolor:
                if not (
                    isinstance(fc, str)
                    or (
                        isinstance(fc, (list, tuple))
                        and len(fc) in [3, 4]
                        and all(isinstance(x, (int, float, integer)) for x in fc)
                    )
                ):
                    raise ArgumentTypeError(
                        "Each facecolor must be a string or RGB/RGBA values."
                    )
            self.facecolor = facecolor
        else:
            raise ArgumentTypeError(
                f"'facecolor'={facecolor} must be a color string, RGB/RGBA values, "
                + "or an array-like of color strings/RGB/RGBA values."
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
                isinstance(l, str) for l in labels
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
                raise ArgumentTypeError("'title_fontsize' must be an integer or string")
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
        self.legend = {"show": show, "kwargs": legend_kwargs} if show else None
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

    def set_plot_non_finite(self, plot_non_finite: bool):
        if plot_non_finite not in [True, False]:
            raise ArgumentTypeError("plot_non_finite must be a bool.")
        self.plot_non_finite = plot_non_finite
        return self

    def set_figsize(self, figsize: Tuple[float, float]):
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

    def set_hover(self, hover: bool):
        if hover not in [True, False]:
            raise ArgumentTypeError("hover must be a bool.")
        self.hover = hover
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

    def set_suptitle(self, suptitle: str, **kwargs):
        self.suptitle = dict(t=suptitle, **kwargs)
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

    def draw(self, show: bool = False):
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
            "cmap": self.color_map,
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

        try:
            self.ax.scatter(**scatter_kwargs)
        except Exception as e:
            raise ScatterPlotterException(f"Error while creating scatter plot: {e}")
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

    def save(self, file_path: Path, dpi: int = 300, bbox_inches: str = "tight"):
        if self.figure:
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise ScatterPlotterException(f"Error while saving scatter plot: {e}")
        else:
            raise ScatterPlotterException(
                "Plot not drawn yet. Call draw() before saving."
            )
        return self

    def clear(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self
