from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from numpy import ndarray
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
        self.title: Text = ""
        if "title" in kwargs:
            self.set_title(kwargs["title"])
        self.xlabel: Text = ""
        if "xlabel" in kwargs:
            self.set_xlabel(kwargs["xlabel"])
        self.ylabel: Text = ""
        if "ylabel" in kwargs:
            self.set_ylabel(kwargs["ylabel"])
        self.size: Union[Sequence[float], float] = None
        if "size" in kwargs:
            self.set_size(kwargs["size"])
        self.color: Union[Sequence[Color], Color] = None
        if "color" in kwargs:
            self.set_color(kwargs["color"])
        self.marker: Markers = "o"
        if "marker" in kwargs:
            self.set_marker(kwargs["marker"])
        self.color_map: Cmap = None
        if "color_map" in kwargs:
            self.set_colormap(kwargs["color_map"])
        self.normalization: Norm = None
        if "normalization" in kwargs:
            self.set_normalization(kwargs["normalization"])
        self.vmin: float = None
        if "vmin" in kwargs:
            self.set_vmin(kwargs["vmin"])
        self.vmax: float = None
        if "vmax" in kwargs:
            self.set_vmax("vmax")
        self.alpha: Union[Sequence[float], float] = 1.0
        if "alpha" in kwargs:
            self.set_alpha(kwargs["alpha"])
        self.linewidth: Union[Sequence[float], float] = None
        if "linewidth" in kwargs:
            self.set_linewidth(kwargs["linewidth"])
        self.edgecolor: EdgeColor = None
        if "edgecolor" in kwargs:
            self.set_edgecolor(kwargs["edgecolor"])
        self.facecolor: FaceColor = None
        if "facecolor" in kwargs:
            self.set_facecolor(kwargs["facecolor"])
        self.label: Union[Sequence[str], str] = None
        if "label" in kwargs:
            self.set_label(kwargs["label"])
        self.zorder: Union[Sequence[float], float] = None
        if "zorder" in kwargs:
            self.set_zorder(kwargs["zorder"])
        self.plot_non_finite: bool = False
        if "plot_non_finite" in kwargs:
            self.set_plot_non_finite(kwargs["plot_non_finite"])
        self.figsize: Tuple[float, float] = (21, 14)
        if "figsize" in kwargs:
            self.set_figsize(kwargs["figsize"])
        self.style: str = "dark_background"
        if "style" in kwargs:
            self.set_style(kwargs["style"])
        self.grid: Dict[str, Any] = None
        if "grid" in kwargs:
            self.set_grid(kwargs["grid"])
        self.hover: bool = False
        if "hover" in kwargs:
            self.set_hover(kwargs["hover"])
        self.tight_layout: bool = False
        if "tight_layout" in kwargs:
            self.set_tight_layout(kwargs["tight_layout"])
        self.texts: Union[Sequence[Text], Text] = None
        if "texts" in kwargs:
            self.set_texts(kwargs["texts"])
        self.xscale: Scale = None
        if "xscale" in kwargs:
            self.set_scales(xscale=kwargs["xscale"])
        self.yscale: Scale = None
        if "yscale" in kwargs:
            self.set_scales(yscale=kwargs["yscale"])
        self.figure: Figure = None
        self.ax: Axes = None

    def _validate_data(self, data: Any, name: str) -> Any:
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
        if style in plt.style.available:
            self.style = style
        else:
            raise ArgumentValueError(f"Style '{style}' is not available in Matplotlib.")
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

    def set_color(self, color: Union[Sequence[Color], Color]):
        if isinstance(color, (list, tuple, ndarray, Series, str)):
            if not isinstance(color, str):
                if len(color) != self.data_size:
                    raise ArgumentStructureError(
                        "color must be of the same length as x_data and y_data, "
                        + f"len(color)={len(color)} != len(x_data)={self.data_size}."
                    )
                if not all(isinstance(c, (str, tuple, list, ndarray)) for c in color):
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

    def set_vmin(self, vmin: float):
        if self.normalization is not None and not isinstance(self.normalization, str):
            raise ArgumentValueError(
                f"Normalization {self.normalization} has been set. vmin only work when 'self.normalization' is a str."
            )
        if isinstance(vmin, float):
            self.vmin = vmin
        else:
            raise ArgumentTypeError(f"'vmin'={vmin} must be a float.")
        return self

    def set_vmax(self, vmax: float):
        if self.normalization is not None and not isinstance(self.normalization, str):
            raise ArgumentValueError(
                f"Normalization {self.normalization} has been set. vmax only work when 'self.normalization' is a str."
            )
        if isinstance(vmax, float):
            self.vmin = vmax
        else:
            raise ArgumentTypeError(f"'vmax'={vmax} must be a float.")
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
        if isinstance(linewidth, float):
            self.linewidth = linewidth
        elif isinstance(linewidth, list, tuple, ndarray, Series) and all(
            isinstance(lw, float) for lw in linewidth
        ):
            if len(linewidth) != self.data_size:
                raise ArgumentStructureError(
                    "linewidth must be of the same length as x_data and y_data, "
                    + f"len(linewdith)={len(linewidth)} != len(x_data)={self.data_size}."
                )
            self.linewidth = linewidth
        else:
            raise ArgumentTypeError(
                "linewidth must be a float or an array-like of floats."
            )
        return self

    def set_edgecolor(self, edgecolor: EdgeColor):
        if edgecolor in ["face", "none", None]:
            self.edgecolor = edgecolor
        elif isinstance(edgecolor, str) or (
            isinstance(edgecolor, (list, tuple))
            and len(edgecolor) in [3, 4]
            and all(isinstance(x, (int, float)) for x in edgecolor)
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
                        and all(isinstance(x, (int, float)) for x in ec)
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
            and all(isinstance(x, (int, float)) for x in facecolor)
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
                        and all(isinstance(x, (int, float)) for x in fc)
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
            self.labels = labels
        elif isinstance(labels, (list, tuple, ndarray, Series)) and all(
            isinstance(label, str) for label in labels
        ):
            if len(labels) != len(self.data_size):
                raise ArgumentStructureError(
                    "labels must be of the same length as x_data and y_data, "
                    + f"len(labels)={len(labels)} != len(x_data)={self.data_size}."
                )
            self.labels = labels
        else:
            raise ArgumentTypeError("labels must be a str or a sequence of strs.")
        return self

    def set_zorder(self, zorder: Union[Sequence[float], float]):
        if isinstance(zorder, float):
            self.zorder = zorder
        elif isinstance(
            zorder,
            (list, tuple, ndarray, Series)
            and all(isinstance(zo, float) for zo in zorder),
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
            isinstance(val, float) for val in figsize
        ):
            self.figsize = figsize
        else:
            raise ArgumentTypeError("figsize must be a tuple of floats.")
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

    def set_grid(
        self,
        visible: bool = None,
        which: Literal["major", "minor", "both"] = None,
        axis: Literal["both", "x", "y"] = None,
        **kwargs,
    ):
        self.grid = dict(visible=visible, which=which, axis=axis, **kwargs)

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
            raise ArgumentValueError(f"tight_layout must be a bool.")
        return self

    def set_limits(self, xlim=None, ylim=None):
        raise NotImplementedError

    def set_ticks(self, x_ticks=None, y_ticks=None):
        raise NotImplementedError

    def set_tick_labels(self, x_tick_labels=None, y_tick_labels=None):
        raise NotImplementedError

    def set_legend(self, legend=True):
        raise NotImplementedError

    def add_line(self, x_data, y_data, **kwargs):
        raise NotImplementedError

    def draw(self, show: bool = False):
        if self.style is not None:
            plt.style.use(self.style)
        if not self.ax:
            self.figure, self.ax = plt.subplots(figsize=self.figsize)

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
        if self.ylabel:
            self.ax.set_ylabel(**self.ylabel)
        if self.xscale:
            self.ax.set_xscale(self.xscale)
        if self.yscale:
            self.ax.set_yscale(self.yscale)
        if self.grid is not None:
            self.ax.grid(**self.grid)
        if self.texts is not None:
            for text in self.texts:
                self.ax.text(**text)
        if self.hover and self.label is not None:
            pass
        if self.tight_layout:
            plt.tight_layout()
        if show:
            plt.show()
        return self.ax

    def save(self, file_path: Path, dpi: int = 300, bbox_inches: str = "tight"):
        if self.figure:
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                self.logger.error(f"Error saving figure: {e}")
                raise
        else:
            raise RuntimeError("Plot not drawn yet. Call draw() before saving.")
        return self

    def clear(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self
