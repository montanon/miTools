from pathlib import Path
from typing import Any, Literal, Sequence, Tuple, Union

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
        self.size_data: Union[Sequence[float], float] = None
        if "size_data" in kwargs:
            self.set_size(kwargs["size_data"])
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
        self.linestyle: Union[Sequence[LineStyle], LineStyle] = None
        self.edgecolor: EdgeColor = None
        self.facecolor: FaceColor = None
        self.plot_non_finite: bool = False
        self.label: Union[Sequence[str], str] = None
        self.zorder: Union[Sequence[float], float] = None
        self.figsize: Tuple[float, float] = (21, 14)
        self.style: str = "dark_background"
        self.hover: bool = False
        self.figure: Figure = None
        self.ax: Axes = None

    def _validate_data(self, data: Any, name: str) -> Any:
        return data

    def set_title(self, title: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html"""
        self.title = Text(text=title, **kwargs)
        return self

    def set_xlabel(self, xlabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel"""
        self.xlabel = Text(text=xlabel, **kwargs)
        return self

    def set_ylabel(self, ylabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel"""
        self.ylabel = Text(text=ylabel, **kwargs)
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
            self.size_data = size_data
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

    def set_labels(self, labels):
        self.label = self._validate_data(labels, "labels")
        if len(self.label) != len(self.x_data):
            raise ValueError("labels must be of the same length as x_data and y_data.")
        return self

    def enable_hover(self, hover=True):
        self.hover = hover
        return self

    def set_color_data(self, color_data):
        self.color_data = self._validate_data(color_data, "color_data")
        if len(self.color_data) != len(self.x_data):
            raise ValueError(
                "color_data must be of the same length as x_data and y_data."
            )
        return self

    def set_size_data(self, size_data):
        return self.set_size(size_data)

    def set_figure_size(self, width, height):
        if self.figure:
            self.figure.set_size_inches(width, height)
        else:
            self.figure = plt.figure(figsize=(width, height))
        return self

    def set_limits(self, xlim=None, ylim=None):
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        return self

    def set_ticks(self, x_ticks=None, y_ticks=None):
        if x_ticks is not None:
            self.ax.set_xticks(x_ticks)
        if y_ticks is not None:
            self.ax.set_yticks(y_ticks)
        return self

    def set_tick_labels(self, x_tick_labels=None, y_tick_labels=None):
        if x_tick_labels is not None:
            self.ax.set_xticklabels(x_tick_labels)
        if y_tick_labels is not None:
            self.ax.set_yticklabels(y_tick_labels)
        return self

    def set_grid(self, grid=True):
        self.ax.grid(grid)
        return self

    def set_legend(self, legend=True):
        if legend:
            self.ax.legend()
        return self

    def add_text(self, x, y, text, **kwargs):
        self.ax.text(x, y, text, **kwargs)
        return self

    def add_line(self, x_data, y_data, **kwargs):
        self.ax.plot(x_data, y_data, **kwargs)
        return self

    def set_log_scale(self, x_log=False, y_log=False):
        if x_log:
            self.ax.set_xscale("log")
        if y_log:
            self.ax.set_yscale("log")
        return self

    def invert_axes(self, x_invert=False, y_invert=False):
        if x_invert:
            self.ax.invert_xaxis()
        if y_invert:
            self.ax.invert_yaxis()
        return self

    def set_aspect_ratio(self, aspect="auto"):
        self.ax.set_aspect(aspect)
        return self

    def apply_theme(self, theme):
        if theme == "dark":
            self.set_style("dark_background")
            self.set_color("cyan")
            self.set_edgecolor("white")
        elif theme == "light":
            self.set_style("default")
            self.set_color("blue")
            self.set_edgecolor("black")
        else:
            raise ValueError(f"Theme '{theme}' is not recognized.")
        return self

    def draw(self):
        plt.style.use(self.style)
        if not self.figure or not self.ax:
            self.figure, self.ax = plt.subplots()

        scatter_kwargs = {
            "x": self.x_data,
            "y": self.y_data,
            "c": self.color_data if self.color_data is not None else self.color,
            "cmap": self.color_map,
            "alpha": self.alpha,
            "edgecolor": self.edgecolor,
            "marker": self.marker,
        }

        if self.size_data is not None:
            scatter_kwargs["s"] = self.size_data

        try:
            sc = self.ax.scatter(**scatter_kwargs)
        except Exception as e:
            self.logger.error(f"Error while creating scatter plot: {e}")
            raise

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        if self.color_data is not None and self.color_map is not None:
            cbar = self.figure.colorbar(sc, ax=self.ax)
            cbar.set_label("Color Scale")

        if self.hover and self.label is not None:
            # Implement hover functionality
            annot = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(False)

            def update_annot(ind):
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = "{}".format(" ".join([str(self.label[n]) for n in ind["ind"]]))
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.4)

            def hover_event(event):
                vis = annot.get_visible()
                if event.inaxes == self.ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        self.figure.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            self.figure.canvas.draw_idle()

            self.figure.canvas.mpl_connect("motion_notify_event", hover_event)

        plt.show()
        return self

    def save(self, filename, dpi=300, bbox_inches="tight"):
        if self.figure:
            try:
                self.figure.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
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
