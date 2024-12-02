import re
from typing import Any, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Color,
    EdgeColor,
    FaceColor,
    LineStyle,
    Markers,
    MarkerStyle,
    _colors,
)
from mitools.visuals.plots.plotter import Plotter


class LinePlotterException(Exception):
    pass


class LinePlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any, **kwargs):
        line_params = {
            "marker": {"default": None, "type": Markers},
            "markersize": {"default": None, "type": Union[Sequence[float], float]},
            "markeredgewidth": {"default": None, "type": Union[Sequence[float], float]},
            "markeredgecolor": {"default": None, "type": EdgeColor},
            "markerfacecolor": {"default": None, "type": FaceColor},
            "linestyle": {"default": "-", "type": LineStyle},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
        }
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(line_params)
        self._set_init_params(**kwargs)
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

    def set_color(
        self, color: Union[Sequence[Color], Color, Sequence[float], Sequence[int]]
    ):
        if isinstance(color, str):
            if color not in _colors and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", color
            ):
                raise ArgumentTypeError(
                    f"'color'='{color}' must be a valid Matplotlib color string or HEX code."
                )
            self.color = color
            return self
        if (
            isinstance(color, (tuple, list, ndarray))
            and len(color) in [3, 4]
            and all(isinstance(c, (float, int, integer)) for c in color)
        ):
            self.color = color
            return self
        if isinstance(color, (list, tuple, ndarray, Series)):
            if len(color) != self.data_size:
                raise ArgumentStructureError(
                    "color must be of the same length as x_data and y_data, "
                    + f"len(color)={len(color)} != len(x_data)={self.data_size}."
                )
            if not all(
                isinstance(c, str)
                or (
                    isinstance(c, (tuple, list, ndarray))
                    and len(c) in [3, 4]
                    and all(isinstance(x, (float, int, integer)) for x in c)
                )
                for c in color
            ):
                raise ArgumentTypeError(
                    "All elements in color must be strings or RGB/RGBA values."
                )
            self.color = color
            return self
        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_marker(self, marker: Union[Markers, str]):
        if isinstance(marker, str):
            if marker not in MarkerStyle.markers:
                raise ArgumentValueError(
                    f"'marker'={marker} must be a valid Matplotlib marker string"
                    + f" {MarkerStyle.markers}."
                )
            self.marker = marker
        elif isinstance(marker, Sequence):
            if len(marker) != self.data_size:
                raise ArgumentStructureError(
                    "marker sequence must be same length as data"
                )
            if not all(m in MarkerStyle.markers for m in marker):
                raise ArgumentValueError("Invalid marker in sequence")
            self.marker = marker
        else:
            raise ArgumentTypeError(
                "marker must be a string or sequence of valid Matplotlib marker strings"
            )
        return self

    def set_markersize(self, markersize: Union[float, Sequence[float]]):
        if isinstance(markersize, (int, float)):
            self.markersize = markersize
        elif isinstance(markersize, Sequence):
            if len(markersize) != self.data_size:
                raise ArgumentStructureError(
                    "markersize sequence must be same length as data"
                )
            self.markersize = markersize
        return self

    def set_markerfacecolor(self, markerfacecolor: Union[Color, Sequence[Color]]):
        self.markerfacecolor = markerfacecolor
        return self

    def set_markeredgecolor(self, markeredgecolor: Union[Color, Sequence[Color]]):
        self.markeredgecolor = markeredgecolor
        return self

    def set_markeredgewidth(self, markeredgewidth: Union[float, Sequence[float]]):
        if isinstance(markeredgewidth, (int, float)):
            self.markeredgewidth = markeredgewidth
        elif isinstance(markeredgewidth, Sequence):
            if len(markeredgewidth) != self.data_size:
                raise ArgumentStructureError(
                    "markeredgewidth sequence must be same length as data"
                )
            self.markeredgewidth = markeredgewidth
        return self

    def set_linestyle(self, linestyle: LineStyle):
        _valid_styles = ["-", "--", "-.", ":", "None", "none", " ", ""]
        if isinstance(linestyle, str):
            if linestyle not in _valid_styles:
                raise ArgumentValueError(f"linestyle must be one of {_valid_styles}")
            self.linestyle = linestyle
        elif isinstance(linestyle, (list, tuple)) and all(
            ls in _valid_styles for ls in linestyle
        ):
            if len(linestyle) != self.data_size:
                raise ArgumentStructureError(
                    "linestyle must be of the same length as x_data and y_data, "
                    + f"len(linestyle)={len(linestyle)} != len(x_data)={self.data_size}."
                )
            self.linestyle = linestyle
        else:
            raise ArgumentTypeError("Invalid linestyle format")
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

    def _create_plot(self, show: bool = True):
        plot_kwargs = {
            "color": self.color,
            "marker": self.marker,
            "markersize": self.markersize,
            "markerfacecolor": self.markerfacecolor,
            "markeredgecolor": self.markeredgecolor,
            "markeredgewidth": self.markeredgewidth,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "alpha": self.alpha,
            "label": self.label,
            "zorder": self.zorder,
        }
        plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
        try:
            self.ax.plot(self.x_data, self.y_data, **plot_kwargs)
        except Exception as e:
            raise LinePlotterException(f"Error while creating line plot: {e}")
