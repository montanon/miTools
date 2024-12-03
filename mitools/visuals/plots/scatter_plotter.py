import re
from typing import Any, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Cmap,
    Color,
    EdgeColor,
    FaceColor,
    Markers,
    Norm,
    _colors,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_sequence,
    validate_length,
    validate_same_length,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
)


class ScatterPlotterException(Exception):
    pass


class ScatterPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any, **kwargs):
        self._scatter_params = {
            "size": {"default": None, "type": Union[Sequence[float], float]},
            "marker": {"default": "o", "type": Markers},
            "colormap": {"default": None, "type": Cmap},
            "normalization": {"default": None, "type": Norm},
            "vmin": {"default": None, "type": float},
            "vmax": {"default": None, "type": float},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
            "edgecolor": {"default": None, "type": EdgeColor},
            "facecolor": {"default": None, "type": FaceColor},
            "plot_non_finite": {"default": False, "type": bool},
            "hover": {"default": False, "type": bool},
        }
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(self._scatter_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_size(self, size_data: Union[Sequence[float], float]):
        validate_type(size_data, (*SEQUENCE_TYPES, *NUMERIC_TYPES), "size_data")
        if is_sequence(size_data):
            validate_sequence_length(size_data, self.data_size, "size_data")
            validate_sequence_type(size_data, NUMERIC_TYPES, "size_data")
        self.size = size_data
        return self

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
        if isinstance(color, (tuple, list, ndarray)):
            if len(color) in [3, 4]:
                validate_sequence_type(color, NUMERIC_TYPES, "color")
                self.color = color
                return self
        if is_sequence(color):
            validate_length(color, self.data_size, "color")
            if not all(
                isinstance(c, str)
                or (
                    isinstance(c, (tuple, list, ndarray))
                    and len(c) in [3, 4]
                    and all(isinstance(x, (float, int, integer)) for x in c)
                )
                or isinstance(c, (int, float, integer))
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

    def set_marker(
        self,
        marker: Union[Markers, dict],
        fillstyle: Literal["full", "left", "right", "bottom", "top", "none"] = None,
        **kwargs,
    ):
        if fillstyle is not None:
            validate_value_in_options(fillstyle, MarkerStyle.fillstyles, "fillstyle")
        if isinstance(marker, dict):
            marker = MarkerStyle(**marker)
        if isinstance(marker, str):
            validate_value_in_options(marker, MarkerStyle.markers, "marker")
            self.marker = MarkerStyle(marker, fillstyle=fillstyle, **kwargs)
        elif isinstance(marker, MarkerStyle):
            self.marker = MarkerStyle(
                marker.get_marker(), fillstyle=fillstyle, **kwargs
            )
        elif is_sequence(marker):
            validate_sequence_type(marker, (MarkerStyle, str), "marker")
            validate_length(marker, self.data_size, "marker")
            self.marker = [
                MarkerStyle(m.get_marker(), fillstyle=fillstyle, **kwargs)
                if isinstance(m, MarkerStyle)
                else MarkerStyle(m, fillstyle=fillstyle, **kwargs)
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
        validate_type(cmap, (str, Colormap), "cmap")
        if isinstance(cmap, str):
            validate_value_in_options(cmap, _cmaps, "cmap")
        self.colormap = cmap
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
        validate_type(normalization, (str, Normalize), "normalization")
        if isinstance(normalization, str):
            validate_value_in_options(normalization, _normalizations, "normalization")
        self.normalization = normalization
        return self

    def set_vmin(self, vmin: Union[float, int]):
        if self.normalization is not None:
            validate_type(self.normalization, str, "normalization")
        validate_type(vmin, NUMERIC_TYPES, "vmin")
        self.vmin = vmin
        return self

    def set_vmax(self, vmax: Union[float, int]):
        if self.normalization is not None:
            validate_type(self.normalization, str, "normalization")
        validate_type(vmax, NUMERIC_TYPES, "vmax")
        self.vmax = vmax
        return self

    def set_normalization_range(self, vmin: float, vmax: float):
        self.set_vmin(vmin)
        self.set_vmax(vmax)
        return self

    def set_linewidth(self, linewidth: Union[Sequence[float], float]):
        if is_sequence(linewidth):
            validate_type(linewidth, SEQUENCE_TYPES, "linewidth")
            validate_sequence_type(linewidth, NUMERIC_TYPES, "linewidth")
            validate_same_length(linewidth, self.x_data, "linewidth", "x_data")
        else:
            validate_type(linewidth, NUMERIC_TYPES, "linewidth")
        self.linewidth = linewidth
        return self

    def set_edgecolor(self, edgecolor: EdgeColor):
        if edgecolor in ["face", "none", None]:
            self.edgecolor = edgecolor
            return self
        if isinstance(edgecolor, str):
            self.edgecolor = edgecolor
            return self
        if isinstance(edgecolor, (list, tuple)):
            validate_sequence_length(edgecolor, (3, 4), "edgecolor")
            validate_sequence_type(edgecolor, NUMERIC_TYPES, "edgecolor")
            self.edgecolor = edgecolor
            return self
        validate_type(edgecolor, SEQUENCE_TYPES, "edgecolor")
        validate_same_length(edgecolor, self.x_data, "edgecolor", "x_data")
        for ec in edgecolor:
            if isinstance(ec, str):
                continue
            if isinstance(ec, (list, tuple)):
                validate_sequence_length(ec, (3, 4), "edgecolor values")
                validate_sequence_type(ec, NUMERIC_TYPES, "edgecolor values")
                continue
            raise ArgumentTypeError(
                "Each edgecolor must be a string or RGB/RGBA values."
            )
        self.edgecolor = edgecolor
        return self

    def set_facecolor(self, facecolor: FaceColor):
        if isinstance(facecolor, str):
            self.facecolor = facecolor
            return self
        if isinstance(facecolor, (list, tuple)):
            validate_sequence_length(facecolor, (3, 4), "facecolor")
            validate_sequence_type(facecolor, NUMERIC_TYPES, "facecolor")
            self.facecolor = facecolor
            return self
        validate_type(facecolor, SEQUENCE_TYPES, "facecolor")
        validate_same_length(facecolor, self.x_data, "facecolor", "x_data")
        for fc in facecolor:
            if isinstance(fc, str):
                continue
            if isinstance(fc, (list, tuple)):
                validate_sequence_length(fc, (3, 4), "facecolor values")
                validate_sequence_type(fc, NUMERIC_TYPES, "facecolor values")
                continue
            raise ArgumentTypeError(
                "Each facecolor must be a string or RGB/RGBA values."
            )
        self.facecolor = facecolor
        return self

    def set_plot_non_finite(self, plot_non_finite: bool):
        validate_type(plot_non_finite, bool, "plot_non_finite")
        self.plot_non_finite = plot_non_finite
        return self

    def set_hover(self, hover: bool):
        validate_type(hover, bool, "hover")
        self.hover = hover
        return self

    def _create_plot(self):
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
            raise ScatterPlotterException(f"Error while creating scatter plot: {e}")
        if self.hover and self.label is not None:
            pass
