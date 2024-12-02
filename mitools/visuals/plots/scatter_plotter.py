import json
import re
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
    Cmap,
    Color,
    EdgeColor,
    FaceColor,
    Markers,
    Norm,
    _colors,
)
from mitools.visuals.plots.plotter import Plotter


class ScatterPlotterException(Exception):
    pass


class ScatterPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any, **kwargs):
        scatter_params = {
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
        self._init_params.update(scatter_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_size(self, size_data: Union[Sequence[float], float]):
        if isinstance(size_data, (list, tuple, ndarray, Series, float, int, integer)):
            if not isinstance(size_data, (float, int, integer)):
                if len(size_data) != self.data_size:
                    raise ArgumentStructureError(
                        "size_data must be of the same length as x_data and y_data,"
                        + f"len(size_data)={len(size_data)} != len(x_data)={self.data_size}."
                    )
                if not all(isinstance(s, (float, int, integer)) for s in size_data):
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
        if fillstyle is not None and fillstyle not in MarkerStyle.fillstyles:
            raise ArgumentValueError(
                f"'fillstyle'={fillstyle} must be a valid Matplotlib fillstyle string {MarkerStyle.fillstyles}."
            )
        if isinstance(marker, dict):
            marker = MarkerStyle(**marker)
        if isinstance(marker, str):
            if marker not in MarkerStyle.markers:
                raise ArgumentValueError(
                    f"'marker'={marker} must be a valid Matplotlib marker string"
                    + f" {MarkerStyle.markers} or a MarkerStyle object."
                )
            self.marker = MarkerStyle(marker, fillstyle=fillstyle, **kwargs)
        elif isinstance(marker, MarkerStyle):
            self.marker = MarkerStyle(
                marker.get_marker(), fillstyle=fillstyle, **kwargs
            )
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
        if isinstance(cmap, str) and cmap in _cmaps or isinstance(cmap, Colormap):
            self.colormap = cmap
        else:
            raise ArgumentTypeError(
                f"'cmap'={cmap} must be a Colormap object or a valid Colormap string from {_cmaps}."
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

    def set_plot_non_finite(self, plot_non_finite: bool):
        if plot_non_finite not in [True, False]:
            raise ArgumentTypeError("plot_non_finite must be a bool.")
        self.plot_non_finite = plot_non_finite
        return self

    def set_hover(self, hover: bool):
        if hover not in [True, False]:
            raise ArgumentTypeError("hover must be a bool.")
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
