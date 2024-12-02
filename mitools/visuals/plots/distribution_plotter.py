import re
from typing import Any, Dict, Literal, Sequence, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import Color, EdgeColor, FaceColor, _colors
from mitools.visuals.plots.plotter import Plotter


class DistributionPlotterException(Exception):
    pass


class DistributionPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any = None, **kwargs):
        super().__init__(
            x_data=x_data, y_data=x_data if y_data is None else y_data, **kwargs
        )
        self._hist_params = {
            "bins": {"default": "auto", "type": Union[int, str, Sequence[float]]},
            "range": {"default": None, "type": Union[Tuple[float, float], None]},
            "density": {"default": False, "type": bool},
            "weights": {"default": None, "type": Union[Sequence[float], None]},
            "cumulative": {"default": False, "type": bool},
            "bottom": {"default": None, "type": Union[Sequence[float], float, None]},
            "histtype": {
                "default": "bar",
                "type": Literal["bar", "barstacked", "step", "stepfilled"],
            },
            "align": {"default": "mid", "type": Literal["left", "mid", "right"]},
            "orientation": {
                "default": "vertical",
                "type": Literal["horizontal", "vertical"],
            },
            "rwidth": {"default": None, "type": Union[float, None]},
            "log": {"default": False, "type": bool},
            "stacked": {"default": False, "type": bool},
            "edgecolor": {"default": None, "type": EdgeColor},
            "facecolor": {"default": None, "type": FaceColor},
            "fill": {"default": True, "type": bool},
            "linestyle": {"default": "-", "type": str},
            "linewidth": {"default": None, "type": Union[float, None]},
            "hatch": {"default": None, "type": Union[Sequence[str], str]},
        }
        self._init_params.update(self._hist_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

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

    def set_bins(self, bins: Union[int, str, Sequence[float]]):
        if isinstance(bins, (int, str)):
            if isinstance(bins, int) and bins <= 0:
                raise ArgumentValueError("Number of bins must be positive")
            if isinstance(bins, str) and bins not in [
                "auto",
                "fd",
                "doane",
                "scott",
                "stone",
                "rice",
                "sturges",
                "sqrt",
            ]:
                raise ArgumentValueError("Invalid string value for bins")
            self.bins = bins
        elif isinstance(bins, (list, tuple, ndarray)) and all(
            isinstance(b, (float, int, integer)) for b in bins
        ):
            self.bins = bins
        else:
            raise ArgumentTypeError(
                "bins must be an integer, a string, or a sequence of numbers"
            )
        return self

    def set_range(self, range: Tuple[float, float]):
        if not isinstance(range, (tuple, list)) or len(range) != 2:
            raise ArgumentStructureError("range must be a tuple of (min, max)")
        if not all(isinstance(x, (int, float, integer)) for x in range):
            raise ArgumentTypeError("range values must be numbers")
        if range[0] >= range[1]:
            raise ArgumentValueError("range[0] must be less than range[1]")
        self.range = range
        return self

    def set_density(self, density: bool):
        if not isinstance(density, bool):
            raise ArgumentTypeError("density must be a boolean")
        self.density = density
        return self

    def set_weights(self, weights: Sequence[float]):
        if not isinstance(weights, (list, tuple, ndarray, Series)):
            raise ArgumentTypeError("weights must be a sequence")
        if len(weights) != self.data_size:
            raise ArgumentStructureError("weights must have the same length as data")
        if not all(isinstance(w, (float, int, integer)) for w in weights):
            raise ArgumentTypeError("weights must be numeric")
        self.weights = weights
        return self

    def set_cumulative(self, cumulative: bool):
        if not isinstance(cumulative, bool):
            raise ArgumentTypeError("cumulative must be a boolean")
        self.cumulative = cumulative
        return self

    def set_bottom(self, bottom: Union[Sequence[float], float]):
        if isinstance(bottom, (int, float, integer)):
            self.bottom = bottom
        elif isinstance(bottom, (list, tuple, ndarray, Series)):
            if not all(isinstance(b, (float, int, integer)) for b in bottom):
                raise ArgumentTypeError("bottom values must be numeric")
            self.bottom = bottom
        else:
            raise ArgumentTypeError("bottom must be a number or sequence of numbers")
        return self

    def set_histtype(
        self, histtype: Literal["bar", "barstacked", "step", "stepfilled"]
    ):
        if histtype not in ["bar", "barstacked", "step", "stepfilled"]:
            raise ArgumentValueError("Invalid histtype")
        self.histtype = histtype
        return self

    def set_align(self, align: Literal["left", "mid", "right"]):
        if align not in ["left", "mid", "right"]:
            raise ArgumentValueError("Invalid align value")
        self.align = align
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        if orientation not in ["horizontal", "vertical"]:
            raise ArgumentValueError("Invalid orientation")
        self.orientation = orientation
        return self

    def set_rwidth(self, rwidth: float):
        if not isinstance(rwidth, (float, int)) or not 0 <= rwidth <= 1:
            raise ArgumentValueError("rwidth must be a number between 0 and 1")
        self.rwidth = float(rwidth)
        return self

    def set_log(self, log: bool):
        if not isinstance(log, bool):
            raise ArgumentTypeError("log must be a boolean")
        self.log = log
        return self

    def set_stacked(self, stacked: bool):
        if not isinstance(stacked, bool):
            raise ArgumentTypeError("stacked must be a boolean")
        self.stacked = stacked
        return self

    def set_edgecolor(self, edgecolor: EdgeColor):
        if isinstance(edgecolor, str) or (
            isinstance(edgecolor, (list, tuple))
            and len(edgecolor) in [3, 4]
            and all(isinstance(x, (int, float)) for x in edgecolor)
        ):
            self.edgecolor = edgecolor
        else:
            raise ArgumentTypeError(
                "edgecolor must be a color string or RGB/RGBA values"
            )
        return self

    def set_facecolor(self, facecolor: FaceColor):
        if isinstance(facecolor, str) or (
            isinstance(facecolor, (list, tuple))
            and len(facecolor) in [3, 4]
            and all(isinstance(x, (int, float)) for x in facecolor)
        ):
            self.facecolor = facecolor
        else:
            raise ArgumentTypeError(
                "facecolor must be a color string or RGB/RGBA values"
            )
        return self

    def set_linewidth(self, linewidth: float):
        if not isinstance(linewidth, (int, float)) or linewidth < 0:
            raise ArgumentValueError("linewidth must be a non-negative number")
        self.linewidth = float(linewidth)
        return self

    def set_fill(self, fill: Union[bool, Sequence[bool]]):
        self.fill = fill
        return self

    def set_linestyle(self, linestyle: str):
        self.linestyle = linestyle
        return self

    def set_hatch(self, hatch: Union[Sequence[str], str]):
        if isinstance(hatch, str):
            self.hatch = hatch
        elif isinstance(hatch, (list, tuple, ndarray, Series)):
            if len(hatch) != self.data_size:
                raise ArgumentStructureError(
                    "hatch must be of the same length as x_data and y_data"
                )
            if not all(isinstance(h, str) for h in hatch):
                raise ArgumentTypeError("All hatch values must be strings")
            self.hatch = hatch
        else:
            raise ArgumentTypeError("hatch must be a string or sequence of strings")
        return self

    def _create_plot(self):
        hist_kwargs = {
            "bins": self.bins,
            "range": self.range,
            "density": self.density,
            "weights": self.weights,
            "cumulative": self.cumulative,
            "bottom": self.bottom,
            "histtype": self.histtype,
            "align": self.align,
            "orientation": self.orientation,
            "rwidth": self.rwidth,
            "log": self.log,
            "stacked": self.stacked,
            "color": self.color,
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "fill": self.fill,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "hatch": self.hatch,
            "alpha": self.alpha,
            "label": self.label,
            "zorder": self.zorder,
        }
        hist_kwargs = {k: v for k, v in hist_kwargs.items() if v is not None}

        try:
            self.ax.hist(self.x_data, **hist_kwargs)
        except Exception as e:
            raise DistributionPlotterException(f"Error while creating histogram: {e}")
