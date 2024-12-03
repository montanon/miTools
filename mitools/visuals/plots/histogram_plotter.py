import re
from typing import Any, Literal, Sequence, Tuple, Union

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
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_sequence,
    validate_length,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
)


class HistogramPlotterException(Exception):
    pass


class HistogramPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any = None, **kwargs):
        super().__init__(
            x_data=x_data, y_data=x_data if y_data is None else y_data, **kwargs
        )
        self._hist_params = {
            "bins": {"default": "auto", "type": Union[int, str, Sequence[float]]},
            "range": {"default": None, "type": Union[Tuple[float, float], None]},
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
        if isinstance(color, SEQUENCE_TYPES):
            validate_sequence_type(color, NUMERIC_TYPES, "color")
            validate_sequence_length(color, (3, 4), "color")
            validate_length(color, 1, "color")
            self.color = color
            return self
        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_bins(self, bins: Union[int, str, Sequence[float]]):
        _valid_bins = [
            "auto",
            "fd",
            "doane",
            "scott",
            "stone",
            "rice",
            "sturges",
            "sqrt",
        ]
        if isinstance(bins, int):
            if bins <= 0:
                raise ArgumentValueError("Number of bins must be positive")
            self.bins = bins
            return self
        if isinstance(bins, str):
            validate_value_in_options(bins, _valid_bins, "bins")
            self.bins = bins
            return self
        if is_sequence(bins):
            validate_sequence_type(bins, NUMERIC_TYPES, "bins")
            self.bins = bins
            return self
        raise ArgumentTypeError(
            "bins must be an integer, a string, or a sequence of numbers"
        )

    def set_range(self, range: Tuple[float, float]):
        validate_type(range, (tuple, list), "range")
        validate_sequence_length(range, 2, "range")
        validate_sequence_type(range, NUMERIC_TYPES, "range")
        if range[0] >= range[1]:
            raise ArgumentValueError("range[0] must be less than range[1]")
        self.range = range
        return self

    def set_weights(self, weights: Sequence[float]):
        validate_type(weights, SEQUENCE_TYPES, "weights")
        validate_length(weights, self.data_size, "weights")
        validate_sequence_type(weights, NUMERIC_TYPES, "weights")
        self.weights = weights
        return self

    def set_cumulative(self, cumulative: bool):
        validate_type(cumulative, bool, "cumulative")
        self.cumulative = cumulative
        return self

    def set_bottom(self, bottom: Union[Sequence[float], float]):
        if isinstance(bottom, NUMERIC_TYPES):
            self.bottom = bottom
            return self
        if is_sequence(bottom):
            validate_sequence_type(bottom, NUMERIC_TYPES, "bottom")
            self.bottom = bottom
            return self
        raise ArgumentTypeError("bottom must be a number or sequence of numbers")

    def set_histtype(
        self, histtype: Literal["bar", "barstacked", "step", "stepfilled"]
    ):
        _valid_histtypes = ["bar", "barstacked", "step", "stepfilled"]
        validate_value_in_options(histtype, _valid_histtypes, "histtype")
        self.histtype = histtype
        return self

    def set_align(self, align: Literal["left", "mid", "right"]):
        _valid_aligns = ["left", "mid", "right"]
        validate_value_in_options(align, _valid_aligns, "align")
        self.align = align
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        _valid_orientations = ["horizontal", "vertical"]
        validate_value_in_options(orientation, _valid_orientations, "orientation")
        self.orientation = orientation
        return self

    def set_rwidth(self, rwidth: float):
        validate_type(rwidth, NUMERIC_TYPES, "rwidth")
        if not 0 <= rwidth <= 1:
            raise ArgumentValueError("rwidth must be between 0 and 1")
        self.rwidth = float(rwidth)
        return self

    def set_log(self, log: bool):
        validate_type(log, bool, "log")
        self.log = log
        return self

    def set_stacked(self, stacked: bool):
        validate_type(stacked, bool, "stacked")
        self.stacked = stacked
        return self

    def set_edgecolor(self, edgecolor: EdgeColor):
        if isinstance(edgecolor, str):
            if edgecolor not in _colors and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", edgecolor
            ):
                raise ArgumentTypeError(
                    f"'edgecolor'='{edgecolor}' must be a valid Matplotlib color string or HEX code."
                )
            self.edgecolor = edgecolor
            return self
        if isinstance(edgecolor, SEQUENCE_TYPES):
            validate_sequence_type(edgecolor, NUMERIC_TYPES, "edgecolor")
            validate_sequence_length(edgecolor, (3, 4), "edgecolor")
            self.edgecolor = edgecolor
            return self
        raise ArgumentTypeError("edgecolor must be a string or RGB/RGBA values.")

    def set_facecolor(self, facecolor: FaceColor):
        if isinstance(facecolor, str):
            if facecolor not in _colors and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", facecolor
            ):
                raise ArgumentTypeError(
                    f"'facecolor'='{facecolor}' must be a valid Matplotlib color string or HEX code."
                )
            self.facecolor = facecolor
            return self
        if isinstance(facecolor, SEQUENCE_TYPES):
            validate_sequence_type(facecolor, NUMERIC_TYPES, "facecolor")
            validate_sequence_length(facecolor, (3, 4), "facecolor")
            self.facecolor = facecolor
            return self
        raise ArgumentTypeError("facecolor must be a string or RGB/RGBA values.")

    def set_linewidth(self, linewidth: float):
        validate_type(linewidth, NUMERIC_TYPES, "linewidth")
        if linewidth < 0:
            raise ArgumentValueError("linewidth must be non-negative")
        self.linewidth = float(linewidth)
        return self

    def set_fill(self, fill: Union[bool, Sequence[bool]]):
        if isinstance(fill, bool):
            self.fill = fill
            return self
        if is_sequence(fill):
            validate_length(fill, self.data_size, "fill")
            validate_sequence_type(fill, bool, "fill")
            self.fill = fill
            return self
        raise ArgumentTypeError("fill must be a boolean or sequence of booleans")

    def set_linestyle(self, linestyle: str):
        _valid_styles = ["-", "--", "-.", ":", "None", "none", " ", ""]
        validate_type(linestyle, str, "linestyle")
        validate_value_in_options(linestyle, _valid_styles, "linestyle")
        self.linestyle = linestyle
        return self

    def set_hatch(self, hatch: Union[Sequence[str], str]):
        if isinstance(hatch, str):
            self.hatch = hatch
            return self
        if is_sequence(hatch):
            validate_length(hatch, self.data_size, "hatch")
            validate_sequence_type(hatch, str, "hatch")
            self.hatch = hatch
            return self
        raise ArgumentTypeError("hatch must be a string or sequence of strings")

    def _create_plot(self):
        hist_kwargs = {
            "bins": self.bins,
            "range": self.range,
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
            raise HistogramPlotterException(f"Error while creating histogram: {e}")
