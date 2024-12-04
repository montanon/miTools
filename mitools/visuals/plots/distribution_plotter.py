import re
from typing import Any, Dict, Literal, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import integer, ndarray
from pandas import Series
from scipy import stats

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    BANDWIDTH_METHODS,
    HATCHES,
    HIST_ALIGN,
    HIST_HISTTYPE,
    KERNELS,
    LINESTYLES,
    Bins,
    BinsSequence,
    BinsSequences,
    Color,
    ColorSequence,
    ColorSequences,
    EdgeColor,
    EdgeColorSequence,
    EdgeColorSequences,
    LiteralSequence,
    LiteralSequences,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    is_bins,
    is_bins_sequence,
    is_bins_sequences,
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_literal,
    is_literal_sequence,
    is_literal_sequences,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_numeric_tuple_sequence,
    is_sequence,
    validate_bins,
    validate_color,
    validate_consistent_len,
    validate_edgecolor,
    validate_literal,
    validate_numeric,
    validate_numeric_tuple,
    validate_same,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
    validate_value_in_range,
)


class DistributionPlotterException(Exception):
    pass


class DistributionPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: None = None,
        **kwargs,
    ):
        super().__init__(x_data=x_data, y_data=None, **kwargs)
        self._dist_params = {
            # General Axes.scatter Parameters that are independent of the number of data sequences
            "kernel": {"default": "gaussian", "type": Literal["kernels"]},
            "bandwidth": {
                "default": "scott",
                "type": Union[Literal["bandwidth_methods"], float],
            },
            "gridsize": {"default": 1_000, "type": int},
            "cut": {"default": 3, "type": float},
            "orientation": {
                "default": "vertical",
                "type": Literal["vertical", "horizontal"],
            },
            # Specific Parameters that are based on the number of data sequences
            "fill": {"default": True, "type": bool},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
            "linestyle": {"default": "-", "type": str},
            "facecolor": {"default": {}, "type": Dict[str, Any]},
            "hatch": {
                "default": None,
                "type": Union[LiteralSequences, LiteralSequence, Literal["hatches"]],
            },
        }
        self._init_params.update(self._dist_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_kernel(self, kernel: str):
        validate_literal(kernel, KERNELS, "kernel")
        self.kernel = kernel
        return self

    def set_bandwidth(self, bandwidth: Union[Literal["bandwidth_methods"], float]):
        if isinstance(bandwidth, str):
            validate_literal(bandwidth, BANDWIDTH_METHODS)
            self.bandwidth = bandwidth
        elif isinstance(bandwidth, NUMERIC_TYPES):
            validate_value_in_range(bandwidth, 1e-9, np.inf, "bandwidth")
            self.bandwidth = float(bandwidth)
        return self

    def set_gridsize(self, gridsize: NumericType):
        validate_numeric(gridsize, NUMERIC_TYPES, "gridsize")
        validate_value_in_range(gridsize, 1, np.inf, "gridsize")
        self.gridsize = int(gridsize)
        return self

    def set_cut(self, cut: NumericType):
        validate_numeric(cut, NUMERIC_TYPES, "cut")
        validate_value_in_range(cut, 0, np.inf, "cut")
        self.cut = float(cut)
        return self

    def set_fill(self, fill: bool):
        validate_type(fill, bool, "fill")
        self.fill = fill
        return self

    def set_linewidth(self, linewidth: Union[Sequence[float], float]):
        if isinstance(linewidth, NUMERIC_TYPES):
            validate_non_negative(linewidth, "linewidth")
            self.linewidth = linewidth
        elif is_sequence(linewidth):
            validate_length(linewidth, self.data_size, "linewidth")
            validate_sequence_type(linewidth, NUMERIC_TYPES, "linewidth")
            validate_sequence_non_negative(linewidth, "linewidth")
            self.linewidth = linewidth
        else:
            raise ArgumentTypeError("linewidth must be a number or sequence of numbers")
        return self

    def set_linestyle(self, linestyle: str):
        _valid_styles = ["-", "--", "-.", ":", "None", "none", " ", ""]
        validate_value_in_options(linestyle, _valid_styles, "linestyle")
        self.linestyle = linestyle
        return self

    def set_facecolor(self, facecolor: Color, alpha: float = None):
        if isinstance(facecolor, str):
            if facecolor not in COLORS and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", facecolor
            ):
                raise ArgumentTypeError(
                    f"'facecolor'='{facecolor}' must be a valid Matplotlib color string or HEX code."
                )
        elif isinstance(facecolor, SEQUENCE_TYPES):
            validate_sequence_type(facecolor, NUMERIC_TYPES, "facecolor")
            validate_sequence_length(facecolor, (3, 4), "facecolor")
        else:
            raise ArgumentTypeError(
                "facecolor must be a color string or RGB/RGBA values"
            )
        if alpha is not None:
            validate_type(alpha, NUMERIC_TYPES, "alpha")
            if alpha > 1.0 or alpha < 0.0:
                raise ArgumentValueError(f"'alpha'={alpha} must be between 0.0 and 1.0")
        self.facecolor = dict(facecolor=facecolor, alpha=alpha)
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        validate_value_in_options(
            orientation, ["vertical", "horizontal"], "orientation"
        )
        self.orientation = orientation
        return self

    def _compute_kde(self, data):
        kde = stats.gaussian_kde(
            data,
            bw_method=self.bandwidth,
        )
        if self.orientation == "vertical":
            grid_min = min(data) - self.cut * kde.covariance_factor()
            grid_max = max(data) + self.cut * kde.covariance_factor()
        else:
            grid_min = min(data) - self.cut * kde.covariance_factor()
            grid_max = max(data) + self.cut * kde.covariance_factor()
        grid = np.linspace(grid_min, grid_max, self.gridsize)
        kde_values = kde(grid)
        return grid, kde_values

    def _create_plot(self):
        try:
            if isinstance(self.x_data, (list, tuple, ndarray, Series)):
                grid, density = self._compute_kde(self.x_data)
            else:
                raise ArgumentTypeError("Data must be array-like")
            plot_kwargs = {
                "color": self.color,
                "alpha": self.alpha,
                "label": self.label,
                "linewidth": self.linewidth,
                "linestyle": self.linestyle,
                "zorder": self.zorder,
            }
            if self.fill:
                fill_kwargs = {
                    "facecolor": self.facecolor.get("facecolor", self.color),
                    "alpha": self.facecolor.get("alpha", self.alpha),
                    "hatch": self.hatch,
                }
                if self.orientation == "vertical":
                    self.ax.fill_between(grid, density, **fill_kwargs)
                else:
                    self.ax.fill_betweenx(grid, density, **fill_kwargs)
            if self.orientation == "vertical":
                self.ax.plot(grid, density, **plot_kwargs)
            else:
                self.ax.plot(density, grid, **plot_kwargs)
        except Exception as e:
            raise DistributionPlotterException(
                f"Error while creating distribution plot: {e}"
            )
