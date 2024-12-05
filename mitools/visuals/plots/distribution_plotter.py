from typing import Literal, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import Series
from scipy import stats

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
)
from mitools.visuals.plots.matplotlib_typing import (
    BANDWIDTH_METHODS,
    HATCHES,
    KERNELS,
    LINESTYLES,
    ORIENTATIONS,
    Color,
    ColorSequence,
    LiteralSequence,
    NumericSequence,
    NumericSequences,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    is_color,
    is_color_sequence,
    is_literal,
    is_literal_sequence,
    is_numeric,
    is_numeric_sequence,
    is_sequence,
    validate_literal,
    validate_numeric,
    validate_sequence_length,
    validate_sequence_type,
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
                "type": Literal["horizontal", "vertical"],
            },
            # Specific Parameters that are based on the number of data sequences
            "fill": {"default": True, "type": Union[Sequence[bool], bool]},
            "linestyle": {
                "default": "-",
                "type": Union[LiteralSequence, Literal["linestyles"]],
            },
            "linewidth": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "facecolor": {
                "default": None,
                "type": Union[ColorSequence, Color],
            },
            "hatch": {
                "default": None,
                "type": Union[LiteralSequence, Literal["hatches"]],
            },
        }
        self._init_params.update(self._dist_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_kernel(self, kernel: str):
        validate_literal(kernel, KERNELS)
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
        validate_numeric(gridsize, "gridsize")
        validate_value_in_range(gridsize, 1, np.inf, "gridsize")
        self.gridsize = int(gridsize)
        return self

    def set_cut(self, cut: NumericType):
        validate_numeric(cut, "cut")
        validate_value_in_range(cut, 0, np.inf, "cut")
        self.cut = float(cut)
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        validate_literal(orientation, ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_fill(self, fill: Union[Sequence[bool], bool]):
        if self._multi_data and is_sequence(fill):
            validate_sequence_length(fill, self._n_sequences, "fill")
            validate_sequence_type(fill, bool, "fill")
            self.fill = fill
            self._multi_params_structure["fill"] = "sequence"
            return self
        elif isinstance(fill, bool):
            self.fill = fill
            self._multi_params_structure["fill"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid fill, must be a boolean or sequence of booleans."
        )

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        if self._multi_data and is_literal_sequence(linestyles, LINESTYLES):
            validate_sequence_length(linestyles, self._n_sequences, "linestyle")
            self.linestyle = linestyles
            self._multi_params_structure["linestyle"] = "sequence"
            return self
        elif is_literal(linestyles, LINESTYLES):
            self.linestyle = linestyles
            self._multi_params_structure["linestyle"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid linestyle, must be a literal or sequence of literals from {LINESTYLES}."
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        if self._multi_data and is_numeric_sequence(linewidths):
            validate_sequence_length(linewidths, self._n_sequences, "linewidth")
            self.linewidth = np.asarray(linewidths)
            self._multi_params_structure["linewidth"] = "sequence"
            return self
        elif is_numeric(linewidths):
            self.linewidth = linewidths
            self._multi_params_structure["linewidth"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid linewidth, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_facecolor(self, facecolors: Union[ColorSequence, Color]):
        if self._multi_data and is_color_sequence(facecolors):
            validate_sequence_length(facecolors, self._n_sequences, "facecolors")
            self.facecolor = facecolors
            self._multi_params_structure["facecolor"] = "sequence"
            return self
        elif is_color(facecolors):
            self.facecolor = facecolors
            self._multi_params_structure["facecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid facecolors, must be a color, sequence of colors, or sequences of colors."
        )

    def set_hatch(self, hatches: Union[LiteralSequence, Literal["hatches"]]):
        if self._multi_data and is_literal_sequence(hatches, HATCHES):
            validate_sequence_length(hatches, self._n_sequences, "hatch")
            self.hatch = hatches
            self._multi_params_structure["hatch"] = "sequence"
            return self
        elif is_literal(hatches, HATCHES):
            self.hatch = hatches
            self._multi_params_structure["hatch"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid hatch, must be a literal or sequence of literals from {HATCHES}."
        )

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

    def _create_dist_kwargs(self, n_sequence: int):
        dist_kwargs = {
            "color": self.get_sequences_param("color", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        return dist_kwargs

    def _create_fill_kwargs(self, n_sequence: int):
        fill_kwargs = {
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
            "edgecolor": self.get_sequences_param("color", n_sequence),
        }
        return fill_kwargs

    def _create_plot(self):
        for n_sequence in range(self._n_sequences):
            try:
                if isinstance(self.x_data[n_sequence], (list, tuple, ndarray, Series)):
                    grid, density = self._compute_kde(self.x_data[n_sequence])
                else:
                    raise ArgumentTypeError("Data must be array-like")
                plot_kwargs = self._create_dist_kwargs(n_sequence)
                fill = self.get_sequences_param("fill", n_sequence)
                orientation = self.get_sequences_param("orientation", n_sequence)
                if fill:
                    fill_kwargs = self._create_fill_kwargs(n_sequence)
                    if orientation == "vertical":
                        self.ax.fill_between(grid, density, **fill_kwargs)
                    else:
                        self.ax.fill_betweenx(grid, density, **fill_kwargs)
                if orientation == "vertical":
                    self.ax.plot(grid, density, **plot_kwargs)
                else:
                    self.ax.plot(density, grid, **plot_kwargs)
            except Exception as e:
                raise DistributionPlotterException(
                    f"Error while creating distribution plot: {e}"
                )
