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
    Color,
    EdgeColor,
    FaceColor,
    _colors,
)
from mitools.visuals.plots.plotter import Plotter


class DistributionPlotterException(Exception):
    pass


class DistributionPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any = None, **kwargs):
        self._dist_params = {
            "kernel": {"default": "gaussian", "type": str},
            "bandwidth": {"default": "scott", "type": Union[str, float]},
            "gridsize": {"default": 1_000, "type": int},
            "cut": {"default": 3, "type": float},
            "fill": {"default": True, "type": bool},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
            "linestyle": {"default": "-", "type": str},
            "facecolor": {"default": {}, "type": Dict[str, Any]},
            "orientation": {
                "default": "vertical",
                "type": Literal["vertical", "horizontal"],
            },
        }
        super().__init__(
            x_data=x_data, y_data=x_data if y_data is None else y_data, **kwargs
        )
        self._init_params.update(self._dist_params)
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
        ) and len(color) == 1:
            self.color = color
            return self
        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_kernel(self, kernel: str):
        _kernels = [
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        ]
        if kernel not in _kernels:
            raise ArgumentValueError(f"'kernel'={kernel} must be one of {_kernels}")
        self.kernel = kernel
        return self

    def set_bandwidth(self, bandwidth: Union[str, float]):
        _methods = ["scott", "silverman"]
        if isinstance(bandwidth, str):
            if bandwidth not in _methods:
                raise ArgumentValueError(
                    f"'bandwidth'={bandwidth} must be one of {_methods}"
                )
            self.bandwidth = bandwidth
        elif isinstance(bandwidth, (float, int)):
            if bandwidth <= 0:
                raise ArgumentValueError(f"'bandwidth'={bandwidth} must be positive")
            self.bandwidth = float(bandwidth)
        else:
            raise ArgumentTypeError(
                f"'bandwidth'={bandwidth} must be a string or positive number"
            )
        return self

    def set_gridsize(self, gridsize: int):
        if not isinstance(gridsize, int) or gridsize < 1:
            raise ArgumentValueError(
                f"'gridsize'={gridsize} must be a positive integer"
            )
        self.gridsize = gridsize
        return self

    def set_cut(self, cut: float):
        if not isinstance(cut, (int, float)) or cut <= 0:
            raise ArgumentValueError(f"'cut'={cut} must be a positive number")
        self.cut = float(cut)
        return self

    def set_fill(self, fill: bool):
        if not isinstance(fill, bool):
            raise ArgumentTypeError(f"'fill'={fill} must be a boolean")
        self.fill = fill
        return self

    def set_linewidth(self, linewidth: Union[Sequence[float], float]):
        if isinstance(linewidth, (float, int)):
            if linewidth < 0:
                raise ArgumentValueError(
                    f"'linewidth'={linewidth} must be non-negative"
                )
            self.linewidth = linewidth
        elif isinstance(linewidth, (list, tuple, ndarray, Series)):
            if len(linewidth) != self.data_size:
                raise ArgumentStructureError(
                    "linewidth must be of the same length as data"
                )
            if not all(isinstance(lw, (float, int)) and lw >= 0 for lw in linewidth):
                raise ArgumentTypeError(
                    "All linewidth values must be non-negative numbers"
                )
            self.linewidth = linewidth
        else:
            raise ArgumentTypeError("linewidth must be a number or sequence of numbers")
        return self

    def set_linestyle(self, linestyle: str):
        _valid_styles = ["-", "--", "-.", ":", "None", "none", " ", ""]
        if linestyle not in _valid_styles:
            raise ArgumentValueError(f"linestyle must be one of {_valid_styles}")
        self.linestyle = linestyle
        return self

    def set_facecolor(self, facecolor: FaceColor, alpha: float = None):
        if isinstance(facecolor, str) or (
            isinstance(facecolor, (list, tuple))
            and len(facecolor) in [3, 4]
            and all(isinstance(x, (int, float, integer)) for x in facecolor)
        ):
            if isinstance(alpha, (float, int, integer)) and (
                alpha > 1.0 or alpha < 0.0
            ):
                raise ArgumentValueError(f"'alpha'={alpha} must be between 0.0 and 1.0")
            self.facecolor = dict(facecolor=facecolor, alpha=alpha)
        else:
            raise ArgumentTypeError(
                "facecolor must be a color string or RGB/RGBA values"
            )
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        if orientation not in ["vertical", "horizontal"]:
            raise ArgumentValueError(
                "orientation must be either 'vertical' or 'horizontal'"
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
