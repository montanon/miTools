import re
from typing import Any, Dict, Literal, Sequence, Union

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
    Cmap,
    Color,
    EdgeColor,
    FaceColor,
    _colors,
)
from mitools.visuals.plots.plotter import Plotter


class BarPlotterException(Exception):
    pass


class BarPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any, **kwargs):
        self._bar_params = {
            "width": {"default": 0.8, "type": Union[Sequence[float], float]},
            "bottom": {"default": None, "type": Union[Sequence[float], float]},
            "align": {"default": "center", "type": Literal["center", "edge"]},
            "edgecolor": {"default": None, "type": EdgeColor},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
            "xerr": {"default": None, "type": Union[Sequence[float], float]},
            "yerr": {"default": None, "type": Union[Sequence[float], float]},
            "ecolor": {"default": None, "type": EdgeColor},
            "capsize": {"default": None, "type": float},
            "error_kw": {"default": None, "type": Dict},
            "log": {"default": False, "type": bool},
            "orientation": {
                "default": "vertical",
                "type": Literal["vertical", "horizontal"],
            },
            "facecolor": {"default": None, "type": FaceColor},
            "fill": {"default": True, "type": bool},
            "linestyle": {"default": "-", "type": str},
            "hatch": {"default": None, "type": Union[Sequence[str], str]},
            "colormap": {"default": None, "type": Cmap},
        }
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(self._bar_params)
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

    def set_width(self, width: Union[Sequence[float], float]):
        if isinstance(width, (float, int)):
            if width <= 0:
                raise ArgumentValueError("width must be positive")
            self.width = width
        elif isinstance(width, (list, tuple, ndarray, Series)):
            if len(width) != self.data_size:
                raise ArgumentStructureError(
                    "width must be of the same length as x_data and y_data"
                )
            if not all(isinstance(w, (float, int)) and w > 0 for w in width):
                raise ArgumentTypeError("All width values must be positive numbers")
            self.width = width
        else:
            raise ArgumentTypeError("width must be a number or sequence of numbers")
        return self

    def set_bottom(self, bottom: Union[Sequence[float], float]):
        if isinstance(bottom, (float, int)):
            self.bottom = bottom
        elif isinstance(bottom, (list, tuple, ndarray, Series)):
            if len(bottom) != self.data_size:
                raise ArgumentStructureError(
                    "bottom must be of the same length as x_data and y_data"
                )
            if not all(isinstance(b, (float, int)) for b in bottom):
                raise ArgumentTypeError("All bottom values must be numbers")
            self.bottom = bottom
        else:
            raise ArgumentTypeError("bottom must be a number or sequence of numbers")
        return self

    def set_align(self, align: Literal["center", "edge"]):
        if align not in ["center", "edge"]:
            raise ArgumentValueError("align must be either 'center' or 'edge'")
        self.align = align
        return self

    def set_edgecolor(self, edgecolor: EdgeColor):
        if isinstance(edgecolor, str) or (
            isinstance(edgecolor, (list, tuple))
            and len(edgecolor) in [3, 4]
            and all(isinstance(x, (int, float, integer)) for x in edgecolor)
        ):
            self.edgecolor = edgecolor
        elif isinstance(edgecolor, (list, tuple, ndarray, Series)):
            if len(edgecolor) != self.data_size:
                raise ArgumentStructureError(
                    "edgecolor must be of the same length as x_data and y_data"
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
                "edgecolor must be a color string, RGB/RGBA values, "
                + "or an array-like of color strings/RGB/RGBA values."
            )
        return self

    def set_linewidth(self, linewidth: Union[Sequence[float], float]):
        if isinstance(linewidth, (float, int)):
            if linewidth < 0:
                raise ArgumentValueError("linewidth must be non-negative")
            self.linewidth = linewidth
        elif isinstance(linewidth, (list, tuple, ndarray, Series)):
            if len(linewidth) != self.data_size:
                raise ArgumentStructureError(
                    "linewidth must be of the same length as x_data and y_data"
                )
            if not all(isinstance(lw, (float, int)) and lw >= 0 for lw in linewidth):
                raise ArgumentTypeError(
                    "All linewidth values must be non-negative numbers"
                )
            self.linewidth = linewidth
        else:
            raise ArgumentTypeError("linewidth must be a number or sequence of numbers")
        return self

    def set_xerr(self, xerr: Union[Sequence[float], float]):
        if isinstance(xerr, (float, int)):
            self.xerr = xerr
        elif isinstance(xerr, (list, tuple, ndarray, Series)):
            if len(xerr) != self.data_size:
                raise ArgumentStructureError(
                    "xerr must be of the same length as x_data and y_data"
                )
            if not all(isinstance(x, (float, int)) for x in xerr):
                raise ArgumentTypeError("All xerr values must be numbers")
            self.xerr = xerr
        else:
            raise ArgumentTypeError("xerr must be a number or sequence of numbers")
        return self

    def set_yerr(self, yerr: Union[Sequence[float], float]):
        if isinstance(yerr, (float, int)):
            self.yerr = yerr
        elif isinstance(yerr, (list, tuple, ndarray, Series)):
            if len(yerr) != self.data_size:
                raise ArgumentStructureError(
                    "yerr must be of the same length as x_data and y_data"
                )
            if not all(isinstance(y, (float, int)) for y in yerr):
                raise ArgumentTypeError("All yerr values must be numbers")
            self.yerr = yerr
        else:
            raise ArgumentTypeError("yerr must be a number or sequence of numbers")
        return self

    def set_ecolor(self, ecolor: EdgeColor):
        self.ecolor = ecolor
        return self

    def set_capsize(self, capsize: float):
        if not isinstance(capsize, (int, float)) or capsize < 0:
            raise ArgumentValueError("capsize must be a non-negative number")
        self.capsize = capsize
        return self

    def set_error_kw(self, **kwargs):
        self.error_kw = kwargs
        return self

    def set_log(self, log: bool):
        self.log = log
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        if orientation not in ["vertical", "horizontal"]:
            raise ArgumentValueError(
                "orientation must be either 'vertical' or 'horizontal'"
            )
        self.orientation = orientation
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
                    "facecolor must be of the same length as x_data and y_data"
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
                "facecolor must be a color string, RGB/RGBA values, "
                + "or an array-like of color strings/RGB/RGBA values."
            )
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

    def set_colormap(self, cmap: Cmap):
        self.colormap = cmap
        return self

    def _create_plot(self):
        bar_kwargs = {
            "width": self.width,
            "bottom": self.bottom,
            "align": self.align,
            "color": self.color,
            "edgecolor": self.edgecolor,
            "linewidth": self.linewidth,
            "xerr": self.xerr,
            "yerr": self.yerr,
            "ecolor": self.ecolor,
            "capsize": self.capsize,
            "error_kw": self.error_kw,
            "log": self.log,
            "facecolor": self.facecolor,
            "fill": self.fill,
            "linestyle": self.linestyle,
            "hatch": self.hatch,
            "colormap": self.colormap,
            "alpha": self.alpha,
            "label": self.label,
            "zorder": self.zorder,
        }
        bar_kwargs = {k: v for k, v in bar_kwargs.items() if v is not None}

        try:
            if self.orientation == "vertical":
                self.ax.bar(self.x_data, self.y_data, **bar_kwargs)
            else:
                bar_kwargs["height"] = self.y_data
                y_data = bar_kwargs.pop("width")
                bar_kwargs["left"] = bar_kwargs.pop("bottom")
                self.ax.barh(self.x_data, y_data, **bar_kwargs)
        except Exception as e:
            raise BarPlotterException(f"Error while creating bar plot: {e}")
