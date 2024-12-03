import re
from typing import Any, Dict, Sequence, Union

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
    _colors,
)
from mitools.visuals.plots.plotter import Plotter


class PiePlotterException(Exception):
    pass


class PiePlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any = None, **kwargs):
        self._pie_params = {
            "explode": {"default": None, "type": Union[Sequence[float], float]},
            "hatch": {"default": None, "type": str},
            "autopct": {"default": None, "type": Union[str, callable]},
            "pctdistance": {"default": 0.6, "type": float},
            "labeldistance": {"default": 1.1, "type": float},
            "shadow": {"default": False, "type": bool},
            "startangle": {"default": None, "type": float},
            "radius": {"default": None, "type": float},
            "counterclock": {"default": True, "type": bool},
            "wedgeprops": {"default": None, "type": Dict},
            "textprops": {"default": None, "type": Dict},
            "center": {"default": (0, 0), "type": tuple},
            "frame": {"default": False, "type": bool},
            "rotatelabels": {"default": False, "type": bool},
            "normalize": {"default": True, "type": bool},
        }
        super().__init__(
            x_data=x_data, y_data=x_data if y_data is None else y_data, **kwargs
        )
        self._init_params.update(self._pie_params)
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
                    "color must be of the same length as data, "
                    + f"len(color)={len(color)} != len(data)={self.data_size}."
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

    def set_explode(self, explode: Union[Sequence[float], float]):
        if isinstance(explode, (float, int, integer)):
            if explode < 0:
                raise ArgumentValueError("explode must be non-negative")
            self.explode = explode
        elif isinstance(explode, (list, tuple, ndarray, Series)):
            if len(explode) != self.data_size:
                raise ArgumentStructureError(
                    "explode must be of the same length as data"
                )
            if not all(isinstance(e, (float, int)) and e >= 0 for e in explode):
                raise ArgumentTypeError(
                    "All explode values must be non-negative numbers"
                )
            self.explode = explode
        else:
            raise ArgumentTypeError("explode must be a number or sequence of numbers")
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

    def set_autopct(self, autopct: Union[str, callable]):
        if isinstance(autopct, str) or callable(autopct):
            self.autopct = autopct
        else:
            raise ArgumentTypeError("autopct must be a string or callable")
        return self

    def set_pctdistance(self, pctdistance: float):
        if not isinstance(pctdistance, (float, int, integer)) or pctdistance < 0:
            raise ArgumentValueError("pctdistance must be a non-negative number")
        self.pctdistance = float(pctdistance)
        return self

    def set_labeldistance(self, labeldistance: float):
        if not isinstance(labeldistance, (float, int, integer)) or labeldistance < 0:
            raise ArgumentValueError("labeldistance must be a non-negative number")
        self.labeldistance = float(labeldistance)
        return self

    def set_shadow(self, shadow: bool):
        if not isinstance(shadow, bool):
            raise ArgumentTypeError("shadow must be a boolean")
        self.shadow = shadow
        return self

    def set_startangle(self, startangle: float):
        if not isinstance(startangle, (float, int, integer)):
            raise ArgumentTypeError("startangle must be a number")
        self.startangle = float(startangle)
        return self

    def set_radius(self, radius: float):
        if not isinstance(radius, (float, int, integer)) or radius <= 0:
            raise ArgumentValueError("radius must be a positive number")
        self.radius = float(radius)
        return self

    def set_counterclock(self, counterclock: bool):
        if not isinstance(counterclock, bool):
            raise ArgumentTypeError("counterclock must be a boolean")
        self.counterclock = counterclock
        return self

    def set_wedgeprops(self, **kwargs):
        self.wedgeprops = kwargs
        return self

    def set_textprops(self, **kwargs):
        self.textprops = kwargs
        return self

    def set_center(self, center: tuple):
        if not isinstance(center, tuple) or len(center) != 2:
            raise ArgumentTypeError("center must be a tuple of (x, y) coordinates")
        if not all(isinstance(c, (float, int)) for c in center):
            raise ArgumentTypeError("center coordinates must be numbers")
        self.center = center
        return self

    def set_frame(self, frame: bool):
        if not isinstance(frame, bool):
            raise ArgumentTypeError("frame must be a boolean")
        self.frame = frame
        return self

    def set_rotatelabels(self, rotatelabels: bool):
        if not isinstance(rotatelabels, bool):
            raise ArgumentTypeError("rotatelabels must be a boolean")
        self.rotatelabels = rotatelabels
        return self

    def set_normalize(self, normalize: bool):
        if not isinstance(normalize, bool):
            raise ArgumentTypeError("normalize must be a boolean")
        self.normalize = normalize
        return self

    def _create_plot(self):
        pie_kwargs = {
            "x": self.x_data,
            "explode": self.explode,
            "hatch": self.hatch,
            "labels": self.label,
            "colors": self.color,
            "autopct": self.autopct,
            "pctdistance": self.pctdistance,
            "labeldistance": self.labeldistance,
            "shadow": self.shadow,
            "startangle": self.startangle,
            "radius": self.radius,
            "counterclock": self.counterclock,
            "wedgeprops": self.wedgeprops,
            "textprops": self.textprops,
            "center": self.center,
            "frame": self.frame,
            "rotatelabels": self.rotatelabels,
            "normalize": self.normalize,
        }
        pie_kwargs = {k: v for k, v in pie_kwargs.items() if v is not None}

        try:
            self.ax.pie(**pie_kwargs)
            self.ax.axis("equal")
        except Exception as e:
            raise PiePlotterException(f"Error while creating pie plot: {e}")
