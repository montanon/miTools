import re
from typing import Any, Dict, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import (
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    COLORS,
    Color,
)
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
            if color not in COLORS and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", color
            ):
                raise ArgumentTypeError(
                    f"'color'='{color}' must be a valid Matplotlib color string or HEX code."
                )
            self.color = color
            return self
        if is_sequence(color):
            validate_length(color, self.data_size, "color")
            for c in color:
                if isinstance(c, str):
                    validate_value_in_options(c, COLORS, "color")
                else:
                    validate_type(c, SEQUENCE_TYPES, "color")
                    validate_sequence_type(c, NUMERIC_TYPES, "color")
                    validate_sequence_length(c, (3, 4), "color")
            self.color = color
            return self
        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_explode(self, explode: Union[Sequence[float], float]):
        if isinstance(explode, NUMERIC_TYPES):
            if explode < 0:
                raise ArgumentValueError("explode must be non-negative")
            self.explode = explode
        else:
            validate_type(explode, SEQUENCE_TYPES, "explode")
            validate_sequence_type(explode, NUMERIC_TYPES, "explode")
            validate_length(explode, self.data_size, "explode")
            if not all(e >= 0 for e in explode):
                raise ArgumentValueError("All explode values must be non-negative")
            self.explode = explode
        return self

    def set_hatch(self, hatch: Union[Sequence[str], str]):
        if isinstance(hatch, str):
            self.hatch = hatch
        else:
            validate_type(hatch, SEQUENCE_TYPES, "hatch")
            validate_sequence_type(hatch, str, "hatch")
            validate_length(hatch, self.data_size, "hatch")
            self.hatch = hatch
        return self

    def set_autopct(self, autopct: Union[str, callable]):
        validate_type(autopct, (str, callable), "autopct")
        self.autopct = autopct
        return self

    def set_pctdistance(self, pctdistance: float):
        validate_type(pctdistance, NUMERIC_TYPES, "pctdistance")
        if pctdistance < 0:
            raise ArgumentValueError("pctdistance must be a non-negative number")
        self.pctdistance = float(pctdistance)
        return self

    def set_labeldistance(self, labeldistance: float):
        validate_type(labeldistance, NUMERIC_TYPES, "labeldistance")
        if labeldistance < 0:
            raise ArgumentValueError("labeldistance must be a non-negative number")
        self.labeldistance = float(labeldistance)
        return self

    def set_shadow(self, shadow: bool):
        validate_type(shadow, bool, "shadow")
        self.shadow = shadow
        return self

    def set_startangle(self, startangle: float):
        validate_type(startangle, NUMERIC_TYPES, "startangle")
        self.startangle = float(startangle)
        return self

    def set_radius(self, radius: float):
        validate_type(radius, NUMERIC_TYPES, "radius")
        if radius <= 0:
            raise ArgumentValueError("radius must be a positive number")
        self.radius = float(radius)
        return self

    def set_counterclock(self, counterclock: bool):
        validate_type(counterclock, bool, "counterclock")
        self.counterclock = counterclock
        return self

    def set_wedgeprops(self, **kwargs):
        self.wedgeprops = kwargs
        return self

    def set_textprops(self, **kwargs):
        self.textprops = kwargs
        return self

    def set_center(self, center: tuple):
        validate_type(center, tuple, "center")
        validate_sequence_length(center, 2, "center")
        validate_sequence_type(center, NUMERIC_TYPES, "center")
        self.center = center
        return self

    def set_frame(self, frame: bool):
        validate_type(frame, bool, "frame")
        self.frame = frame
        return self

    def set_rotatelabels(self, rotatelabels: bool):
        validate_type(rotatelabels, bool, "rotatelabels")
        self.rotatelabels = rotatelabels
        return self

    def set_normalize(self, normalize: bool):
        validate_type(normalize, bool, "normalize")
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
