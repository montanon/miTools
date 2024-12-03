import re
from typing import Any, Dict, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
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


class BoxPlotterException(Exception):
    pass


class BoxPlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any = None, **kwargs):
        self._box_params = {
            "notch": {"default": False, "type": bool},
            "sym": {"default": None, "type": str},
            "whis": {"default": 1.5, "type": Union[float, Sequence[float]]},
            "positions": {"default": None, "type": Sequence[float]},
            "widths": {"default": None, "type": Union[float, Sequence[float]]},
            "patch_artist": {"default": False, "type": bool},
            "bootstrap": {"default": None, "type": Union[int, None]},
            "usermedians": {"default": None, "type": Union[Sequence[float], None]},
            "conf_intervals": {"default": None, "type": Union[Sequence[float], None]},
            "meanline": {"default": False, "type": bool},
            "showmeans": {"default": False, "type": bool},
            "showcaps": {"default": True, "type": bool},
            "showbox": {"default": True, "type": bool},
            "showfliers": {"default": True, "type": bool},
            "boxprops": {"default": None, "type": Dict},
            "box_labels": {"default": None, "type": Union[Sequence[str], str]},
            "flierprops": {"default": None, "type": Dict},
            "medianprops": {"default": None, "type": Dict},
            "meanprops": {"default": None, "type": Dict},
            "capprops": {"default": None, "type": Dict},
            "whiskerprops": {"default": None, "type": Dict},
            "manage_ticks": {"default": True, "type": bool},
            "autorange": {"default": False, "type": bool},
            "capwidths": {"default": None, "type": Union[float, Sequence[float]]},
            "orientation": {
                "default": "vertical",
                "type": Literal["vertical", "horizontal"],
            },
        }
        super().__init__(
            x_data=x_data, y_data=x_data if y_data is None else y_data, **kwargs
        )
        self._init_params.update(self._box_params)
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
        if is_sequence(color):
            if len(color) in [3, 4]:
                validate_sequence_type(color, NUMERIC_TYPES, "color")
                self.color = color
                return self
            validate_length(color, self.data_size, "color")
            for c in color:
                if isinstance(c, str):
                    if c not in _colors and not re.match(
                        r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", c
                    ):
                        raise ArgumentTypeError(
                            f"'color' elements must be valid Matplotlib color strings or HEX codes, got {c}"
                        )
                else:
                    validate_type(c, SEQUENCE_TYPES, "color elements")
                    validate_sequence_length(c, (3, 4), "color elements")
                    validate_sequence_type(c, NUMERIC_TYPES, "color elements")
            self.color = color
            return self
        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_notch(self, notch: bool):
        validate_type(notch, bool, "notch")
        self.notch = notch
        return self

    def set_sym(self, sym: str):
        validate_type(sym, str, "sym")
        self.sym = sym
        return self

    def set_whis(self, whis: Union[float, Sequence[float]]):
        if isinstance(whis, NUMERIC_TYPES):
            validate_type(whis, NUMERIC_TYPES, "whis")
            self.whis = float(whis)
        elif is_sequence(whis):
            validate_sequence_type(whis, NUMERIC_TYPES, "whis")
            validate_sequence_length(whis, 2, "whis")
            self.whis = [float(w) for w in whis]
        else:
            raise ArgumentTypeError("whis must be a number or sequence of two numbers")
        return self

    def set_positions(self, positions: Sequence[float]):
        validate_type(positions, SEQUENCE_TYPES, "positions")
        validate_sequence_type(positions, NUMERIC_TYPES, "positions")
        self.positions = positions
        return self

    def set_widths(self, widths: Union[float, Sequence[float]]):
        if isinstance(widths, NUMERIC_TYPES):
            validate_type(widths, NUMERIC_TYPES, "widths")
            self.widths = float(widths)
        elif is_sequence(widths):
            validate_type(widths, SEQUENCE_TYPES, "widths")
            validate_sequence_type(widths, NUMERIC_TYPES, "widths")
            validate_length(widths, self.data_size, "widths")
            self.widths = [float(w) for w in widths]
        else:
            raise ArgumentTypeError("widths must be a number or sequence of numbers")
        return self

    def set_patch_artist(self, patch_artist: bool):
        validate_type(patch_artist, bool, "patch_artist")
        self.patch_artist = patch_artist
        return self

    def set_bootstrap(self, bootstrap: Union[int, None]):
        validate_type(bootstrap, (int, type(None)), "bootstrap")
        self.bootstrap = bootstrap
        return self

    def set_usermedians(self, usermedians: Union[Sequence[float], None]):
        if usermedians is not None:
            validate_type(usermedians, SEQUENCE_TYPES, "usermedians")
            validate_sequence_type(usermedians, NUMERIC_TYPES, "usermedians")
        self.usermedians = usermedians
        return self

    def set_conf_intervals(self, conf_intervals: Union[Sequence[float], None]):
        if conf_intervals is not None:
            validate_type(conf_intervals, SEQUENCE_TYPES, "conf_intervals")
            validate_sequence_type(conf_intervals, NUMERIC_TYPES, "conf_intervals")
        self.conf_intervals = conf_intervals
        return self

    def set_meanline(self, meanline: bool):
        validate_type(meanline, bool, "meanline")
        self.meanline = meanline
        return self

    def set_showmeans(self, showmeans: bool):
        validate_type(showmeans, bool, "showmeans")
        self.showmeans = showmeans
        return self

    def set_showcaps(self, showcaps: bool):
        validate_type(showcaps, bool, "showcaps")
        self.showcaps = showcaps
        return self

    def set_showbox(self, showbox: bool):
        validate_type(showbox, bool, "showbox")
        self.showbox = showbox
        return self

    def set_showfliers(self, showfliers: bool):
        validate_type(showfliers, bool, "showfliers")
        self.showfliers = showfliers
        return self

    def set_boxprops(self, **kwargs):
        self.boxprops = kwargs
        return self

    def set_box_labels(self, box_labels: Union[Sequence[str], str]):
        self.box_labels = box_labels
        return self

    def set_flierprops(self, **kwargs):
        self.flierprops = kwargs
        return self

    def set_medianprops(self, **kwargs):
        self.medianprops = kwargs
        return self

    def set_meanprops(self, **kwargs):
        self.meanprops = kwargs
        return self

    def set_capprops(self, **kwargs):
        self.capprops = kwargs
        return self

    def set_whiskerprops(self, **kwargs):
        self.whiskerprops = kwargs
        return self

    def set_manage_ticks(self, manage_ticks: bool):
        validate_type(manage_ticks, bool, "manage_ticks")
        self.manage_ticks = manage_ticks
        return self

    def set_autorange(self, autorange: bool):
        validate_type(autorange, bool, "autorange")
        self.autorange = autorange
        return self

    def set_capwidths(self, capwidths: Union[float, Sequence[float]]):
        self.capwidths = capwidths
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        validate_value_in_options(
            orientation, ["vertical", "horizontal"], "orientation"
        )
        self.orientation = orientation
        return self

    def _create_plot(self):
        try:
            if isinstance(self.x_data, (list, tuple, ndarray)):
                data = self.x_data if self.y_data is None else self.y_data
            else:
                raise ArgumentTypeError("Data must be array-like")
            box_kwargs = {
                "notch": self.notch,
                "sym": self.sym,
                "vert": self.orientation == "vertical",
                "whis": self.whis,
                "positions": self.positions,
                "widths": self.widths,
                "patch_artist": self.patch_artist,
                "bootstrap": self.bootstrap,
                "usermedians": self.usermedians,
                "conf_intervals": self.conf_intervals,
                "meanline": self.meanline,
                "showmeans": self.showmeans,
                "showcaps": self.showcaps,
                "showbox": self.showbox,
                "showfliers": self.showfliers,
                "boxprops": self.boxprops,
                "labels": self.box_labels,
                "flierprops": self.flierprops,
                "medianprops": self.medianprops,
                "meanprops": self.meanprops,
                "capprops": self.capprops,
                "whiskerprops": self.whiskerprops,
                "manage_ticks": self.manage_ticks,
                "autorange": self.autorange,
                "capwidths": self.capwidths,
            }
            box_kwargs = {k: v for k, v in box_kwargs.items() if v is not None}
            self.ax.boxplot(data, **box_kwargs)
            if self.patch_artist and self.color is not None:
                boxes = [
                    box for box in self.ax.get_children() if isinstance(box, PathPatch)
                ]
                for box in boxes:
                    box.set_facecolor(self.color)
        except Exception as e:
            raise BoxPlotterException(f"Error while creating box plot: {e}")
