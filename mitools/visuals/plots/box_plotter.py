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
        if (
            isinstance(color, (tuple, list, ndarray))
            and len(color) in [3, 4]
            and all(isinstance(c, (float, int, integer)) for c in color)
        ):
            self.color = color
            return self
        if isinstance(color, (list, tuple, ndarray, Series)):
            if len(color) != self.data_size:
                raise ArgumentStructureError("color must be of the same length as data")
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

    def set_notch(self, notch: bool):
        if not isinstance(notch, bool):
            raise ArgumentTypeError("notch must be a boolean")
        self.notch = notch
        return self

    def set_sym(self, sym: str):
        self.sym = sym
        return self

    def set_whis(self, whis: Union[float, Sequence[float]]):
        if isinstance(whis, (float, int)):
            self.whis = float(whis)
        elif isinstance(whis, (list, tuple)) and len(whis) == 2:
            self.whis = [float(w) for w in whis]
        else:
            raise ArgumentTypeError("whis must be a float or a sequence of two floats")
        return self

    def set_positions(self, positions: Sequence[float]):
        if not isinstance(positions, (list, tuple, ndarray)):
            raise ArgumentTypeError("positions must be array-like")
        self.positions = positions
        return self

    def set_widths(self, widths: Union[float, Sequence[float]]):
        if isinstance(widths, (float, int)):
            self.widths = float(widths)
        elif isinstance(widths, (list, tuple, ndarray)):
            if len(widths) != self.data_size:
                raise ArgumentStructureError(
                    "widths must be of the same length as data"
                )
            self.widths = [float(w) for w in widths]
        else:
            raise ArgumentTypeError("widths must be a float or array-like of floats")
        return self

    def set_patch_artist(self, patch_artist: bool):
        if not isinstance(patch_artist, bool):
            raise ArgumentTypeError("patch_artist must be a boolean")
        self.patch_artist = patch_artist
        return self

    def set_bootstrap(self, bootstrap: Union[int, None]):
        if not isinstance(bootstrap, (int, type(None))):
            raise ArgumentTypeError("bootstrap must be an integer or None")
        self.bootstrap = bootstrap
        return self

    def set_usermedians(self, usermedians: Union[Sequence[float], None]):
        if not isinstance(usermedians, (list, tuple, ndarray, Series, type(None))):
            raise ArgumentTypeError("usermedians must be a sequence of floats or None")
        self.usermedians = usermedians
        return self

    def set_conf_intervals(self, conf_intervals: Union[Sequence[float], None]):
        if not isinstance(conf_intervals, (list, tuple, ndarray, Series, type(None))):
            raise ArgumentTypeError(
                "conf_intervals must be a sequence of floats or None"
            )
        self.conf_intervals = conf_intervals
        return self

    def set_meanline(self, meanline: bool):
        if not isinstance(meanline, bool):
            raise ArgumentTypeError("meanline must be a boolean")
        self.meanline = meanline
        return self

    def set_showmeans(self, showmeans: bool):
        if not isinstance(showmeans, bool):
            raise ArgumentTypeError("showmeans must be a boolean")
        self.showmeans = showmeans
        return self

    def set_showcaps(self, showcaps: bool):
        if not isinstance(showcaps, bool):
            raise ArgumentTypeError("showcaps must be a boolean")
        self.showcaps = showcaps
        return self

    def set_showbox(self, showbox: bool):
        if not isinstance(showbox, bool):
            raise ArgumentTypeError("showbox must be a boolean")
        self.showbox = showbox
        return self

    def set_showfliers(self, showfliers: bool):
        if not isinstance(showfliers, bool):
            raise ArgumentTypeError("showfliers must be a boolean")
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
        if not isinstance(manage_ticks, bool):
            raise ArgumentTypeError("manage_ticks must be a boolean")
        self.manage_ticks = manage_ticks
        return self

    def set_autorange(self, autorange: bool):
        if not isinstance(autorange, bool):
            raise ArgumentTypeError("autorange must be a boolean")
        self.autorange = autorange
        return self

    def set_capwidths(self, capwidths: Union[float, Sequence[float]]):
        self.capwidths = capwidths
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        if orientation not in ["vertical", "horizontal"]:
            raise ArgumentValueError(
                "orientation must be either 'vertical' or 'horizontal'"
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
