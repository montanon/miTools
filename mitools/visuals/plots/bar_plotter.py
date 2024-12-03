import re
from typing import Any, Dict, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from mitools.exceptions import (
    ArgumentTypeError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Cmap,
    Color,
    EdgeColor,
    FaceColor,
    _colors,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_sequence,
    validate_length,
    validate_non_negative,
    validate_sequence_length,
    validate_sequence_non_negative,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
)


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
                            f"'color' elements must be valid Matplotlib color strings or HEX codes, got '{c}'"
                        )
                elif is_sequence(c):
                    validate_sequence_length(c, (3, 4), "color elements")
                    validate_sequence_type(c, NUMERIC_TYPES, "color elements")
                else:
                    raise ArgumentTypeError(
                        "color elements must be strings or RGB/RGBA values"
                    )
            self.color = color
            return self

        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_width(self, width: Union[Sequence[float], float]):
        if isinstance(width, NUMERIC_TYPES):
            validate_non_negative(width, "width")
            self.width = width
        elif is_sequence(width):
            validate_length(width, self.data_size, "width")
            validate_sequence_type(width, NUMERIC_TYPES, "width")
            validate_sequence_non_negative(width, "width")
            self.width = width
        else:
            raise ArgumentTypeError("width must be a number or sequence of numbers")
        return self

    def set_bottom(self, bottom: Union[Sequence[float], float]):
        if isinstance(bottom, NUMERIC_TYPES):
            self.bottom = bottom
        elif is_sequence(bottom):
            validate_length(bottom, self.data_size, "bottom")
            validate_sequence_type(bottom, NUMERIC_TYPES, "bottom")
            self.bottom = bottom
        else:
            raise ArgumentTypeError("bottom must be a number or sequence of numbers")
        return self

    def set_align(self, align: Literal["center", "edge"]):
        validate_value_in_options(align, ["center", "edge"], "align")
        self.align = align
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
        elif isinstance(edgecolor, SEQUENCE_TYPES):
            if len(edgecolor) == 3 or len(edgecolor) == 4:
                validate_sequence_type(edgecolor, NUMERIC_TYPES, "edgecolor")
                self.edgecolor = edgecolor
            else:
                validate_length(edgecolor, self.data_size, "edgecolor")
                for ec in edgecolor:
                    if isinstance(ec, str):
                        if ec not in _colors and not re.match(
                            r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", ec
                        ):
                            raise ArgumentTypeError(
                                f"'edgecolor' elements must be valid Matplotlib color strings or HEX codes, got '{ec}'"
                            )
                    elif is_sequence(ec):
                        validate_sequence_length(ec, (3, 4), "edgecolor elements")
                        validate_sequence_type(ec, NUMERIC_TYPES, "edgecolor elements")
                    else:
                        raise ArgumentTypeError(
                            "edgecolor elements must be strings or RGB/RGBA values"
                        )
                self.edgecolor = edgecolor
        else:
            raise ArgumentTypeError(
                "edgecolor must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
            )
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

    def set_xerr(self, xerr: Union[Sequence[float], float]):
        if isinstance(xerr, NUMERIC_TYPES):
            self.xerr = xerr
        elif is_sequence(xerr):
            validate_length(xerr, self.data_size, "xerr")
            validate_sequence_type(xerr, NUMERIC_TYPES, "xerr")
            self.xerr = xerr
        else:
            raise ArgumentTypeError("xerr must be a number or sequence of numbers")
        return self

    def set_yerr(self, yerr: Union[Sequence[float], float]):
        if isinstance(yerr, NUMERIC_TYPES):
            self.yerr = yerr
        elif is_sequence(yerr):
            validate_length(yerr, self.data_size, "yerr")
            validate_sequence_type(yerr, NUMERIC_TYPES, "yerr")
            self.yerr = yerr
        else:
            raise ArgumentTypeError("yerr must be a number or sequence of numbers")
        return self

    def set_ecolor(self, ecolor: EdgeColor):
        if isinstance(ecolor, str):
            if (
                ecolor not in ["face", "none"]
                and ecolor not in _colors
                and not re.match(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", ecolor)
            ):
                raise ArgumentTypeError(
                    f"'ecolor'='{ecolor}' must be 'face', 'none', a valid Matplotlib color string or HEX code."
                )
        elif is_sequence(ecolor):
            validate_sequence_type(ecolor, NUMERIC_TYPES, "ecolor")
            validate_sequence_length(ecolor, (3, 4), "ecolor")
        else:
            raise ArgumentTypeError(
                "ecolor must be 'face', 'none', a color string or RGB/RGBA values"
            )
        self.ecolor = ecolor
        return self

    def set_capsize(self, capsize: float):
        validate_type(capsize, NUMERIC_TYPES, "capsize")
        validate_non_negative(capsize, "capsize")
        self.capsize = capsize
        return self

    def set_error_kw(self, **kwargs):
        validate_type(kwargs, dict, "error_kw")
        self.error_kw = kwargs
        return self

    def set_log(self, log: bool):
        validate_type(log, bool, "log")
        self.log = log
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        validate_value_in_options(
            orientation, ["vertical", "horizontal"], "orientation"
        )
        self.orientation = orientation
        return self

    def set_facecolor(self, facecolor: FaceColor):
        if isinstance(facecolor, str):
            if facecolor not in _colors and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", facecolor
            ):
                raise ArgumentTypeError(
                    f"'facecolor'='{facecolor}' must be a valid Matplotlib color string or HEX code."
                )
            self.facecolor = facecolor
        elif isinstance(facecolor, SEQUENCE_TYPES):
            if is_sequence(facecolor, check_length=True):
                validate_length(facecolor, self.data_size, "facecolor")
                for fc in facecolor:
                    if isinstance(fc, str):
                        if fc not in _colors and not re.match(
                            r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", fc
                        ):
                            raise ArgumentTypeError(
                                f"'facecolor' element '{fc}' must be a valid Matplotlib color string or HEX code."
                            )
                    else:
                        validate_sequence_type(fc, NUMERIC_TYPES, "facecolor")
                        validate_sequence_length(fc, (3, 4), "facecolor")
            else:
                validate_sequence_type(facecolor, NUMERIC_TYPES, "facecolor")
                validate_sequence_length(facecolor, (3, 4), "facecolor")
            self.facecolor = facecolor
        else:
            raise ArgumentTypeError(
                "facecolor must be a color string, RGB/RGBA values, "
                + "or an array-like of color strings/RGB/RGBA values."
            )
        return self

    def set_fill(self, fill: Union[bool, Sequence[bool]]):
        if isinstance(fill, bool):
            validate_type(fill, bool, "fill")
            self.fill = fill
        else:
            validate_type(fill, SEQUENCE_TYPES, "fill")
            validate_sequence_type(fill, bool, "fill")
            validate_length(fill, self.data_size, "fill")
            self.fill = fill
        return self

    def set_linestyle(self, linestyle: str):
        _valid_styles = ["-", "--", "-.", ":", "None", "none", " ", ""]
        validate_value_in_options(linestyle, _valid_styles, "linestyle")
        self.linestyle = linestyle
        return self

    def set_hatch(self, hatch: Union[Sequence[str], str]):
        if isinstance(hatch, str):
            validate_type(hatch, str, "hatch")
            self.hatch = hatch
        else:
            validate_type(hatch, SEQUENCE_TYPES, "hatch")
            validate_sequence_type(hatch, str, "hatch")
            validate_length(hatch, self.data_size, "hatch")
            self.hatch = hatch
        return self

    def set_colormap(self, cmap: Cmap):
        _valid_cmaps = [
            "magma",
            "inferno",
            "plasma",
            "viridis",
            "cividis",
            "twilight",
            "twilight_shifted",
            "turbo",
        ]
        if isinstance(cmap, str):
            validate_value_in_options(cmap, _valid_cmaps, "cmap")
        else:
            validate_type(cmap, Colormap, "cmap")
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
