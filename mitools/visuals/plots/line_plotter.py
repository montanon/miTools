import re
from typing import Any, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import (
    ArgumentTypeError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Color,
    EdgeColor,
    FaceColor,
    LineStyle,
    Markers,
    MarkerStyle,
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


class LinePlotterException(Exception):
    pass


class LinePlotter(Plotter):
    def __init__(self, x_data: Any, y_data: Any, **kwargs):
        self._line_params = {
            "marker": {"default": None, "type": Markers},
            "markersize": {"default": None, "type": Union[Sequence[float], float]},
            "markeredgewidth": {"default": None, "type": Union[Sequence[float], float]},
            "markeredgecolor": {"default": None, "type": EdgeColor},
            "markerfacecolor": {"default": None, "type": FaceColor},
            "linestyle": {"default": "-", "type": LineStyle},
            "linewidth": {"default": None, "type": Union[Sequence[float], float]},
        }
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(self._line_params)
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
            if validate_sequence_length(color, (3, 4), "color") is None:
                validate_sequence_type(color, NUMERIC_TYPES, "color")
                self.color = color
                return self
            validate_length(color, self.data_size, "color")
            for c in color:
                if isinstance(c, str):
                    validate_value_in_options(c, _colors, "color")
                else:
                    validate_type(c, SEQUENCE_TYPES, "color")
                    validate_sequence_type(c, NUMERIC_TYPES, "color")
                    validate_sequence_length(c, (3, 4), "color")
            self.color = color
            return self
        raise ArgumentTypeError(
            "color must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_marker(self, marker: Union[Markers, str]):
        if isinstance(marker, str):
            validate_value_in_options(marker, MarkerStyle.markers, "marker")
            self.marker = marker
        elif is_sequence(marker):
            validate_length(marker, self.data_size, "marker")
            validate_sequence_type(marker, str, "marker")
            for m in marker:
                validate_value_in_options(m, MarkerStyle.markers, "marker")
            self.marker = marker
        else:
            raise ArgumentTypeError(
                "marker must be a string or sequence of valid Matplotlib marker strings"
            )
        return self

    def set_markersize(self, markersize: Union[float, Sequence[float]]):
        if isinstance(markersize, NUMERIC_TYPES):
            self.markersize = markersize
        elif is_sequence(markersize):
            validate_length(markersize, self.data_size, "markersize")
            validate_sequence_type(markersize, NUMERIC_TYPES, "markersize")
            self.markersize = markersize
        else:
            raise ArgumentTypeError(
                "markersize must be a number or sequence of numbers"
            )
        return self

    def set_markerfacecolor(self, markerfacecolor: Union[Color, Sequence[Color]]):
        if isinstance(markerfacecolor, str):
            if markerfacecolor not in _colors and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", markerfacecolor
            ):
                raise ArgumentTypeError(
                    f"'markerfacecolor'='{markerfacecolor}' must be a valid Matplotlib color string or HEX code."
                )
            self.markerfacecolor = markerfacecolor
            return self
        if isinstance(markerfacecolor, SEQUENCE_TYPES):
            validate_sequence_type(markerfacecolor, NUMERIC_TYPES, "markerfacecolor")
            validate_sequence_length(markerfacecolor, (3, 4), "markerfacecolor")
            self.markerfacecolor = markerfacecolor
            return self
        if is_sequence(markerfacecolor):
            validate_length(markerfacecolor, self.data_size, "markerfacecolor")
            for c in markerfacecolor:
                if isinstance(c, str):
                    validate_value_in_options(c, _colors, "markerfacecolor")
                else:
                    validate_type(c, SEQUENCE_TYPES, "markerfacecolor")
                    validate_sequence_type(c, NUMERIC_TYPES, "markerfacecolor")
                    validate_sequence_length(c, (3, 4), "markerfacecolor")
            self.markerfacecolor = markerfacecolor
            return self
        raise ArgumentTypeError(
            "markerfacecolor must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_markeredgecolor(self, markeredgecolor: Union[Color, Sequence[Color]]):
        if isinstance(markeredgecolor, str):
            if markeredgecolor not in _colors and not re.match(
                r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", markeredgecolor
            ):
                raise ArgumentTypeError(
                    f"'markeredgecolor'='{markeredgecolor}' must be a valid Matplotlib color string or HEX code."
                )
            self.markeredgecolor = markeredgecolor
            return self
        if isinstance(markeredgecolor, SEQUENCE_TYPES):
            validate_sequence_type(markeredgecolor, NUMERIC_TYPES, "markeredgecolor")
            validate_sequence_length(markeredgecolor, (3, 4), "markeredgecolor")
            self.markeredgecolor = markeredgecolor
            return self
        if is_sequence(markeredgecolor):
            validate_length(markeredgecolor, self.data_size, "markeredgecolor")
            for c in markeredgecolor:
                if isinstance(c, str):
                    validate_value_in_options(c, _colors, "markeredgecolor")
                else:
                    validate_type(c, SEQUENCE_TYPES, "markeredgecolor")
                    validate_sequence_type(c, NUMERIC_TYPES, "markeredgecolor")
                    validate_sequence_length(c, (3, 4), "markeredgecolor")
            self.markeredgecolor = markeredgecolor
            return self
        raise ArgumentTypeError(
            "markeredgecolor must be a string, RGB/RGBA values, or array-like of strings/RGB/RGBA values."
        )

    def set_markeredgewidth(self, markeredgewidth: Union[float, Sequence[float]]):
        if isinstance(markeredgewidth, NUMERIC_TYPES):
            self.markeredgewidth = markeredgewidth
            return self
        if is_sequence(markeredgewidth):
            validate_length(markeredgewidth, self.data_size, "markeredgewidth")
            validate_sequence_type(markeredgewidth, NUMERIC_TYPES, "markeredgewidth")
            self.markeredgewidth = markeredgewidth
            return self
        raise ArgumentTypeError(
            "markeredgewidth must be a number or array-like of numbers."
        )

    def set_linestyle(self, linestyle: LineStyle):
        _valid_styles = ["-", "--", "-.", ":", "None", "none", " ", ""]
        if isinstance(linestyle, str):
            validate_value_in_options(linestyle, _valid_styles, "linestyle")
            self.linestyle = linestyle
        elif is_sequence(linestyle):
            validate_length(linestyle, self.data_size, "linestyle")
            validate_sequence_type(linestyle, str, "linestyle")
            for ls in linestyle:
                validate_value_in_options(ls, _valid_styles, "linestyle")
            self.linestyle = linestyle
        else:
            raise ArgumentTypeError(
                "linestyle must be a string or array-like of strings"
            )
        return self

    def set_linewidth(self, linewidth: Union[Sequence[float], float]):
        if isinstance(linewidth, NUMERIC_TYPES):
            self.linewidth = linewidth
            return self
        if is_sequence(linewidth):
            validate_length(linewidth, self.data_size, "linewidth")
            validate_sequence_type(linewidth, NUMERIC_TYPES, "linewidth")
            self.linewidth = linewidth
            return self
        raise ArgumentTypeError("linewidth must be a number or array-like of numbers.")

    def _create_plot(self):
        plot_kwargs = {
            "color": self.color,
            "marker": self.marker,
            "markersize": self.markersize,
            "markerfacecolor": self.markerfacecolor,
            "markeredgecolor": self.markeredgecolor,
            "markeredgewidth": self.markeredgewidth,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "alpha": self.alpha,
            "label": self.label,
            "zorder": self.zorder,
        }
        plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
        try:
            self.ax.plot(self.x_data, self.y_data, **plot_kwargs)
        except Exception as e:
            raise LinePlotterException(f"Error while creating line plot: {e}")
