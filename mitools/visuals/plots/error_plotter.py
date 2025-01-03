from typing import Dict, Literal, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch

from mitools.exceptions import ArgumentStructureError, ArgumentValueError
from mitools.visuals.plots.matplotlib_typing import (
    LINESTYLES,
    ORIENTATIONS,
    BoolSequence,
    Color,
    ColorSequence,
    ColorSequences,
    DictSequence,
    EdgeColor,
    EdgeColorSequence,
    LiteralSequence,
    Marker,
    MarkerSequence,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericTupleSequence,
    NumericTupleSequences,
    NumericType,
    StrSequence,
    StrSequences,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_numeric_tuple,
    is_numeric_tuple_sequence,
    is_numeric_tuple_sequences,
    is_str,
    is_str_sequence,
    validate_literal,
    validate_type,
)


class ErrorPlotterException(Exception):
    pass


class ErrorPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        ax: Axes = None,
        **kwargs,
    ):
        self._error_params = {
            # General Axes Parameters that are independent of the number of data sequences
            "fmt": {"default": None, "type": str},
            # Specific Parameters that are based on the number of data sequences
            "xerr": {
                "default": None,
                "type": Union[
                    NumericTupleSequences,
                    NumericSequences,
                    NumericTupleSequence,
                    NumericSequence,
                    NumericType,
                ],
            },
            "yerr": {
                "default": None,
                "type": Union[
                    NumericTupleSequences,
                    NumericSequences,
                    NumericTupleSequence,
                    NumericSequence,
                    NumericType,
                ],
            },
            "ecolor": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "elinewidth": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "capsize": {"default": None, "type": Union[NumericSequence, NumericType]},
            "capthick": {"default": None, "type": Union[NumericSequence, NumericType]},
            "barsabove": {"default": None, "type": Union[BoolSequence, bool]},
            "lolims": {"default": None, "type": Union[BoolSequence, bool]},
            "uplims": {"default": None, "type": Union[BoolSequence, bool]},
            "xuplims": {"default": None, "type": Union[BoolSequence, bool]},
            "yuplims": {"default": None, "type": Union[BoolSequence, bool]},
            "errorevery": {
                "default": None,
                "type": Union[
                    NumericTupleSequence, NumericSequence, NumericTuple, NumericType
                ],
            },
            "marker": {
                "default": "",
                "type": Union[MarkerSequence, Marker],
            },
            "markersize": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "markeredgewidth": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "markeredgecolor": {
                "default": None,
                "type": Union[EdgeColorSequence, EdgeColor],
            },
            "markerfacecolor": {
                "default": None,
                "type": Union[ColorSequence, Color],
            },
            "linestyle": {
                "default": "-",
                "type": Union[LiteralSequence, Literal["linestyles"]],
            },
            "linewidth": {"default": None, "type": Union[NumericSequence, NumericType]},
        }
        super().__init__(x_data, y_data, ax=ax, **kwargs)
        self._init_params.update(self._error_params)
        self._set_init_params(**kwargs)

    def set_fmt(self, fmt: str):
        validate_type(fmt, str, "fmt")
        self.fmt = fmt
        return self

    def set_xerr(self, xerrs: Union[NumericSequences, NumericSequence, NumericType]):
        if (
            is_numeric_sequences(xerrs)
            or is_numeric_sequence(xerrs)
            or is_numeric(xerrs)
        ):
            return self.set_numeric_sequences(xerrs, "xerr")
        elif (
            is_numeric_tuple_sequences(xerrs)
            or is_numeric_tuple_sequence(xerrs)
            or is_numeric_tuple(xerrs)
        ):
            return self.set_numeric_tuple_sequences(xerrs, "xerr")
        raise ArgumentValueError(
            "xerrs must be numeric or numeric tuple or sequences or sequence of them."
        )

    def set_yerr(self, yerrs: Union[NumericSequences, NumericSequence, NumericType]):
        if (
            is_numeric_sequences(yerrs)
            or is_numeric_sequence(yerrs)
            or is_numeric(yerrs)
        ):
            return self.set_numeric_sequences(yerrs, "yerr")
        elif (
            is_numeric_tuple_sequences(yerrs)
            or is_numeric_tuple_sequence(yerrs)
            or is_numeric_tuple(yerrs)
        ):
            return self.set_numeric_tuple_sequences(yerrs, "yerr")
        raise ArgumentValueError(
            "yerrs must be numeric or numeric tuple or sequences or sequence of them."
        )

    def set_ecolor(self, ecolors: Union[ColorSequences, ColorSequence, Color]):
        return self.set_color_sequences(ecolors, "ecolor")

    def set_elinewidth(self, elinewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(elinewidths, "elinewidth")

    def set_capsize(self, capsize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(capsize, "capsize")

    def set_capthick(self, capthick: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(capthick, "capthick")

    def set_barsabove(self, barsabove: Union[BoolSequence, bool]):
        return self.set_bool_sequence(barsabove, "barsabove")

    def set_lolims(self, lolims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(lolims, "lolims")

    def set_uplims(self, uplims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(uplims, "uplims")

    def set_xuplims(self, xuplims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(xuplims, "xuplims")

    def set_yuplims(self, yuplims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(yuplims, "yuplims")

    def set_errorevery(
        self,
        errorevery: Union[
            NumericTupleSequence, NumericSequence, NumericTuple, NumericType
        ],
    ):
        if (
            is_numeric_sequences(errorevery)
            or is_numeric_sequence(errorevery)
            or is_numeric(errorevery)
        ):
            return self.set_numeric_tuple_sequence(errorevery, "errorevery")
        elif (
            is_numeric_tuple_sequences(errorevery)
            or is_numeric_tuple_sequence(errorevery)
            or is_numeric_tuple(errorevery)
        ):
            return self.set_numeric_tuple_sequences(errorevery, "errorevery")
        raise ArgumentValueError(
            "errorevery must be numeric or numeric tuple or sequences or sequence of them."
        )

    def set_marker(self, markers: Union[MarkerSequence, Marker]):
        return self.set_marker_sequence(markers, param_name="marker")

    def set_markersize(self, markersize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(markersize, param_name="markersize")

    def set_markeredgewidth(self, markeredgewidth: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(markeredgewidth, param_name="markeredgewidth")

    def set_markeredgecolor(self, markeredgecolor: Union[EdgeColorSequence, EdgeColor]):
        return self.set_edgecolor_sequence(
            markeredgecolor, param_name="markeredgecolor"
        )

    def set_markerfacecolor(self, markerfacecolor: Union[ColorSequence, Color]):
        return self.set_color_sequence(markerfacecolor, param_name="markerfacecolor")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequence(
            linestyles, options=LINESTYLES, param_name="linestyle"
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(linewidths, param_name="linewidth")

    def _create_error_kwargs(self, n_sequence: int):
        error_kwargs = {
            "fmt": self.get_sequences_param("fmt", n_sequence),
            "xerr": self.get_sequences_param("xerr", n_sequence),
            "yerr": self.get_sequences_param("yerr", n_sequence),
            "ecolor": self.get_sequences_param("ecolor", n_sequence),
            "elinewidth": self.get_sequences_param("elinewidth", n_sequence),
            "capsize": self.get_sequences_param("capsize", n_sequence),
            "capthick": self.get_sequences_param("capthick", n_sequence),
            "barsabove": self.get_sequences_param("barsabove", n_sequence),
            "lolims": self.get_sequences_param("lolims", n_sequence),
            "uplims": self.get_sequences_param("uplims", n_sequence),
            "xuplims": self.get_sequences_param("xuplims", n_sequence),
            "yuplims": self.get_sequences_param("yuplims", n_sequence),
            "errorevery": self.get_sequences_param("errorevery", n_sequence),
            "marker": self.get_sequences_param("marker", n_sequence),
            "markersize": self.get_sequences_param("markersize", n_sequence),
            "markeredgewidth": self.get_sequences_param("markeredgewidth", n_sequence),
            "markeredgecolor": self.get_sequences_param("markeredgecolor", n_sequence),
            "markerfacecolor": self.get_sequences_param("markerfacecolor", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
        }
        if (
            not isinstance(error_kwargs.get("alpha", []), NUMERIC_TYPES)
            and len(error_kwargs.get("alpha", [])) == 1
        ):
            error_kwargs["alpha"] = error_kwargs["alpha"][0]
        return error_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            error_kwargs = self._create_error_kwargs(n_sequence)
            error_kwargs = {k: v for k, v in error_kwargs.items() if v is not None}
            try:
                self.ax.errorbar(
                    self.x_data[n_sequence], self.y_data[n_sequence], **error_kwargs
                )
            except Exception as e:
                raise ErrorPlotterException(f"Error while creating error plot: {e}")
        return self.ax
