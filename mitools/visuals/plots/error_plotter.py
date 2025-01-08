from typing import Literal, Union

import numpy as np
from matplotlib.axes import Axes

from mitools.exceptions import ArgumentValueError
from mitools.visuals.plots.matplotlib_typing import (
    LINESTYLES,
    BoolSequence,
    Color,
    ColorSequence,
    ColorSequences,
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
                "default": 1.0,
                "type": Union[NumericSequence, NumericType],
            },
            "capsize": {"default": 1.0, "type": Union[NumericSequence, NumericType]},
            "capthick": {"default": 1.0, "type": Union[NumericSequence, NumericType]},
            "barsabove": {"default": None, "type": Union[BoolSequence, bool]},
            "lolims": {"default": None, "type": Union[BoolSequence, bool]},
            "uplims": {"default": None, "type": Union[BoolSequence, bool]},
            "xuplims": {"default": None, "type": Union[BoolSequence, bool]},
            "xlolims": {"default": None, "type": Union[BoolSequence, bool]},
            "errorevery": {
                "default": None,
                "type": Union[
                    NumericTupleSequence, NumericSequence, NumericTuple, NumericType
                ],
            },
            "marker": {
                "default": "o",
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
                "default": "",
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
            isinstance(xerrs, np.ndarray)
            and xerrs.ndim == 2
            and xerrs.shape[0] == 2
            and xerrs.shape[1] == self.data_size
        ):
            self.xerr = xerrs
            self._multi_params_structure["xerr"] = "sequence"
            return self
        elif (
            isinstance(xerrs, np.ndarray)
            and xerrs.ndim == 3
            and xerrs.shape[0] == self.n_sequences
            and xerrs.shape[1] == 2
            and xerrs.shape[2] == self.data_size
        ):
            self.xerr = xerrs
            self._multi_params_structure["xerr"] = "sequences"
            return self
        elif (
            is_numeric_tuple_sequences(xerrs)
            or is_numeric_tuple_sequence(xerrs)
            or is_numeric_tuple(xerrs)
        ):
            return self.set_numeric_tuple_sequences(xerrs, 2, "xerr")
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
            isinstance(yerrs, np.ndarray)
            and yerrs.ndim == 2
            and yerrs.shape[0] == 2
            and yerrs.shape[1] == self.data_size
        ):
            self.yerr = yerrs
            self._multi_params_structure["yerr"] = "sequence"
            return self
        elif (
            isinstance(yerrs, np.ndarray)
            and yerrs.ndim == 3
            and yerrs.shape[0] == self.n_sequences
            and yerrs.shape[1] == 2
            and yerrs.shape[2] == self.data_size
        ):
            self.yerr = yerrs
            self._multi_params_structure["yerr"] = "sequences"
            return self
        elif (
            is_numeric_tuple_sequences(yerrs)
            or is_numeric_tuple_sequence(yerrs)
            or is_numeric_tuple(yerrs)
        ):
            return self.set_numeric_tuple_sequences(yerrs, 2, "yerr")
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

    def set_xlolims(self, xlolims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(xlolims, "xlolims")

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
            return self.set_numeric_sequences(errorevery, "errorevery")
        elif (
            is_numeric_tuple_sequences(errorevery)
            or is_numeric_tuple_sequence(errorevery)
            or is_numeric_tuple(errorevery)
        ):
            return self.set_numeric_tuple_sequences(errorevery, 2, "errorevery")
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
            "xerr": self.get_sequences_param("xerr", n_sequence),
            "yerr": self.get_sequences_param("yerr", n_sequence),
            "fmt": self.get_sequences_param("fmt", n_sequence),
            "ecolor": self.get_sequences_param("ecolor", n_sequence),
            "elinewidth": self.get_sequences_param("elinewidth", n_sequence),
            "capsize": self.get_sequences_param("capsize", n_sequence),
            "capthick": self.get_sequences_param("capthick", n_sequence),
            "barsabove": self.get_sequences_param("barsabove", n_sequence),
            "lolims": self.get_sequences_param("lolims", n_sequence),
            "uplims": self.get_sequences_param("uplims", n_sequence),
            "xlolims": self.get_sequences_param("xlolims", n_sequence),
            "xuplims": self.get_sequences_param("xuplims", n_sequence),
            "errorevery": self.get_sequences_param("errorevery", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "color": self.get_sequences_param("color", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "marker": self.get_sequences_param("marker", n_sequence),
            "markersize": self.get_sequences_param("markersize", n_sequence),
            "markeredgewidth": self.get_sequences_param("markeredgewidth", n_sequence),
            "markeredgecolor": self.get_sequences_param("markeredgecolor", n_sequence),
            "markerfacecolor": self.get_sequences_param("markerfacecolor", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
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
