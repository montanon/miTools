from typing import Dict, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import ArgumentStructureError
from mitools.visuals.plots.matplotlib_typing import (
    BARS_ALIGN,
    HATCHES,
    LINESTYLES,
    ORIENTATIONS,
    Color,
    ColorSequence,
    ColorSequences,
    DictSequence,
    EdgeColor,
    EdgeColorSequence,
    EdgeColorSequences,
    LiteralSequence,
    LiteralSequences,
    NumericSequence,
    NumericSequences,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_dict_sequence,
    is_edgecolor,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_literal,
    is_literal_sequence,
    is_literal_sequences,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_sequence,
    validate_color,
    validate_consistent_len,
    validate_edgecolor,
    validate_literal,
    validate_numeric,
    validate_same,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
)


class BarPlotterException(Exception):
    pass


class BarPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        **kwargs,
    ):
        self._bar_params = {
            # General Axes Parameters that are independent of the number of data sequences
            "log": {"default": False, "type": bool},
            "orientation": {
                "default": "vertical",
                "type": Literal["vertical", "horizontal"],
            },
            # Specific Parameters that are based on the number of data sequences
            "width": {
                "default": 0.8,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "bottom": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "align": {
                "default": "center",
                "type": Union[LiteralSequence, Literal["center", "edge"]],
            },
            "edgecolor": {
                "default": None,
                "type": Union[EdgeColorSequence, EdgeColor],
            },
            "linewidth": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "xerr": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "yerr": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "ecolor": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "capsize": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "error_kw": {"default": None, "type": Union[DictSequence, Dict]},
            "facecolor": {
                "default": None,
                "type": Union[ColorSequence, Color],
            },
            "fill": {"default": True, "type": Union[Sequence[bool], bool]},
            "hatch": {
                "default": None,
                "type": Union[LiteralSequences, LiteralSequence, Literal["hatches"]],
            },
            "linestyle": {
                "default": "-",
                "type": Union[LiteralSequence, Literal["linestyles"]],
            },
        }
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(self._bar_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_log(self, log: bool):
        validate_type(log, bool, "log")
        self.log = log
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        validate_literal(orientation, ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_width(self, widths: Union[NumericSequences, NumericSequence, NumericType]):
        if self._multi_data:
            if is_numeric_sequences(widths):
                validate_consistent_len(widths, "width")
                validate_sequence_length(widths, self._n_sequences, "width")
                if any(len(sequence) != 1 for sequence in widths):
                    max_len = max(len(sequence) for sequence in widths)
                    validate_same(max_len, self.data_size, "len(width)", "data_size")
                self.width = widths
                self._multi_params_structure["width"] = "sequences"
                return self
            elif is_numeric_sequence(widths):
                validate_sequence_length(widths, self.data_size, "width")
                self.width = widths
                self._multi_params_structure["width"] = "sequence"
                return self
            elif is_numeric(widths):
                self.width = widths
                self._multi_params_structure["width"] = "value"
                return self
        else:
            if is_numeric_sequence(widths):
                validate_sequence_length(widths, self.data_size, "width")
                self.width = widths
                self._multi_params_structure["width"] = "sequence"
                return self
            validate_numeric(widths, "width")
            self.width = widths
            self._multi_params_structure["width"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid width, must be a numeric, sequence of numerics, or sequences of numerics."
        )

    def set_bottom(
        self, bottoms: Union[NumericSequences, NumericSequence, NumericType]
    ):
        if self._multi_data:
            if is_numeric_sequences(bottoms):
                validate_consistent_len(bottoms, "bottom")
                validate_sequence_length(bottoms, self._n_sequences, "bottom")
                if any(len(sequence) != 1 for sequence in bottoms):
                    max_len = max(len(sequence) for sequence in bottoms)
                    validate_same(max_len, self.data_size, "len(bottom)", "data_size")
                self.bottom = bottoms
                self._multi_params_structure["bottom"] = "sequences"
                return self
            elif is_numeric_sequence(bottoms):
                validate_sequence_length(bottoms, self._n_sequences, "bottom")
                self.bottom = bottoms
                self._multi_params_structure["bottom"] = "sequence"
                return self
            elif is_numeric(bottoms):
                self.bottom = bottoms
                self._multi_params_structure["bottom"] = "value"
                return self
        else:
            if is_numeric_sequence(bottoms):
                validate_sequence_length(bottoms, self.data_size, "bottom")
                self.bottom = bottoms
                self._multi_params_structure["bottom"] = "sequence"
                return self
            validate_numeric(bottoms, "bottom")
            self.bottom = bottoms
            self._multi_params_structure["bottom"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid bottom, must be a numeric, sequence of numerics, or sequences of numerics."
        )

    def set_align(self, align: Union[LiteralSequence, Literal["center", "edge"]]):
        if self._multi_data and is_literal_sequence(align, BARS_ALIGN):
            validate_sequence_length(align, self._n_sequences, "align")
            self.align = align
            self._multi_params_structure["align"] = "sequence"
            return self
        elif is_literal(align, BARS_ALIGN):
            self.align = align
            self._multi_params_structure["align"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid align, must be a literal or sequence of literals from {BARS_ALIGN}."
        )

    def set_edgecolor(
        self, edgecolors: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor]
    ):
        if self._multi_data:
            if is_edgecolor_sequences(edgecolors):
                validate_consistent_len(edgecolors, "edgecolor")
                if any(len(sequence) != 1 for sequence in edgecolors):
                    max_len = max(len(sequence) for sequence in edgecolors)
                    validate_same(
                        max_len, self.data_size, "len(edgecolor)", "data_size"
                    )
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "sequences"
                return self
            elif is_edgecolor_sequence(edgecolors):
                validate_sequence_length(edgecolors, self._n_sequences, "edgecolor")
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "sequence"
                return self
            elif is_edgecolor(edgecolors):
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "value"
                return self
        else:
            if is_edgecolor_sequence(edgecolors):
                validate_sequence_length(edgecolors, self.data_size, "edgecolor")
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "sequence"
                return self
            if edgecolors is not None:
                validate_edgecolor(edgecolors)
            self.edgecolor = edgecolors
            self._multi_params_structure["edgecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid edgecolors, must be a color, sequence of colors, or sequences of colors."
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        if self._multi_data and is_numeric_sequence(linewidths):
            validate_sequence_length(linewidths, self._n_sequences, "linewidth")
            self.linewidth = linewidths
            self._multi_params_structure["linewidth"] = "sequence"
            return self
        elif is_numeric(linewidths):
            self.linewidth = linewidths
            self._multi_params_structure["linewidth"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid linewidth, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_xerr(self, xerrs: Union[NumericSequences, NumericSequence, NumericType]):
        if self._multi_data:
            if is_numeric_sequences(xerrs):
                validate_consistent_len(xerrs, "xerr")
                validate_sequence_length(xerrs, self._n_sequences, "xerr")
                if any(len(sequence) != 1 for sequence in xerrs):
                    max_len = max(len(sequence) for sequence in xerrs)
                    validate_same(max_len, self.data_size, "len(xerr)", "data_size")
                self.xerr = xerrs
                self._multi_params_structure["xerr"] = "sequences"
                return self
            elif is_numeric_sequence(xerrs):
                validate_sequence_length(xerrs, self.data_size, "xerr")
                self.xerr = xerrs
                self._multi_params_structure["xerr"] = "sequence"
                return self
            elif is_numeric(xerrs):
                self.xerr = xerrs
                self._multi_params_structure["xerr"] = "value"
                return self
        else:
            if is_numeric_sequence(xerrs):
                validate_sequence_length(xerrs, self.data_size, "xerr")
                self.xerr = xerrs
                self._multi_params_structure["xerr"] = "sequence"
                return self
            validate_numeric(xerrs, "xerr")
            self.xerr = xerrs
            self._multi_params_structure["xerr"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid xerr, must be a numeric, sequence of numerics, or sequences of numerics."
        )

    def set_yerr(self, yerrs: Union[NumericSequences, NumericSequence, NumericType]):
        if self._multi_data:
            if is_numeric_sequences(yerrs):
                validate_consistent_len(yerrs, "yerr")
                validate_sequence_length(yerrs, self._n_sequences, "yerr")
                if any(len(sequence) != 1 for sequence in yerrs):
                    max_len = max(len(sequence) for sequence in yerrs)
                    validate_same(max_len, self.data_size, "len(yerr)", "data_size")
                self.yerr = yerrs
                self._multi_params_structure["yerr"] = "sequences"
                return self
            elif is_numeric_sequence(yerrs):
                validate_sequence_length(yerrs, self.data_size, "yerr")
                self.yerr = yerrs
                self._multi_params_structure["yerr"] = "sequence"
                return self
            elif is_numeric(yerrs):
                self.yerr = yerrs
                self._multi_params_structure["yerr"] = "value"
                return self
        else:
            if is_numeric_sequence(yerrs):
                validate_sequence_length(yerrs, self.data_size, "yerr")
                self.yerr = yerrs
                self._multi_params_structure["yerr"] = "sequence"
                return self
            validate_numeric(yerrs, "yerr")
            self.yerr = yerrs
            self._multi_params_structure["yerr"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid yerr, must be a numeric, sequence of numerics, or sequences of numerics."
        )

    def set_ecolor(self, ecolors: Union[ColorSequences, ColorSequence, Color]):
        if self._multi_data:
            if is_color_sequences(ecolors):
                validate_consistent_len(ecolors, "ecolor")
                validate_sequence_length(ecolors, self._n_sequences, "ecolor")
                if any(len(sequence) != 1 for sequence in ecolors):
                    max_len = max(len(sequence) for sequence in ecolors)
                    validate_same(max_len, self.data_size, "len(ecolor)", "data_size")
                self.ecolor = ecolors
                self._multi_params_structure["ecolor"] = "sequences"
                return self
            elif is_color_sequence(ecolors):
                validate_sequence_length(ecolors, self._n_sequences, "ecolor")
                self.ecolor = ecolors
                self._multi_params_structure["ecolor"] = "sequence"
                return self
            elif is_color(ecolors):
                self.ecolor = ecolors
                self._multi_params_structure["ecolor"] = "value"
                return self
        else:
            if is_color_sequence(ecolors):
                validate_sequence_length(ecolors, self.data_size, "ecolor")
                self.ecolor = ecolors
                self._multi_params_structure["ecolor"] = "sequence"
                return self
            if ecolors is not None:
                validate_color(ecolors)
            self.ecolor = ecolors
            self._multi_params_structure["ecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid ecolor, must be a color, sequence of colors, or sequences of colors."
        )

    def set_capsize(self, capsize: Union[NumericSequence, NumericType]):
        if self._multi_data and is_numeric_sequence(capsize):
            validate_sequence_length(capsize, self._n_sequences, "capsize")
            self.capsize = capsize
            self._multi_params_structure["capsize"] = "sequence"
            return self
        elif is_numeric(capsize):
            self.capsize = capsize
            self._multi_params_structure["capsize"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid capsize, must be a numeric value or sequence of numbers."
        )

    def set_error_kw(self, error_kw: Union[DictSequence, Dict]):
        if self._multi_data and is_dict_sequence(error_kw):
            validate_sequence_length(error_kw, self._n_sequences, "error_kw")
            self.error_kw = error_kw
            self._multi_params_structure["error_kw"] = "sequence"
            return self
        elif isinstance(error_kw, dict):
            self.error_kw = error_kw
            self._multi_params_structure["error_kw"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid error_kw, must be a dictionary or sequence of dictionaries."
        )

    def set_facecolor(self, facecolors: Union[ColorSequence, Color]):
        if self._multi_data and is_color_sequence(facecolors):
            validate_sequence_length(facecolors, self._n_sequences, "facecolors")
            self.facecolor = facecolors
            self._multi_params_structure["facecolor"] = "sequence"
            return self
        elif is_color(facecolors):
            self.facecolor = facecolors
            self._multi_params_structure["facecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid facecolors, must be a color, sequence of colors, or sequences of colors."
        )

    def set_fill(self, fill: Union[Sequence[bool], bool]):
        if self._multi_data and is_sequence(fill):
            validate_sequence_length(fill, self._n_sequences, "fill")
            validate_sequence_type(fill, bool, "fill")
            self.fill = fill
            self._multi_params_structure["fill"] = "sequence"
            return self
        elif isinstance(fill, bool):
            self.fill = fill
            self._multi_params_structure["fill"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid fill, must be a boolean or sequence of booleans."
        )

    def set_hatch(
        self, hatches: Union[LiteralSequences, LiteralSequence, Literal["hatches"]]
    ):
        if self._multi_data:
            if is_literal_sequences(hatches, HATCHES):
                validate_consistent_len(hatches, "hatches")
                if any(len(sequence) != 1 for sequence in hatches):
                    max_len = max(len(sequence) for sequence in hatches)
                    validate_same(max_len, self.data_size, "len(hatches)", "data_size")
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "sequences"
                return self
            elif is_literal_sequence(hatches, HATCHES):
                validate_sequence_length(hatches, self._n_sequences, "hatches")
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "sequence"
                return self
            elif is_literal(hatches, HATCHES):
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "value"
                return self
        else:
            if is_literal_sequence(hatches, HATCHES):
                validate_sequence_length(hatches, self.data_size, "hatches")
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "sequence"
                return self
            if hatches is not None:
                validate_literal(hatches, HATCHES)
            self.hatch = hatches
            self._multi_params_structure["hatch"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid hatch, must be a literal or sequence of literals from {HATCHES}."
        )

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        if self._multi_data and is_literal_sequence(linestyles, LINESTYLES):
            validate_sequence_length(linestyles, self._n_sequences, "linestyle")
            self.linestyle = linestyles
            self._multi_params_structure["linestyle"] = "sequence"
            return self
        elif is_literal(linestyles, LINESTYLES):
            self.linestyle = linestyles
            self._multi_params_structure["linestyle"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid linestyle, must be a literal or sequence of literals from {LINESTYLES}."
        )

    def _create_bar_kwargs(self, n_sequence: int):
        bar_kwargs = {
            "width": self.get_sequences_param("width", n_sequence),
            "bottom": self.get_sequences_param("bottom", n_sequence),
            "align": self.get_sequences_param("align", n_sequence),
            "color": self.get_sequences_param("color", n_sequence),
            "edgecolor": self.get_sequences_param("edgecolor", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "xerr": self.get_sequences_param("xerr", n_sequence),
            "yerr": self.get_sequences_param("yerr", n_sequence),
            "ecolor": self.get_sequences_param("ecolor", n_sequence),
            "capsize": self.get_sequences_param("capsize", n_sequence),
            "error_kw": self.get_sequences_param("error_kw", n_sequence),
            "log": self.log,
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "fill": self.get_sequences_param("fill", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(bar_kwargs.get("alpha", []), NUMERIC_TYPES)
            and len(bar_kwargs.get("alpha", [])) == 1
        ):
            bar_kwargs["alpha"] = bar_kwargs["alpha"][0]
        return bar_kwargs

    def _create_plot(self):
        for n_sequence in range(self._n_sequences):
            bar_kwargs = self._create_bar_kwargs(n_sequence)
            bar_kwargs = {k: v for k, v in bar_kwargs.items() if v is not None}
            try:
                if self.orientation == "vertical":
                    self.ax.bar(
                        self.x_data[n_sequence],
                        self.y_data[n_sequence],
                        **bar_kwargs,
                    )
                else:
                    bar_kwargs["height"] = self.y_data[n_sequence]
                    y_data = bar_kwargs.pop("width")
                    bar_kwargs["left"] = bar_kwargs.pop("bottom")
                    self.ax.barh(
                        self.x_data[n_sequence],
                        y_data,
                        **bar_kwargs,
                    )
            except Exception as e:
                raise BarPlotterException(f"Error while creating bar plot: {e}")
