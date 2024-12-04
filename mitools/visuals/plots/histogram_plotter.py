from typing import Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import ArgumentStructureError
from mitools.visuals.plots.matplotlib_typing import (
    HATCHES,
    HIST_ALIGN,
    HIST_HISTTYPE,
    LINESTYLES,
    Bins,
    BinsSequence,
    BinsSequences,
    Color,
    ColorSequence,
    ColorSequences,
    EdgeColor,
    EdgeColorSequence,
    EdgeColorSequences,
    LiteralSequence,
    LiteralSequences,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    is_bins,
    is_bins_sequence,
    is_bins_sequences,
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_literal,
    is_literal_sequence,
    is_literal_sequences,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_numeric_tuple_sequence,
    is_sequence,
    validate_bins,
    validate_color,
    validate_consistent_len,
    validate_edgecolor,
    validate_literal,
    validate_numeric,
    validate_numeric_tuple,
    validate_same,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
)


class HistogramPlotterException(Exception):
    pass


class HistogramPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: None = None,
        **kwargs,
    ):
        super().__init__(x_data=x_data, y_data=None, **kwargs)
        self._hist_params = {
            # General Axes.scatter Parameters that are independent of the number of data sequences
            "orientation": {
                "default": "vertical",
                "type": Literal["horizontal", "vertical"],
            },
            "stacked": {"default": False, "type": bool},
            "log": {"default": False, "type": bool},
            # Specific Parameters that are based on the number of data sequences
            "bins": {
                "default": "auto",
                "type": Union[BinsSequences, BinsSequence, Bins],
            },
            "range": {
                "default": None,
                "type": Union[Sequence[NumericTuple], NumericTuple, None],
            },
            "weights": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, None],
            },
            "cumulative": {"default": False, "type": Union[Sequence[bool], bool]},
            "bottom": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType, None],
            },
            "histtype": {
                "default": "bar",
                "type": Union[
                    LiteralSequence,
                    Literal["bar", "barstacked", "step", "stepfilled"],
                ],
            },
            "align": {
                "default": "mid",
                "type": Union[LiteralSequence, Literal["left", "mid", "right"]],
            },
            "rwidth": {
                "default": None,
                "type": Union[NumericSequence, NumericType, None],
            },
            "edgecolor": {
                "default": None,
                "type": Union[EdgeColorSequences, EdgeColorSequence, EdgeColor],
            },
            "facecolor": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "fill": {"default": True, "type": Union[Sequence[bool], bool]},
            "linestyle": {
                "default": "-",
                "type": Union[LiteralSequences, LiteralSequence, Literal],
            },
            "linewidth": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "hatch": {
                "default": None,
                "type": Union[LiteralSequences, LiteralSequence, Literal],
            },
        }
        self._init_params.update(self._hist_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        _valid_orientations = ["horizontal", "vertical"]
        validate_value_in_options(orientation, _valid_orientations, "orientation")
        self.orientation = orientation
        return self

    def set_stacked(self, stacked: bool):
        validate_type(stacked, bool, "stacked")
        self.stacked = stacked
        return self

    def set_log(self, log: bool):
        validate_type(log, bool, "log")
        self.log = log
        return self

    def set_bins(self, bins: Union[BinsSequences, BinsSequence, Bins]):
        if self._multi_data:
            if is_bins_sequences(bins):
                validate_consistent_len(bins, "bins")
                if any(len(sequence) != 1 for sequence in bins):
                    max_len = max(len(sequence) for sequence in bins)
                    validate_same(max_len, self.data_size, "len(bins)", "data_size")
                self.bins = bins
                self._multi_params_structure["bins"] = "sequences"
                return self
            elif is_bins_sequence(bins):
                validate_sequence_length(bins, self.data_size, "bins")
                self.bins = bins
                self._multi_params_structure["bins"] = "sequence"
                return self
            elif is_bins(bins):
                self.bins = bins
                self._multi_params_structure["bins"] = "value"
                return self
        else:
            if is_bins_sequence(bins):
                validate_sequence_length(bins, self.data_size, "bins")
                self.bins = bins
                self._multi_params_structure["bins"] = "sequence"
                return self
            if bins is not None:
                validate_bins(bins)
            self.bins = bins
            self._multi_params_structure["bins"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid bins, must be a sequence of sequences of bins or a sequence of bins."
        )

    def set_range(self, range: Union[Sequence[NumericTuple], NumericTuple, None]):
        if self._multi_data and is_numeric_tuple_sequence(range, size=2):
            validate_sequence_length(range, self.data_size, "range")
            self.range = range
            self._multi_params_structure["range"] = "sequence"
            return self
        if range is not None:
            validate_numeric_tuple(range, 2)
        self.range = range
        self._multi_params_structure["range"] = "value"
        return self

    def set_weights(self, weights: Union[NumericSequences, NumericSequence, None]):
        if self._multi_data and is_numeric_sequences(weights):
            validate_consistent_len(weights, "weights")
            validate_sequence_length(weights, self._n_sequences, "weights")
            for sequence in weights:
                validate_sequence_length(sequence, self.data_size, "weights")
            self.weights = weights
            self._multi_params_structure["weights"] = "sequences"
            return self
        elif is_numeric_sequence(weights):
            validate_sequence_length(weights, self.data_size, "weights")
            self.weights = weights
            self._multi_params_structure["weights"] = "sequence"
            return self
        raise ArgumentStructureError(
            "Invalid weights, must be a sequence of numbers or a sequence of sequences of numbers."
        )

    def set_cumulative(self, cumulative: Union[Sequence[bool], bool]):
        if self._multi_data and is_sequence(cumulative):
            validate_sequence_length(cumulative, self._n_sequences, "cumulative")
            validate_sequence_type(cumulative, bool, "cumulative")
            self.cumulative = cumulative
            self._multi_params_structure["cumulative"] = "sequence"
            return self
        elif isinstance(cumulative, bool):
            self.cumulative = cumulative
            self._multi_params_structure["cumulative"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid cumulative, must be a boolean or sequence of booleans."
        )

    def set_bottom(
        self, bottom: Union[NumericSequences, NumericSequence, NumericType, None]
    ):
        if self._multi_data:
            if is_numeric_sequences(bottom):
                validate_consistent_len(bottom, "bottom")
                if any(len(sequence) != 1 for sequence in bottom):
                    max_len = max(len(sequence) for sequence in bottom)
                    validate_same(max_len, self.data_size, "len(bottom)", "data_size")
                self.bottom = bottom
                self._multi_params_structure["bottom"] = "sequences"
                return self
            elif is_numeric_sequence(bottom):
                validate_sequence_length(bottom, self._n_sequences, "bottom")
                self.bottom = bottom
                self._multi_params_structure["bottom"] = "sequence"
                return self
            elif is_numeric(bottom):
                self.bottom = bottom
                self._multi_params_structure["bottom"] = "value"
                return self
        else:
            if is_numeric_sequence(bottom):
                validate_sequence_length(bottom, self.data_size, "bottom")
                self.bottom = bottom
                self._multi_params_structure["bottom"] = "sequence"
                return self
            elif bottom is not None:
                validate_numeric(bottom, "bottom")
                self.bottom = bottom
                self._multi_params_structure["bottom"] = "value"
                return self
            self.bottom = bottom
            self._multi_params_structure["bottom"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid bottom, must be a numeric value, sequence of numbers, sequences of numbers, or None."
        )

    def set_histtype(
        self,
        histtype: Union[
            LiteralSequence,
            Literal["bar", "barstacked", "step", "stepfilled"],
        ],
    ):
        if self._multi_data and is_literal_sequence(histtype, HIST_HISTTYPE):
            validate_sequence_length(histtype, self._n_sequences, "histtype")
            self.histtype = histtype
            self._multi_params_structure["histtype"] = "sequence"
            return self
        elif is_literal(histtype, HIST_HISTTYPE):
            self.histtype = histtype
            self._multi_params_structure["histtype"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid histtype, must be a literal or sequence of literals from {HIST_HISTTYPE}."
        )

    def set_align(self, align: Union[LiteralSequence, Literal["left", "mid", "right"]]):
        if self._multi_data and is_literal_sequence(align, HIST_ALIGN):
            validate_sequence_length(align, self._n_sequences, "align")
            self.align = align
            self._multi_params_structure["align"] = "sequence"
            return self
        elif is_literal(align, HIST_ALIGN):
            self.align = align
            self._multi_params_structure["align"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid align, must be a literal or sequence of literals from {HIST_ALIGN}."
        )

    def set_rwidth(self, rwidth: Union[NumericSequence, NumericType, None]):
        if self._multi_data and is_numeric_sequence(rwidth):
            validate_sequence_length(rwidth, self._n_sequences, "rwidth")
            self.rwidth = rwidth
            self._multi_params_structure["rwidth"] = "sequence"
            return self
        elif is_numeric(rwidth) or rwidth is None:
            self.rwidth = rwidth
            self._multi_params_structure["rwidth"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid rwidth, must be a numeric value, sequence of numbers, or None."
        )

    def set_edgecolor(
        self, edgecolors: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor]
    ):
        if self._multi_data:
            if is_edgecolor_sequences(edgecolors):
                validate_consistent_len(edgecolors, "edgecolors")
                if any(len(sequence) != 1 for sequence in edgecolors):
                    max_len = max(len(sequence) for sequence in edgecolors)
                    validate_same(
                        max_len, self.data_size, "len(edgecolors)", "data_size"
                    )
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "sequences"
                return self
            elif is_edgecolor_sequence(edgecolors):
                validate_sequence_length(edgecolors, self._n_sequences, "edgecolors")
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "sequence"
                return self
            elif validate_edgecolor(edgecolors):
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "value"
                return self
        else:
            if is_edgecolor_sequence(edgecolors):
                validate_sequence_length(edgecolors, self.data_size, "edgecolors")
                self.edgecolor = edgecolors
                self._multi_params_structure["edgecolor"] = "sequence"
                return self
            validate_edgecolor(edgecolors)
            self.edgecolor = edgecolors
            self._multi_params_structure["edgecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid edgecolors, must be a edgecolor, sequence of edgecolors, or sequences of edgecolors."
        )

    def set_facecolor(self, facecolors: Union[ColorSequences, ColorSequence, Color]):
        if self._multi_data:
            if is_color_sequences(facecolors):
                validate_consistent_len(facecolors, "facecolors")
                if any(len(sequence) != 1 for sequence in facecolors):
                    max_len = max(len(sequence) for sequence in facecolors)
                    validate_same(
                        max_len, self.data_size, "len(facecolors)", "data_size"
                    )
                self.facecolor = facecolors
                self._multi_params_structure["facecolor"] = "sequences"
                return self
            elif is_color_sequence(facecolors):
                validate_sequence_length(facecolors, self._n_sequences, "facecolors")
                self.facecolor = facecolors
                self._multi_params_structure["facecolor"] = "sequence"
                return self
            elif is_color(facecolors):
                self.facecolor = facecolors
                self._multi_params_structure["facecolor"] = "value"
                return self
        else:
            if is_color_sequence(facecolors):
                validate_sequence_length(facecolors, self.data_size, "facecolors")
                self.facecolor = facecolors
                self._multi_params_structure["facecolor"] = "sequence"
                return self
            validate_color(facecolors)
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

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequences, LiteralSequence, Literal["linestyles"]],
    ):
        if self._multi_data:
            if is_literal_sequences(linestyles, LINESTYLES):
                validate_consistent_len(linestyles, "linestyle")
                if any(len(sequence) != 1 for sequence in linestyles):
                    max_len = max(len(sequence) for sequence in linestyles)
                    validate_same(
                        max_len, self.data_size, "len(linestyle)", "data_size"
                    )
                self.linestyle = linestyles
                self._multi_params_structure["linestyle"] = "sequences"
                return self
            elif is_literal_sequence(linestyles, LINESTYLES):
                validate_sequence_length(linestyles, self._n_sequences, "linestyle")
                self.linestyle = linestyles
                self._multi_params_structure["linestyle"] = "sequence"
                return self
            elif is_literal(linestyles, LINESTYLES):
                self.linestyle = linestyles
                self._multi_params_structure["linestyle"] = "value"
                return self
        else:
            if is_literal_sequence(linestyles, LINESTYLES):
                validate_sequence_length(linestyles, self.data_size, "linestyle")
                self.linestyle = linestyles
                self._multi_params_structure["linestyle"] = "sequence"
                return self
            validate_literal(linestyles, LINESTYLES)
            self.linestyle = linestyles
            self._multi_params_structure["linestyle"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid linestyles, must be a literal or sequence of literals from {LINESTYLES}."
        )

    def set_linewidth(
        self, linewidths: Union[NumericSequences, NumericSequence, NumericType]
    ):
        if self._multi_data:
            if is_numeric_sequences(linewidths):
                validate_consistent_len(linewidths, "linewidths")
                if any(len(sequence) != 1 for sequence in linewidths):
                    max_len = max(len(sequence) for sequence in linewidths)
                    validate_same(
                        max_len, self.data_size, "len(linewidths)", "data_size"
                    )
                self.linewidth = linewidths
                self._multi_params_structure["linewidth"] = "sequences"
                return self
            elif is_numeric_sequence(linewidths):
                validate_sequence_length(linewidths, self._n_sequences, "linewidths")
                self.linewidth = linewidths
                self._multi_params_structure["linewidth"] = "sequence"
                return self
            elif is_numeric(linewidths):
                self.linewidth = linewidths
                self._multi_params_structure["linewidth"] = "value"
                return self
        else:
            if is_numeric_sequence(linewidths):
                validate_sequence_length(linewidths, self.data_size, "linewidths")
                self.linewidth = linewidths
                self._multi_params_structure["linewidth"] = "sequence"
                return self
            validate_numeric(linewidths, "linewidths")
            self.linewidth = linewidths
            self._multi_params_structure["linewidth"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid linewidths, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_hatch(
        self, hatches: Union[LiteralSequences, LiteralSequence, Literal["hatches"]]
    ):
        if self._multi_data:
            if is_literal_sequences(hatches, HATCHES):
                validate_consistent_len(hatches, "hatch")
                if any(len(sequence) != 1 for sequence in hatches):
                    max_len = max(len(sequence) for sequence in hatches)
                    validate_same(max_len, self.data_size, "len(hatch)", "data_size")
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "sequences"
                return self
            elif is_literal_sequence(hatches, HATCHES):
                validate_sequence_length(hatches, self._n_sequences, "hatch")
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "sequence"
                return self
            elif is_literal(hatches, HATCHES):
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "value"
                return self
        else:
            if is_literal_sequence(hatches, HATCHES):
                validate_sequence_length(hatches, self.data_size, "hatch")
                self.hatch = hatches
                self._multi_params_structure["hatch"] = "sequence"
                return self
            validate_literal(hatches, HATCHES)
            self.hatch = hatches
            self._multi_params_structure["hatch"] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid hatches, must be a literal or sequence of literals from {HATCHES}."
        )

    def _create_hist_kwargs(self, n_sequence: int):
        hist_kwargs = {
            "orientation": self.orientation,
            "stacked": self.stacked,
            "log": self.log,
            "bins": self.get_sequences_param("bins", n_sequence),
            "range": self.get_sequences_param("range", n_sequence),
            "weights": self.get_sequences_param("weights", n_sequence),
            "cumulative": self.get_sequences_param("cumulative", n_sequence),
            "bottom": self.get_sequences_param("bottom", n_sequence),
            "histtype": self.get_sequences_param("histtype", n_sequence),
            "align": self.get_sequences_param("align", n_sequence),
            "rwidth": self.get_sequences_param("rwidth", n_sequence),
            "color": self.get_sequences_param("color", n_sequence),
            "edgecolor": self.get_sequences_param("edgecolor", n_sequence),
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "fill": self.get_sequences_param("fill", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(hist_kwargs.get("alpha", []), NUMERIC_TYPES)
            and len(hist_kwargs.get("alpha", [])) == 1
        ):
            hist_kwargs["alpha"] = hist_kwargs["alpha"][0]
        return hist_kwargs

    def _create_plot(self):
        for n_sequence in range(self._n_sequences):
            hist_kwargs = self._create_hist_kwargs(n_sequence)
            hist_kwargs = {k: v for k, v in hist_kwargs.items() if v is not None}
            try:
                self.ax.hist(self.x_data[n_sequence], **hist_kwargs)
            except Exception as e:
                raise HistogramPlotterException(f"Error while creating histogram: {e}")
