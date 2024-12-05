from typing import Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.visuals.plots.matplotlib_typing import (
    HATCHES,
    HIST_ALIGN,
    HIST_HISTTYPE,
    LINESTYLES,
    ORIENTATIONS,
    Bins,
    BinsSequence,
    Color,
    ColorSequence,
    EdgeColor,
    EdgeColorSequence,
    LiteralSequence,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    validate_literal,
    validate_type,
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
                "type": Union[BinsSequence, Bins],
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
                "type": Union[NumericSequence, NumericType, None],
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
                "type": Union[EdgeColorSequence, EdgeColor],
            },
            "facecolor": {
                "default": None,
                "type": Union[ColorSequence, Color],
            },
            "fill": {"default": True, "type": Union[Sequence[bool], bool]},
            "linestyle": {
                "default": "-",
                "type": Union[LiteralSequence, Literal["linestyles"]],
            },
            "linewidth": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "hatch": {
                "default": None,
                "type": Union[LiteralSequence, Literal["hatches"]],
            },
        }
        self._init_params.update(self._hist_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        validate_literal(orientation, ORIENTATIONS)
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

    def set_bins(self, bins: Union[BinsSequence, Bins]):
        self.set_bins_sequence(bins, "bins")

    def set_range(self, range: Union[Sequence[NumericTuple], NumericTuple, None]):
        self.set_numeric_tuple_sequence(range, "range")

    def set_weights(self, weights: Union[NumericSequences, NumericSequence, None]):
        self.set_numeric_sequences(weights, "weights")

    def set_cumulative(self, cumulative: Union[Sequence[bool], bool]):
        self.set_bool_sequence(cumulative, "cumulative")

    def set_bottom(
        self, bottom: Union[NumericSequences, NumericSequence, NumericType, None]
    ):
        self.set_numeric_sequences(bottom, "bottom")

    def set_histtype(
        self,
        histtype: Union[
            LiteralSequence,
            Literal["bar", "barstacked", "step", "stepfilled"],
        ],
    ):
        self.set_literal_sequence(histtype, HIST_HISTTYPE, "histtype")

    def set_align(self, align: Union[LiteralSequence, Literal["left", "mid", "right"]]):
        self.set_literal_sequence(align, HIST_ALIGN, "align")

    def set_rwidth(self, rwidth: Union[NumericSequence, NumericType, None]):
        self.set_numeric_sequence(rwidth, "rwidth")

    def set_edgecolor(self, edgecolors: Union[EdgeColorSequence, EdgeColor]):
        self.set_edgecolor_sequence(edgecolors, "edgecolor")

    def set_facecolor(self, facecolors: Union[ColorSequence, Color]):
        self.set_color_sequence(facecolors, "facecolor")

    def set_fill(self, fill: Union[Sequence[bool], bool]):
        self.set_bool_sequence(fill, "fill")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        self.set_literal_sequence(linestyles, LINESTYLES, "linestyle")

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        self.set_numeric_sequence(linewidths, "linewidth")

    def set_hatch(self, hatches: Union[LiteralSequence, Literal["hatches"]]):
        self.set_literal_sequence(hatches, HATCHES, "hatch")

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
