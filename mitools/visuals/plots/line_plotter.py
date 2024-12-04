from typing import Literal, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import ArgumentStructureError
from mitools.visuals.plots.matplotlib_typing import (
    LINESTYLES,
    Color,
    ColorSequence,
    ColorSequences,
    EdgeColor,
    EdgeColorSequence,
    EdgeColorSequences,
    LiteralSequence,
    Marker,
    MarkerSequence,
    MarkerSequences,
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
    is_edgecolor,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_literal,
    is_literal_sequence,
    is_marker_sequence,
    is_marker_sequences,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    validate_color,
    validate_consistent_len,
    validate_edgecolor,
    validate_marker,
    validate_numeric,
    validate_same,
    validate_sequence_length,
)


class LinePlotterException(Exception):
    pass


class LinePlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        **kwargs,
    ):
        self._line_params = {
            # Specific Parameters that are based on the number of data sequences
            "marker": {
                "default": "o",
                "type": Union[MarkerSequences, MarkerSequence, Marker],
            },
            "markersize": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "markeredgewidth": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "markeredgecolor": {
                "default": None,
                "type": Union[EdgeColorSequences, EdgeColorSequence, EdgeColor],
            },
            "markerfacecolor": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "linestyle": {
                "default": "-",
                "type": Union[LiteralSequence, Literal["linestyles"]],
            },
            "linewidth": {"default": None, "type": Union[NumericSequence, NumericType]},
        }
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(self._line_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_marker(self, markers: Union[MarkerSequences, MarkerSequence, Marker]):
        if self._multi_data:
            if is_marker_sequences(markers):
                validate_consistent_len(markers, "markers")
                if any(len(sequence) != 1 for sequence in markers):
                    max_len = max(len(sequence) for sequence in markers)
                    validate_same(max_len, self.data_size, "len(markers)", "data_size")
                self.marker = markers
                self._multi_params_structure["marker"] = "sequences"
                return self
            elif is_marker_sequence(markers):
                validate_sequence_length(markers, self._n_sequences, "markers")
                self.marker = markers
                self._multi_params_structure["marker"] = "sequence"
                return self
            elif is_numeric(markers):
                self.marker = markers
                self._multi_params_structure["marker"] = "value"
                return self
        else:
            if is_marker_sequence(markers):
                validate_sequence_length(markers, self.data_size, "markers")
                self.marker = markers
                self._multi_params_structure["marker"] = "sequence"
                return self
            validate_marker(markers)
            self.marker = markers
            self._multi_params_structure["marker"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid markers, must be a marker, sequence of markers, or sequences of markers."
        )

    def set_markersize(
        self, markersize: Union[NumericSequences, NumericSequence, NumericType]
    ):
        if self._multi_data:
            if is_numeric_sequences(markersize):
                validate_consistent_len(markersize, "markersize")
                if any(len(sequence) != 1 for sequence in markersize):
                    max_len = max(len(sequence) for sequence in markersize)
                    validate_same(
                        max_len, self.data_size, "len(markersize)", "data_size"
                    )
                self.markersize = markersize
                self._multi_params_structure["markersize"] = "sequences"
                return self
            elif is_numeric_sequence(markersize):
                validate_sequence_length(markersize, self._n_sequences, "markersize")
                self.markersize = markersize
                self._multi_params_structure["markersize"] = "sequence"
                return self
            elif is_numeric(markersize) or markersize is None:
                self.markersize = markersize
                self._multi_params_structure["markersize"] = "value"
                return self
        else:
            if is_numeric_sequence(markersize):
                validate_sequence_length(markersize, self.data_size, "markersize")
                self.markersize = markersize
                self._multi_params_structure["markersize"] = "sequence"
                return self
            if markersize is not None:
                validate_numeric(markersize, "markersize")
            self.markersize = markersize
            self._multi_params_structure["markersize"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid markersizes, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_markeredgewidth(
        self, markeredgewidth: Union[NumericSequences, NumericSequence, NumericType]
    ):
        if self._multi_data:
            if is_numeric_sequences(markeredgewidth):
                validate_consistent_len(markeredgewidth, "markeredgewidth")
                if any(len(sequence) != 1 for sequence in markeredgewidth):
                    max_len = max(len(sequence) for sequence in markeredgewidth)
                    validate_same(
                        max_len, self.data_size, "len(markeredgewidth)", "data_size"
                    )
                self.markeredgewidth = markeredgewidth
                self._multi_params_structure["markeredgewidth"] = "sequences"
                return self
            elif is_numeric_sequence(markeredgewidth):
                validate_sequence_length(
                    markeredgewidth, self._n_sequences, "markeredgewidth"
                )
                self.markeredgewidth = markeredgewidth
                self._multi_params_structure["markeredgewidth"] = "sequence"
                return self
            elif is_numeric(markeredgewidth) or markeredgewidth is None:
                self.markeredgewidth = markeredgewidth
                self._multi_params_structure["markeredgewidth"] = "value"
                return self
        else:
            if is_numeric_sequence(markeredgewidth):
                validate_sequence_length(
                    markeredgewidth, self.data_size, "markeredgewidth"
                )
                self.markeredgewidth = markeredgewidth
                self._multi_params_structure["markeredgewidth"] = "sequence"
                return self
            if markeredgewidth is not None:
                validate_numeric(markeredgewidth, "markeredgewidth")
            self.markeredgewidth = markeredgewidth
            self._multi_params_structure["markeredgewidth"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid markeredgewidth, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_markeredgecolor(
        self, markeredgecolor: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor]
    ):
        if self._multi_data:
            if is_edgecolor_sequences(markeredgecolor):
                validate_consistent_len(markeredgecolor, "markeredgecolor")
                if any(len(sequence) != 1 for sequence in markeredgecolor):
                    max_len = max(len(sequence) for sequence in markeredgecolor)
                    validate_same(
                        max_len, self.data_size, "len(markeredgecolor)", "data_size"
                    )
                self.markeredgecolor = markeredgecolor
                self._multi_params_structure["markeredgecolor"] = "sequences"
                return self
            elif is_edgecolor_sequence(markeredgecolor):
                validate_sequence_length(
                    markeredgecolor, self._n_sequences, "markeredgecolor"
                )
                self.markeredgecolor = markeredgecolor
                self._multi_params_structure["markeredgecolor"] = "sequence"
                return self
            elif is_edgecolor(markeredgecolor):
                self.markeredgecolor = markeredgecolor
                self._multi_params_structure["markeredgecolor"] = "value"
                return self
        else:
            if is_edgecolor_sequence(markeredgecolor):
                validate_sequence_length(
                    markeredgecolor, self.data_size, "markeredgecolor"
                )
                self.markeredgecolor = markeredgecolor
                self._multi_params_structure["markeredgecolor"] = "sequence"
                return self
            if markeredgecolor is not None:
                validate_edgecolor(markeredgecolor)
            self.markeredgecolor = markeredgecolor
            self._multi_params_structure["markeredgecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid markeredgecolor, must be a edgecolor, sequence of edgecolors, or sequences of edgecolors."
        )

    def set_markerfacecolor(
        self, markerfacecolor: Union[ColorSequences, ColorSequence, Color]
    ):
        if self._multi_data:
            if is_color_sequences(markerfacecolor):
                validate_consistent_len(markerfacecolor, "markerfacecolor")
                if any(len(sequence) != 1 for sequence in markerfacecolor):
                    max_len = max(len(sequence) for sequence in markerfacecolor)
                    validate_same(
                        max_len, self.data_size, "len(markerfacecolor)", "data_size"
                    )
                self.markerfacecolor = markerfacecolor
                self._multi_params_structure["markerfacecolor"] = "sequences"
                return self
            elif is_color_sequence(markerfacecolor):
                validate_sequence_length(
                    markerfacecolor, self._n_sequences, "markerfacecolor"
                )
                self.markerfacecolor = markerfacecolor
                self._multi_params_structure["markerfacecolor"] = "sequence"
                return self
            elif is_color(markerfacecolor):
                self.markerfacecolor = markerfacecolor
                self._multi_params_structure["markerfacecolor"] = "value"
                return self
        else:
            if is_color_sequence(markerfacecolor):
                validate_sequence_length(
                    markerfacecolor, self.data_size, "markerfacecolor"
                )
                self.markerfacecolor = markerfacecolor
                self._multi_params_structure["markerfacecolor"] = "sequence"
                return self
            if markerfacecolor is not None:
                validate_color(markerfacecolor)
            self.markerfacecolor = markerfacecolor
            self._multi_params_structure["markerfacecolor"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid markerfacecolor, must be a color, sequence of colors, or sequences of colors."
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

    def _create_line_kwargs(self, n_sequence: int):
        line_kwargs = {
            "color": self.get_sequences_param("color", n_sequence),
            "marker": self.get_sequences_param("marker", n_sequence),
            "markersize": self.get_sequences_param("markersize", n_sequence),
            "markerfacecolor": self.get_sequences_param("markerfacecolor", n_sequence),
            "markeredgecolor": self.get_sequences_param("markeredgecolor", n_sequence),
            "markeredgewidth": self.get_sequences_param("markeredgewidth", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(line_kwargs.get("alpha", []), NUMERIC_TYPES)
            and len(line_kwargs.get("alpha", [])) == 1
        ):
            line_kwargs["alpha"] = line_kwargs["alpha"][0]
        return line_kwargs

    def _create_plot(self):
        for n_sequence in range(self._n_sequences):
            plot_kwargs = self._create_line_kwargs(n_sequence)
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
            try:
                self.ax.plot(
                    self.x_data[n_sequence], self.y_data[n_sequence], **plot_kwargs
                )
            except Exception as e:
                raise LinePlotterException(f"Error while creating line plot: {e}")
