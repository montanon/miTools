from typing import Literal, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.visuals.plots.matplotlib_typing import (
    LINESTYLES,
    Color,
    ColorSequence,
    EdgeColor,
    EdgeColorSequence,
    LiteralSequence,
    Marker,
    MarkerSequence,
    NumericSequence,
    NumericSequences,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import NUMERIC_TYPES


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
        super().__init__(x_data, y_data, **kwargs)
        self._init_params.update(self._line_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

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
        for n_sequence in range(self.n_sequences):
            plot_kwargs = self._create_line_kwargs(n_sequence)
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
            try:
                self.ax.plot(
                    self.x_data[n_sequence], self.y_data[n_sequence], **plot_kwargs
                )
            except Exception as e:
                raise LinePlotterException(f"Error while creating line plot: {e}")
