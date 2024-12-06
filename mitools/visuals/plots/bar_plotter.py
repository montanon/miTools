from typing import Dict, Literal, Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    validate_literal,
    validate_type,
)


class BarPlotterException(Exception):
    pass


class BarPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        kind: Literal["bar", "stacked"] = "bar",
        ax: Axes = None,
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
        super().__init__(x_data, y_data, ax=ax, **kwargs)
        self._init_params.update(self._bar_params)
        self._set_init_params(**kwargs)
        self._kind = kind

    @property
    def kind(self):
        return self._kind

    def set_log(self, log: bool):
        validate_type(log, bool, "log")
        self.log = log
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        validate_literal(orientation, ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_width(self, widths: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(widths, "width")

    def set_bottom(
        self, bottoms: Union[NumericSequences, NumericSequence, NumericType]
    ):
        return self.set_numeric_sequences(bottoms, "bottom")

    def set_align(self, align: Union[LiteralSequence, Literal["center", "edge"]]):
        return self.set_literal_sequence(align, BARS_ALIGN, "align")

    def set_edgecolor(
        self, edgecolors: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor]
    ):
        return self.set_edgecolor_sequences(edgecolors, "edgecolor")

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(linewidths, "linewidth")

    def set_xerr(self, xerrs: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(xerrs, "xerr")

    def set_yerr(self, yerrs: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(yerrs, "yerr")

    def set_ecolor(self, ecolors: Union[ColorSequences, ColorSequence, Color]):
        return self.set_color_sequences(ecolors, "ecolor")

    def set_capsize(self, capsize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(capsize, "capsize")

    def set_error_kw(self, error_kw: Union[DictSequence, Dict]):
        return self.set_dict_sequence(error_kw, "error_kw")

    def set_facecolor(self, facecolors: Union[ColorSequence, Color]):
        return self.set_color_sequence(facecolors, "facecolor")

    def set_fill(self, fill: Union[Sequence[bool], bool]):
        return self.set_bool_sequence(fill, "fill")

    def set_hatch(
        self, hatches: Union[LiteralSequences, LiteralSequence, Literal["hatches"]]
    ):
        return self.set_literal_sequences(hatches, HATCHES, "hatch")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequence(linestyles, LINESTYLES, "linestyle")

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
        for n_sequence in range(self.n_sequences):
            bar_kwargs = self._create_bar_kwargs(n_sequence)
            bar_kwargs = {k: v for k, v in bar_kwargs.items() if v is not None}
            if self.kind == "stacked":
                if n_sequence == 0:
                    bottom_reference = bar_kwargs.get(
                        "bottom", np.zeros_like(self.y_data[n_sequence])
                    )
                bar_kwargs["bottom"] = bottom_reference
            try:
                if self.orientation == "vertical":
                    bar_kwargs["x"] = self.x_data[n_sequence]
                    bar_kwargs["height"] = self.y_data[n_sequence]
                    self.ax.bar(
                        **bar_kwargs,
                    )
                else:
                    bar_kwargs["y"] = self.x_data[n_sequence]
                    bar_kwargs["width"], bar_kwargs["height"] = (
                        self.y_data[n_sequence],
                        bar_kwargs["width"],
                    )
                    if "bottom" in bar_kwargs:
                        bar_kwargs["left"] = bar_kwargs.pop("bottom")
                    self.ax.barh(
                        **bar_kwargs,
                    )
                if self.kind == "stacked":
                    bottom_reference += self.y_data[n_sequence]
            except Exception as e:
                raise BarPlotterException(f"Error while creating bar plot: {e}")
