from typing import Literal, Union

from matplotlib.axes import Axes

from mitools.visuals.plots.matplotlib_typing import (
    BASELINES,
    HATCHES,
    LINESTYLES,
    Color,
    ColorSequence,
    EdgeColor,
    EdgeColorSequence,
    LiteralSequence,
    NumericSequence,
    NumericSequences,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import NUMERIC_TYPES, validate_literal


class StackedPlotterException(Exception):
    pass


class StackedPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        ax: Axes = None,
        **kwargs,
    ):
        self._stacked_params = {
            # Specific Parameters that are based on the number of data sequences
            "baseline": {
                "default": "zero",
                "type": Literal["zero", "sym", "wiggle", "weighted_wiggle"],
            },
            "hatch": {
                "default": None,
                "type": Union[LiteralSequence, Literal["hatches"]],
            },
            "facecolor": {
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
        self._init_params.update(self._stacked_params)
        self._set_init_params(**kwargs)

    def set_baseline(
        self, baseline: Literal["zero", "sym", "wiggle", "weighted_wiggle"]
    ):
        validate_literal(baseline, BASELINES)
        self.baseline = baseline
        return self

    def set_hatch(self, hatches: Union[LiteralSequence, Literal["hatches"]]):
        return self.set_literal_sequence(hatches, HATCHES, "hatch")

    def set_facecolor(self, facecolor: Union[ColorSequence, Color]):
        return self.set_color_sequence(facecolor, param_name="facecolor")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequence(
            linestyles, options=LINESTYLES, param_name="linestyle"
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(linewidths, param_name="linewidth")

    def _create_stacked_kwargs(self, n_sequence: int):
        stacked_kwargs = {
            "color": self.get_sequences_param("color", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(stacked_kwargs.get("alpha", []), NUMERIC_TYPES)
            and len(stacked_kwargs.get("alpha", [])) == 1
        ):
            stacked_kwargs["alpha"] = stacked_kwargs["alpha"][0]
        return stacked_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            plot_kwargs = self._create_stacked_kwargs(n_sequence)
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
            try:
                raise NotImplementedError
            except Exception as e:
                raise StackedPlotterException(f"Error while creating stacked plot: {e}")
            return self.ax
