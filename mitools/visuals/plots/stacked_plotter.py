from typing import Literal, Union

import numpy as np
from matplotlib.axes import Axes

from mitools.visuals.plots.matplotlib_typing import (
    BASELINES,
    HATCHES,
    LINESTYLES,
    Color,
    ColorSequence,
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

    def set_color(self, color: Union[ColorSequence, Color]):
        return self.set_color_sequence(color, param_name="color")

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
        try:
            y_stack = np.cumsum(self.y_data, axis=0, dtype=np.float32)
            if self.baseline == "zero":
                first_line = 0
            elif self.baseline == "sym":
                first_line = -np.sum(self.y_data, axis=0) * 0.5
                y_stack += first_line[None, :]
            elif self.baseline == "wiggle":
                m = self.y_data.shape[0]
                first_line = (self.y_data * (m - 0.5 - np.arange(m)[:, None])).sum(0)
                first_line /= -m
                y_stack += first_line
            elif self.baseline == "weighted_wiggle":
                total = np.sum(self.y_data, axis=0)
                inv_total = np.zeros_like(total)
                mask = total > 0
                inv_total[mask] = 1.0 / total[mask]
                increase = np.hstack((self.y_data[:, 0:1], np.diff(self.y_data)))
                below_size = total - y_stack
                below_size += 0.5 * self.y_data
                move_up = below_size * inv_total
                move_up[:, 0] = 0.5
                center = (move_up - 0.5) * increase
                center = np.cumsum(center.sum(0))
                first_line = center - 0.5 * total
                y_stack += first_line

            plot_kwargs = self._create_stacked_kwargs(0)
            coll = self.ax.fill_between(
                self.x_data[0], first_line, y_stack[0, :], **plot_kwargs
            )
            coll.sticky_edges.y[:] = [0]
            for i in range(self.n_sequences - 1):
                plot_kwargs = self._create_stacked_kwargs(i + 1)
                self.ax.fill_between(
                    self.x_data[0], y_stack[i, :], y_stack[i + 1, :], **plot_kwargs
                )
        except Exception as e:
            raise StackedPlotterException(f"Error while creating stacked plot: {e}")
        return self.ax
