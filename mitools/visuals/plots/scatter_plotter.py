from typing import Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.visuals.plots.matplotlib_typing import (
    Cmap,
    CmapSequence,
    Color,
    ColorSequence,
    ColorSequences,
    EdgeColor,
    EdgeColorSequence,
    EdgeColorSequences,
    Marker,
    MarkerSequence,
    MarkerSequences,
    Norm,
    NormSequence,
    NumericSequence,
    NumericSequences,
    NumericType,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    validate_type,
)


class ScatterPlotterException(Exception):
    pass


class ScatterPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        **kwargs,
    ):
        super().__init__(x_data, y_data, **kwargs)
        self._scatter_params = {
            # General Axes.scatter Parameters that are independent of the number of data sequences
            "plot_non_finite": {"default": False, "type": bool},
            "hover": {"default": False, "type": bool},
            # Specific Parameters that are based on the number of data sequences
            "size": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "marker": {
                "default": "o",
                "type": Union[MarkerSequences, MarkerSequence, Marker],
            },
            "linewidth": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "edgecolor": {
                "default": None,
                "type": Union[EdgeColorSequences, EdgeColorSequence, EdgeColor],
            },
            "facecolor": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "colormap": {"default": None, "type": Union[CmapSequence, Cmap]},
            "normalization": {"default": None, "type": Union[NormSequence, Norm]},
            "vmin": {"default": None, "type": Union[NumericSequence, NumericType]},
            "vmax": {"default": None, "type": Union[NumericSequence, NumericType]},
        }
        self._init_params.update(self._scatter_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_plot_non_finite(self, plot_non_finite: bool):
        validate_type(plot_non_finite, bool, "plot_non_finite")
        self.plot_non_finite = plot_non_finite
        return self

    def set_hover(self, hover: bool):
        validate_type(hover, bool, "hover")
        self.hover = hover
        return self

    def set_size(self, size: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(size, param_name="size")

    def set_marker(self, markers: Union[MarkerSequences, MarkerSequence, Marker]):
        return self.set_marker_sequences(markers, param_name="marker")

    def set_linewidth(
        self, linewidths: Union[NumericSequences, NumericSequence, NumericType]
    ):
        return self.set_numeric_sequences(linewidths, param_name="linewidth")

    def set_edgecolor(
        self, edgecolors: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor]
    ):
        return self.set_edgecolor_sequences(edgecolors, param_name="edgecolor")

    def set_facecolor(self, facecolor: Union[ColorSequences, ColorSequence, Color]):
        return self.set_color_sequences(facecolor, param_name="facecolor")

    def set_colormap(self, colormaps: Union[CmapSequence, Cmap]):
        return self.set_colormap_sequence(colormaps, param_name="colormap")

    def set_normalization(self, normalization: Union[NormSequence, Norm]):
        return self.set_norm_sequence(normalization, param_name="normalization")

    def set_vmin(self, vmin: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(vmin, "vmin")

    def set_vmax(self, vmax: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(vmax, "vmax")

    def set_normalization_range(
        self,
        vmin: Union[NumericSequence, NumericType],
        vmax: Union[NumericSequence, NumericType],
    ):
        self.set_vmin(vmin)
        self.set_vmax(vmax)
        return self

    def _create_scatter_kwargs(self, n_sequence: int):
        scatter_kwargs = {
            "x": self.x_data[n_sequence],
            "y": self.y_data[n_sequence],
            "s": self.get_sequences_param("size", n_sequence),
            "c": self.get_sequences_param("color", n_sequence),
            "marker": self.get_sequences_param("marker", n_sequence),
            "cmap": self.get_sequences_param("colormap", n_sequence),
            "norm": self.get_sequences_param("normalization", n_sequence),
            "vmin": self.get_sequences_param("vmin", n_sequence),
            "vmax": self.get_sequences_param("vmax", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "edgecolor": self.get_sequences_param("edgecolor", n_sequence),
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
            "plotnonfinite": self.plot_non_finite,
        }
        if (
            not isinstance(scatter_kwargs.get("alpha", []), NUMERIC_TYPES)
            and len(scatter_kwargs.get("alpha", [])) == 1
        ):
            scatter_kwargs["alpha"] = scatter_kwargs["alpha"][0]
        return scatter_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            scatter_kwargs = self._create_scatter_kwargs(n_sequence)
            scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}
            try:
                self.ax.scatter(**scatter_kwargs)
            except Exception as e:
                raise ScatterPlotterException(f"Error while creating scatter plot: {e}")
        if self.hover and self.label is not None:
            pass
