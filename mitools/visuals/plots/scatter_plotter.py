import re
from typing import Any, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import ArgumentStructureError, ArgumentTypeError
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
    _colors,
    _markers,
    _markers_fillstyles,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_colormap,
    is_colormap_sequence,
    is_edgecolor,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_marker,
    is_marker_sequence,
    is_marker_sequences,
    is_normalization,
    is_normalization_sequence,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_sequence,
    validate_color,
    validate_consistent_len,
    validate_edgecolor,
    validate_length,
    validate_marker,
    validate_numeric,
    validate_same_length,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
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
        super().__init__(x_data, y_data, **kwargs)
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

    def set_size(self, sizes: Union[NumericSequences, NumericSequence, NumericType]):
        if self.multi_data:
            if is_numeric_sequences(sizes):
                validate_consistent_len(sizes, "sizes")
                validate_sequence_length(sizes[0], self.data_size, "sizes[0]")
                self.size = sizes
                return self
            elif is_numeric_sequence(sizes):
                validate_sequence_length(sizes, self.n_sequences, "sizes")
                self.size = sizes
                return self
            elif is_numeric(sizes):
                self.size = sizes
                return self
        else:
            if is_numeric_sequence(sizes):
                validate_sequence_length(sizes, self.data_size, "sizes")
                self.size = sizes
                return self
            validate_numeric(sizes, "sizes")
            self.size = sizes
            return self
        raise ArgumentStructureError(
            "Invalid sizes, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_marker(self, markers: Union[MarkerSequences, MarkerSequence, Marker]):
        if self.multi_data:
            if is_marker_sequences(markers):
                validate_consistent_len(markers, "markers")
                validate_sequence_length(markers[0], self.data_size, "markers[0]")
                self.marker = markers
                return self
            elif is_marker_sequence(markers):
                validate_sequence_length(markers, self.n_sequences, "markers")
                self.marker = markers
                return self
            elif is_numeric(markers):
                self.marker = markers
                return self
        else:
            if is_marker_sequence(markers):
                validate_sequence_length(markers, self.data_size, "markers")
                self.marker = markers
                return self
            validate_marker(markers, "markers")
            self.marker = markers
            return self
        raise ArgumentStructureError(
            "Invalid markers, must be a marker, sequence of markers, or sequences of markers."
        )

    def set_linewidth(
        self, linewidths: Union[NumericSequences, NumericSequence, NumericType]
    ):
        if self.multi_data:
            if is_numeric_sequences(linewidths):
                validate_consistent_len(linewidths, "linewidths")
                validate_sequence_length(linewidths[0], self.data_size, "linewidths[0]")
                self.linewidth = linewidths
                return self
            elif is_numeric_sequence(linewidths):
                validate_sequence_length(linewidths, self.n_sequences, "linewidths")
                self.linewidth = linewidths
                return self
            elif is_numeric(linewidths):
                self.linewidth = linewidths
                return self
        else:
            if is_numeric_sequence(linewidths):
                validate_sequence_length(linewidths, self.data_size, "linewidths")
                self.linewidth = linewidths
                return self
            validate_numeric(linewidths, "linewidths")
            self.linewidth = linewidths
            return self
        raise ArgumentStructureError(
            "Invalid linewidths, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_edgecolor(
        self, edgecolors: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor]
    ):
        if self.multi_data:
            if is_edgecolor_sequences(edgecolors):
                validate_consistent_len(edgecolors, "edgecolors")
                validate_sequence_length(edgecolors[0], self.data_size, "edgecolors[0]")
                self.edgecolor = edgecolors
                return self
            elif is_edgecolor_sequence(edgecolors):
                validate_sequence_length(edgecolors, self.n_sequences, "edgecolors")
                self.edgecolor = edgecolors
                return self
            elif validate_edgecolor(edgecolors):
                self.edgecolor = edgecolors
                return self
        else:
            if is_edgecolor_sequence(edgecolors):
                validate_sequence_length(edgecolors, self.data_size, "edgecolors")
                self.edgecolor = edgecolors
                return self
            validate_edgecolor(edgecolors)
            self.edgecolor = edgecolors
            return self
        raise ArgumentStructureError(
            "Invalid edgecolors, must be a edgecolor, sequence of edgecolors, or sequences of edgecolors."
        )

    def set_facecolor(self, facecolors: Union[ColorSequences, ColorSequence, Color]):
        if self.multi_data:
            if is_color_sequences(facecolors):
                validate_consistent_len(facecolors, "facecolors")
                validate_sequence_length(facecolors, self.n_sequences, "facecolors")
                validate_sequence_length(facecolors[0], self.data_size, "facecolors[0]")
                self.facecolor = facecolors
                return self
            elif is_color_sequence(facecolors):
                validate_sequence_length(facecolors, self.n_sequences, "facecolors")
                self.facecolor = facecolors
                return self
            elif is_color(facecolors):
                self.facecolor = facecolors
                return self
        else:
            if is_color_sequence(facecolors):
                validate_sequence_length(facecolors, self.data_size, "facecolors")
                self.facecolor = facecolors
                return self
            validate_color(facecolors)
            self.facecolor = facecolors
            return self
        raise ArgumentStructureError(
            "Invalid facecolors, must be a color, sequence of colors, or sequences of colors."
        )

    def set_colormap(self, colormaps: Union[CmapSequence, Cmap]):
        if self.multi_data and is_colormap_sequence(colormaps):
            validate_sequence_length(colormaps, self.n_sequences, "colormaps")
            self.colormap = colormaps
            return self
        elif is_colormap(colormaps):
            self.colormap = colormaps
            return self
        raise ArgumentStructureError(
            "Invalid colormaps, must be a colormap, sequence of colormaps, or sequences of colormaps."
        )

    def set_normalization(self, normalization: Union[NormSequence, Norm]):
        if self.multi_data and is_normalization_sequence(normalization):
            validate_sequence_length(normalization, self.n_sequences, "normalization")
            self.normalization = normalization
            return self
        elif is_normalization(normalization):
            self.normalization = normalization
            return self
        raise ArgumentStructureError(
            "Invalid normalization, must be a normalization, sequence of normalizations, or sequences of normalizations."
        )

    def set_vmin(self, vmin: Union[NumericSequence, NumericType]):
        if self.multi_data and is_numeric_sequence(vmin):
            validate_sequence_length(vmin, self.n_sequences, "vmin")
            self.vmin = vmin
            return self
        elif is_numeric(vmin):
            self.vmin = vmin
            return self
        raise ArgumentStructureError(
            "Invalid vmin, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_vmax(self, vmax: Union[NumericSequence, NumericType]):
        if self.multi_data and is_numeric_sequence(vmax):
            validate_sequence_length(vmax, self.n_sequences, "vmax")
            self.vmax = vmax
            return self
        elif is_numeric(vmax):
            self.vmax = vmax
            return self
        raise ArgumentStructureError(
            "Invalid vmax, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_normalization_range(
        self,
        vmin: Union[NumericSequence, NumericType],
        vmax: Union[NumericSequence, NumericType],
    ):
        self.set_vmin(vmin)
        self.set_vmax(vmax)
        return self

    def _create_plot(self):
        scatter_kwargs = {
            "x": self.x_data,
            "y": self.y_data,
            "s": self.size,
            "c": self.color,
            "marker": self.marker,
            "cmap": self.colormap,
            "norm": self.normalization,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "alpha": self.alpha,
            "linewidth": self.linewidth,
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "label": self.label,
            "zorder": self.zorder,
            "plotnonfinite": self.plot_non_finite,
        }
        scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}
        try:
            self.ax.scatter(**scatter_kwargs)
        except Exception as e:
            raise ScatterPlotterException(f"Error while creating scatter plot: {e}")
        if self.hover and self.label is not None:
            pass
