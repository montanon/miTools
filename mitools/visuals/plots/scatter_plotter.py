from typing import Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import ArgumentStructureError
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
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_colormap,
    is_colormap_sequence,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_marker_sequence,
    is_marker_sequences,
    is_normalization,
    is_normalization_sequence,
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

    def set_size(self, sizes: Union[NumericSequences, NumericSequence, NumericType]):
        if self._multi_data:
            if is_numeric_sequences(sizes):
                validate_consistent_len(sizes, "sizes")
                if any(len(sequence) != 1 for sequence in sizes):
                    max_len = max(len(sequence) for sequence in sizes)
                    validate_same(max_len, self.data_size, "len(sizes)", "data_size")
                self.size = np.asarray(sizes)
                self._multi_params_structure["size"] = "sequences"
                return self
            elif is_numeric_sequence(sizes):
                validate_sequence_length(sizes, self._n_sequences, "sizes")
                self.size = np.asarray(sizes)
                self._multi_params_structure["size"] = "sequence"
                return self
            elif is_numeric(sizes) or sizes is None:
                self.size = sizes
                self._multi_params_structure["size"] = "value"
                return self
        else:
            if is_numeric_sequence(sizes):
                validate_sequence_length(sizes, self.data_size, "sizes")
                self.size = np.asarray(sizes)
                self._multi_params_structure["size"] = "sequence"
                return self
            if sizes is not None:
                validate_numeric(sizes, "sizes")
            self.size = sizes
            self._multi_params_structure["size"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid sizes, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

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
                self.linewidth = np.asarray(linewidths)
                self._multi_params_structure["linewidth"] = "sequences"
                return self
            elif is_numeric_sequence(linewidths):
                validate_sequence_length(linewidths, self._n_sequences, "linewidths")
                self.linewidth = np.asarray(linewidths)
                self._multi_params_structure["linewidth"] = "sequence"
                return self
            elif is_numeric(linewidths):
                self.linewidth = linewidths
                self._multi_params_structure["linewidth"] = "value"
                return self
        else:
            if is_numeric_sequence(linewidths):
                validate_sequence_length(linewidths, self.data_size, "linewidths")
                self.linewidth = np.asarray(linewidths)
                self._multi_params_structure["linewidth"] = "sequence"
                return self
            validate_numeric(linewidths, "linewidths")
            self.linewidth = linewidths
            self._multi_params_structure["linewidth"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid linewidths, must be a numeric value, sequence of numbers, or sequences of numbers."
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

    def set_colormap(self, colormaps: Union[CmapSequence, Cmap]):
        if self._multi_data and is_colormap_sequence(colormaps):
            validate_sequence_length(colormaps, self._n_sequences, "colormaps")
            self.colormap = colormaps
            self._multi_params_structure["colormap"] = "sequence"
            return self
        elif is_colormap(colormaps):
            self.colormap = colormaps
            self._multi_params_structure["colormap"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid colormaps, must be a colormap, sequence of colormaps, or sequences of colormaps."
        )

    def set_normalization(self, normalization: Union[NormSequence, Norm]):
        if self._multi_data and is_normalization_sequence(normalization):
            validate_sequence_length(normalization, self._n_sequences, "normalization")
            self.normalization = normalization
            self._multi_params_structure["normalization"] = "sequence"
            return self
        elif is_normalization(normalization):
            self.normalization = normalization
            self._multi_params_structure["normalization"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid normalization, must be a normalization, sequence of normalizations, or sequences of normalizations."
        )

    def set_vmin(self, vmin: Union[NumericSequence, NumericType]):
        if self._multi_data and is_numeric_sequence(vmin):
            validate_sequence_length(vmin, self._n_sequences, "vmin")
            self.vmin = np.asarray(vmin)
            self._multi_params_structure["vmin"] = "sequence"
            return self
        elif is_numeric(vmin):
            self.vmin = vmin
            self._multi_params_structure["vmin"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid vmin, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_vmax(self, vmax: Union[NumericSequence, NumericType]):
        if self._multi_data and is_numeric_sequence(vmax):
            validate_sequence_length(vmax, self._n_sequences, "vmax")
            self.vmax = np.asarray(vmax)
            self._multi_params_structure["vmax"] = "sequence"
            return self
        elif is_numeric(vmax):
            self.vmax = vmax
            self._multi_params_structure["vmax"] = "value"
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
        for n_sequence in range(self._n_sequences):
            scatter_kwargs = self._create_scatter_kwargs(n_sequence)
            scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}
            try:
                self.ax.scatter(**scatter_kwargs)
            except Exception as e:
                raise ScatterPlotterException(f"Error while creating scatter plot: {e}")
        if self.hover and self.label is not None:
            pass
