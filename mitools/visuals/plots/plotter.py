import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from numpy import ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Color,
    ColorSequence,
    ColorSequences,
    NumericSequence,
    NumericSequences,
    NumericType,
    Scale,
    StrSequence,
)
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_str_sequence,
    validate_color,
    validate_consistent_len,
    validate_numeric,
    validate_numeric_sequences,
    validate_same_length,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
    validate_value_in_range,
)


class PlotterException(Exception):
    pass


class Plotter(ABC):
    def __init__(
        self,
        x_data: Union[NumericSequence, NumericSequences],
        y_data: Union[NumericSequence, NumericSequences, None],
        **kwargs,
    ):
        self.x_data = self._validate_data(x_data, "x_data")
        self.y_data = self._validate_data(y_data, "y_data")
        validate_same_length(
            self.x_data[0],
            self.y_data[0] if self.y_data is not None else self.x_data[0],
            "x_data",
            "y_data",
        )
        self._n_sequences = len(self.x_data)
        self._multi_data = self._n_sequences > 1
        self.data_size = len(self.x_data[0])
        # General Axes Parameters that are independent of the number of data sequences
        self._single_data_params = {
            "title": {"default": "", "type": Text},
            "xlabel": {"default": "", "type": Text},
            "ylabel": {"default": "", "type": Text},
            "legend": {"default": None, "type": Union[Dict, None]},
            "figsize": {"default": (8, 8), "type": Tuple[float, float]},
            "style": {"default": None, "type": str},
            "grid": {"default": None, "type": Dict[str, Any]},
            "tight_layout": {"default": False, "type": bool},
            "texts": {"default": None, "type": Union[Sequence[Text], Text]},
            "xscale": {"default": None, "type": Scale},
            "yscale": {"default": None, "type": Scale},
            "background": {"default": None, "type": Color},
            "figure_background": {"default": None, "type": Color},
            "suptitle": {"default": None, "type": Text},
            "xlim": {"default": None, "type": Union[Tuple[float, float], None]},
            "ylim": {"default": None, "type": Union[Tuple[float, float], None]},
            "xticks": {
                "default": None,
                "type": Union[Sequence[Union[float, int]], None],
            },
            "yticks": {
                "default": None,
                "type": Union[Sequence[Union[float, int]], None],
            },
            "xticklabels": {"default": None, "type": Union[Sequence[str], None]},
            "yticklabels": {"default": None, "type": Union[Sequence[str], None]},
            "xtickparams": {"default": None, "type": Dict[str, Any]},
            "ytickparams": {"default": None, "type": Dict[str, Any]},
            "spines": {"default": {}, "type": Dict[str, Any]},
        }
        # Specific Parameters that are based on the number of data sequences
        self._multi_data_params = {
            "color": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "alpha": {
                "default": 1.0,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "label": {"default": None, "type": Union[StrSequence, str]},
            "zorder": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
        }
        self._multi_params_structure = {}
        self._init_params = {
            **self._single_data_params,
            **self._multi_data_params,
        }
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def _set_init_params(self, **kwargs):
        for param, config in self._init_params.items():
            setattr(self, param, config["default"])
            if param in kwargs and kwargs[param] is not None:
                setter_name = f"set_{param}"
                if hasattr(self, setter_name):
                    if isinstance(kwargs[param], dict):
                        getattr(self, setter_name)(**kwargs[param])
                    else:
                        getattr(self, setter_name)(kwargs[param])
                else:
                    raise ArgumentValueError(f"Parameter '{param}' is not valid.")

    def reset_params(self):
        for param, config in self._init_params.items():
            setattr(self, param, config["default"])
        return self

    def _validate_data(
        self,
        data: Union[NumericSequence, NumericSequences, None],
        name: Literal["x_data", "y_data"],
    ) -> NumericSequences:
        if name == "y_data" and data is None:
            return data
        if is_numeric_sequence(data):
            data = [data]
        validate_numeric_sequences(data, name)
        validate_consistent_len(data, name)
        return data

    def set_title(self, label: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html"""
        self.title = dict(label=label, **kwargs)
        return self

    def set_xlabel(self, xlabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel"""
        self.xlabel = dict(xlabel=xlabel, **kwargs)
        return self

    def set_ylabel(self, ylabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel"""
        self.ylabel = dict(ylabel=ylabel, **kwargs)
        return self

    def set_axes_labels(self, xlabel: str, ylabel: str, **kwargs):
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)
        return self

    def set_legend(
        self,
        show: bool = True,
        labels: Union[Sequence[str], str, None] = None,
        handles: Union[Sequence[Any], None] = None,
        loc: Union[str, int] = "best",
        bbox_to_anchor: Union[Tuple[float, float], None] = None,
        ncol: int = 1,
        fontsize: Union[int, str, None] = None,
        title: Union[str, None] = None,
        title_fontsize: Union[int, str, None] = None,
        frameon: bool = True,
        fancybox: bool = True,
        framealpha: float = 0.8,
        edgecolor: Union[str, None] = None,
        facecolor: Union[str, None] = "inherit",
        **kwargs,
    ):
        validate_type(show, bool, "show")
        validate_type(frameon, bool, "frameon")
        validate_type(fancybox, bool, "fancybox")
        validate_type(ncol, int, "ncol")
        validate_type(framealpha, NUMERIC_TYPES, "framealpha")
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            else:
                validate_type(labels, SEQUENCE_TYPES, "labels")
                validate_sequence_type(labels, str, "labels")
        if handles is not None:
            validate_type(handles, (list, tuple), "handles")
        if bbox_to_anchor is not None:
            validate_type(bbox_to_anchor, tuple, "bbox_to_anchor")
            if len(bbox_to_anchor) not in [2, 4]:
                raise ArgumentStructureError(
                    "'bbox_to_anchor' must be a tuple of 2 or 4 floats"
                )
            validate_sequence_type(bbox_to_anchor, NUMERIC_TYPES, "bbox_to_anchor")
        if fontsize is not None:
            validate_type(fontsize, (int, str), "fontsize")
        if title is not None:
            validate_type(title, str, "title")
        if title_fontsize is not None:
            validate_type(title_fontsize, (int, str), "title_fontsize")
        if edgecolor is not None:
            validate_type(edgecolor, str, "edgecolor")
        if facecolor is not None:
            validate_type(facecolor, str, "facecolor")
        if "kwargs" not in kwargs:
            legend_kwargs = {
                "loc": loc,
                "ncol": ncol,
                "frameon": frameon,
                "fancybox": fancybox,
                "framealpha": framealpha,
            }
            if labels is not None:
                legend_kwargs["labels"] = labels
            if handles is not None:
                legend_kwargs["handles"] = handles
            if bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
            if fontsize is not None:
                legend_kwargs["fontsize"] = fontsize
            if title is not None:
                legend_kwargs["title"] = title
            if title_fontsize is not None:
                legend_kwargs["title_fontsize"] = title_fontsize
            if edgecolor is not None:
                legend_kwargs["edgecolor"] = edgecolor
            if facecolor is not None:
                legend_kwargs["facecolor"] = facecolor
            legend_kwargs.update(kwargs)
            legend = {"show": show, "kwargs": legend_kwargs}
        else:
            legend = {"show": show, "kwargs": kwargs["kwargs"]}
        self.legend = legend if show else None
        return self

    def set_figsize(self, figsize: Tuple[float, float]):
        if isinstance(figsize, list):
            figsize = tuple(figsize)
        validate_type(figsize, tuple, "figsize")
        validate_sequence_type(figsize, NUMERIC_TYPES, "figsize")
        validate_sequence_length(figsize, 2, "figsize")
        self.figsize = figsize
        return self

    def set_style(self, style: str):
        if style is not None:
            validate_value_in_options(style, plt.style.available, "style")
        self.style = style
        return self

    def set_grid(
        self,
        visible: bool = None,
        which: Literal["major", "minor", "both"] = "major",
        axis: Literal["both", "x", "y"] = "both",
        **kwargs,
    ):
        validate_type(visible, bool, "visible")
        validate_value_in_options(which, ["major", "minor", "both"], "which")
        validate_value_in_options(axis, ["both", "x", "y"], "axis")
        self.grid = dict(visible=visible, which=which, axis=axis, **kwargs)
        return self

    def set_tight_layout(self, tight_layout: bool = False):
        validate_type(tight_layout, bool, "tight_layout")
        self.tight_layout = tight_layout
        return self

    def set_texts(self, texts: Union[Sequence[Dict], Dict]):
        if isinstance(texts, dict):
            self.texts = [texts]
        else:
            validate_type(texts, SEQUENCE_TYPES, "texts")
            validate_sequence_type(texts, dict, "texts")
            self.texts = texts
        return self

    def set_xscale(self, xscale: Scale = None):
        if xscale is not None:
            validate_value_in_options(
                xscale, ["linear", "log", "symlog", "logit"], "xscale"
            )
        self.xscale = xscale
        return self

    def set_yscale(self, yscale: Scale = None):
        if yscale is not None:
            validate_value_in_options(
                yscale, ["linear", "log", "symlog", "logit"], "yscale"
            )
        self.yscale = yscale
        return self

    def set_scales(
        self,
        xscale: Scale = None,
        yscale: Scale = None,
    ):
        self.set_xscale(xscale)
        self.set_yscale(yscale)
        return self

    def set_background(self, background: Color):
        validate_type(background, (str, tuple), "background")
        if isinstance(background, tuple):
            validate_sequence_type(background, NUMERIC_TYPES, "background")
            validate_sequence_length(background, (3, 4), "background")
        self.background = background
        return self

    def set_figure_background(self, figure_background: Color):
        validate_type(figure_background, (str, tuple), "figure_background")
        if isinstance(figure_background, tuple):
            validate_sequence_type(
                figure_background, NUMERIC_TYPES, "figure_background"
            )
            validate_sequence_length(figure_background, (3, 4), "figure_background")
        self.figure_background = figure_background
        return self

    def set_suptitle(self, t: str, **kwargs):
        validate_type(t, str, "t")
        self.suptitle = dict(t=t, **kwargs)
        return self

    def set_xlim(self, xlim: Union[Tuple[float, float], None]):
        if xlim is not None:
            validate_type(xlim, (list, tuple), "xlim")
            validate_sequence_length(xlim, 2, "xlim")
            validate_sequence_type(xlim, (*NUMERIC_TYPES, type(None)), "xlim")
            self.xlim = xlim
        return self

    def set_ylim(self, ylim: Union[Tuple[float, float], None]):
        if ylim is not None:
            validate_type(ylim, (list, tuple), "ylim")
            validate_sequence_length(ylim, 2, "ylim")
            validate_sequence_type(ylim, (*NUMERIC_TYPES, type(None)), "ylim")
            self.ylim = ylim
        return self

    def set_limits(
        self,
        xlim: Union[Tuple[float, float], None] = None,
        ylim: Union[Tuple[float, float], None] = None,
    ):
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        return self

    def set_xticks(self, xticks: Union[Sequence[Union[float, int]], None]):
        if xticks is not None:
            validate_type(xticks, SEQUENCE_TYPES, "xticks")
            validate_sequence_type(xticks, NUMERIC_TYPES, "xticks")
            self.xticks = xticks
        return self

    def set_yticks(self, yticks: Union[Sequence[Union[float, int]], None]):
        if yticks is not None:
            validate_type(yticks, SEQUENCE_TYPES, "yticks")
            validate_sequence_type(yticks, NUMERIC_TYPES, "yticks")
            self.yticks = yticks
        return self

    def set_ticks(
        self,
        xticks: Union[Sequence[Union[float, int]], None] = None,
        yticks: Union[Sequence[Union[float, int]], None] = None,
    ):
        self.set_xticks(xticks)
        self.set_yticks(yticks)
        return self

    def set_xticklabels(
        self, xticklabels: Union[Sequence[Union[str, float, int]], None]
    ):
        if xticklabels is not None:
            validate_type(xticklabels, SEQUENCE_TYPES, "xticklabels")
            validate_sequence_type(xticklabels, (str, *NUMERIC_TYPES), "xticklabels")
            self.xticklabels = xticklabels
        return self

    def set_yticklabels(
        self, yticklabels: Union[Sequence[Union[str, float, int]], None]
    ):
        if yticklabels is not None:
            validate_type(yticklabels, SEQUENCE_TYPES, "yticklabels")
            validate_sequence_type(yticklabels, (str, *NUMERIC_TYPES), "yticklabels")
            self.yticklabels = yticklabels
        return self

    def set_ticklabels(
        self,
        xticklabels: Union[Sequence[Union[str, float, int]], None] = None,
        yticklabels: Union[Sequence[Union[str, float, int]], None] = None,
    ):
        self.set_xticklabels(xticklabels)
        self.set_yticklabels(yticklabels)
        return self

    def set_xtickparams(self, xtickparams: Dict[str, Any] = None):
        if xtickparams is not None:
            validate_type(xtickparams, dict, "xtickparams")
            self.xtickparams = xtickparams
        return self

    def set_ytickparams(self, ytickparams: Dict[str, Any] = None):
        if ytickparams is not None:
            validate_type(ytickparams, dict, "ytickparams")
            self.ytickparams = ytickparams
        return self

    def set_tickparams(
        self,
        xtickparams: Dict[str, Any] = None,
        ytickparams: Dict[str, Any] = None,
    ):
        self.set_xtickparams(xtickparams)
        self.set_ytickparams(ytickparams)
        return self

    def _spine_params(
        self,
        visible: Union[bool, Dict[str, bool]] = True,
        position: Union[Dict[str, Union[Tuple[float, float], str]], None] = None,
        color: Union[Color, Dict[str, Color]] = None,
        linewidth: Union[float, Dict[str, float]] = None,
        linestyle: Union[str, Dict[str, str]] = None,
        alpha: Union[float, Dict[str, float]] = None,
        bounds: Union[Tuple[float, float], Dict[str, Tuple[float, float]]] = None,
        capstyle: Union[Literal["butt", "round", "projecting"], Dict[str, str]] = None,
    ):
        return {
            "visible": visible,
            "position": position,
            "color": color,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "alpha": alpha,
            "bounds": bounds,
            "capstyle": capstyle,
        }

    def set_spines(
        self,
        left: Dict[str, Any] = None,
        right: Dict[str, Any] = None,
        bottom: Dict[str, Any] = None,
        top: Dict[str, Any] = None,
    ):
        self.spines = {
            "left": self._spine_params(**left) if left is not None else None,
            "right": self._spine_params(**right) if right is not None else None,
            "bottom": self._spine_params(**bottom) if bottom is not None else None,
            "top": self._spine_params(**top) if top is not None else None,
        }
        return self

    def set_color(self, color: Union[ColorSequences, ColorSequence, Color]):
        if self._multi_data:
            if is_color_sequences(color):
                validate_consistent_len(color, "color")
                validate_sequence_length(color, self._n_sequences, "color")
                validate_sequence_length(color[0], self.data_size, "color[0]")
                self.color = color
                self._multi_params_structure["color"] = "sequences"
                return self
            elif is_color_sequence(color):
                validate_sequence_length(color, self._n_sequences, "color")
                self.color = color
                self._multi_params_structure["color"] = "sequence"
                return self
            elif is_color(color):
                self.color = color
                self._multi_params_structure["color"] = "value"
                return self
        else:
            if is_color_sequence(color):
                validate_sequence_length(color, self.data_size, "color")
                self.color = color
                self._multi_params_structure["color"] = "sequence"
                return self
            validate_color(color)
            self.color = color
            self._multi_params_structure["color"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid color, must be a color, sequence of colors, or sequences of colors."
        )

    def set_alpha(self, alpha: Union[Sequence[float], float]):
        if is_numeric(alpha):
            validate_value_in_range(alpha, 0, 1, "alpha")
        if self._multi_data:
            if is_numeric_sequences(alpha):
                validate_consistent_len(alpha, "alpha")
                validate_sequence_length(alpha[0], self.data_size, "alpha[0]")
                for seq in alpha:
                    for val in seq:
                        validate_value_in_range(val, 0, 1, "alpha")
                self.alpha = alpha
                self._multi_params_structure["alpha"] = "sequences"
                return self
            elif is_numeric_sequence(alpha):
                validate_sequence_length(alpha, self._n_sequences, "alpha")
                for val in alpha:
                    validate_value_in_range(val, 0, 1, "alpha")
                self.alpha = alpha
                self._multi_params_structure["alpha"] = "sequence"
                return self
            elif is_numeric(alpha):
                self.alpha = alpha
                self._multi_params_structure["alpha"] = "value"
                return self
        else:
            if is_numeric_sequence(alpha):
                validate_sequence_length(alpha, self.data_size, "alpha")
                self.alpha = alpha
                self._multi_params_structure["alpha"] = "sequence"
                return self
            validate_numeric(alpha, "alpha")
            self.alpha = alpha
            self._multi_params_structure["alpha"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid alpha, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def set_label(self, labels: Union[Sequence[str], str]):
        if self._multi_data and is_str_sequence(labels):
            validate_sequence_length(labels, self._n_sequences, "labels")
            self.label = labels
            self._multi_params_structure["label"] = "sequence"
            return self
        if isinstance(labels, str):
            self.label = labels
            self._multi_params_structure["label"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid label, must be a string or sequence of strings."
        )

    def set_zorder(self, zorder: Union[NumericSequences, NumericSequence, NumericType]):
        if self._multi_data:
            if is_numeric_sequences(zorder):
                validate_consistent_len(zorder, "zorder")
                validate_sequence_length(zorder[0], self.data_size, "zorder[0]")
                self.zorder = zorder
                self._multi_params_structure["zorder"] = "sequences"
                return self
            elif is_numeric_sequence(zorder):
                validate_sequence_length(zorder, self._n_sequences, "zorder")
                self.zorder = zorder
                self._multi_params_structure["zorder"] = "sequence"
                return self
            elif is_numeric(zorder):
                self.zorder = zorder
                self._multi_params_structure["zorder"] = "value"
                return self
        else:
            if is_numeric_sequence(zorder):
                validate_sequence_length(zorder, self.data_size, "zorder")
                self.zorder = zorder
                self._multi_params_structure["zorder"] = "sequence"
                return self
            validate_numeric(zorder, "zorder")
            self.zorder = zorder
            self._multi_params_structure["zorder"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid zorder, must be a numeric value, sequence of numbers, or sequences of numbers."
        )

    def prepare_draw(self):
        if self.figure is not None or self.ax is not None:
            self.clear()
        if self.style is not None:
            self._default_style = plt.rcParams.copy()
            plt.style.use(self.style)
        if not self.ax and not self.figure:
            self.figure, self.ax = plt.subplots(figsize=self.figsize)
        if self.grid is not None and self.grid["visible"]:
            self.ax.grid(**self.grid)

    def _apply_common_properties(self):
        if self.title:
            self.ax.set_title(**self.title)
        if self.xlabel:
            self.ax.set_xlabel(**self.xlabel)
            if "color" in self.xlabel:
                self.ax.tick_params(axis="x", colors=self.xlabel["color"])
        if self.ylabel:
            self.ax.set_ylabel(**self.ylabel)
            if "color" in self.ylabel:
                self.ax.tick_params(axis="y", colors=self.ylabel["color"])
        if self.xscale:
            self.ax.set_xscale(self.xscale)
        if self.yscale:
            self.ax.set_yscale(self.yscale)
        if self.texts is not None:
            for text in self.texts:
                self.ax.text(**text)
        if self.legend is not None and self.legend["show"]:
            self.ax.legend(**self.legend["kwargs"])
        if self.background:
            self.ax.set_facecolor(self.background)
        if self.figure_background:
            self.figure.set_facecolor(self.figure_background)
        if self.suptitle:
            self.figure.suptitle(**self.suptitle)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        if self.xticks is not None:
            self.ax.set_xticks(self.xticks)
        if self.yticks is not None:
            self.ax.set_yticks(self.yticks)
        if self.xticklabels is not None:
            self.ax.set_xticklabels(self.xticklabels)
        if self.yticklabels is not None:
            self.ax.set_yticklabels(self.yticklabels)
        if self.xtickparams is not None:
            self.ax.tick_params(axis="x", **self.xtickparams)
        if self.ytickparams is not None:
            self.ax.tick_params(axis="y", **self.ytickparams)
        if self.spines:
            for spine, spine_params in self.spines.items():
                if spine_params is not None:
                    for param, values in spine_params.items():
                        if values is not None:
                            if param == "visible":
                                self.ax.spines[spine].set_visible(values)
                            elif param == "position":
                                if isinstance(values, str):
                                    self.ax.spines[spine].set_position(values)
                                else:
                                    self.ax.spines[spine].set_position(("data", values))
                            elif param == "color":
                                self.ax.spines[spine].set_color(values)
                            elif param == "linewidth":
                                self.ax.spines[spine].set_linewidth(values)
                            elif param == "linestyle":
                                self.ax.spines[spine].set_linestyle(values)
                            elif param == "alpha":
                                self.ax.spines[spine].set_alpha(values)
                            elif param == "bounds":
                                self.ax.spines[spine].set_bounds(*values)
                            elif param == "capstyle":
                                self.ax.spines[spine].set_capstyle(values)

    def _finalize_draw(self, show: bool = True):
        if self.tight_layout:
            plt.tight_layout()
        if show:
            plt.show()
        if self.style is not None:
            plt.rcParams.update(self._default_style)
        return self.ax

    @abstractmethod
    def _create_plot(self):
        raise NotImplementedError

    def draw(self, show: bool = True):
        self.prepare_draw()
        try:
            self._create_plot()
        except Exception as e:
            raise PlotterException(f"Error while creating plot: {e}")
        self._apply_common_properties()
        return self._finalize_draw(show)

    def save_plot(
        self,
        file_path: Path,
        dpi: int = 300,
        bbox_inches: str = "tight",
        draw: bool = False,
    ):
        if self.figure or draw:
            if self.figure is None and draw:
                self.draw()
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise PlotterException(f"Error while saving scatter plot: {e}")
        else:
            raise PlotterException("Plot not drawn yet. Call draw() before saving.")
        return self

    def clear(self):
        if self.figure or self.ax:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self

    def _to_serializable(self, value: Any) -> Any:
        if value is None:
            return None
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, ndarray):
            return value.tolist()
        elif isinstance(value, Series):
            return value.to_list()
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, Colormap):
            return value.name
        elif isinstance(value, Normalize):
            return value.__class__.__name__.lower()
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, MarkerStyle):
            marker = dict(
                marker=value.get_marker(),
                fillstyle=value.get_fillstyle(),
                capstyle=value.get_capstyle(),
                joinstyle=value.get_joinstyle(),
            )
            return marker

        return value

    def save_plotter(self, file_path: Union[str, Path], data: bool = True) -> None:
        init_params = {}
        for param, config in self._init_params.items():
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        if data:
            init_params["x_data"] = self._to_serializable(self.x_data)
            init_params["y_data"] = self._to_serializable(self.y_data)
        with open(file_path, "w") as f:
            json.dump(init_params, f, indent=4)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Plotter":
        with open(file_path, "r") as f:
            params = json.load(f)
        x_data = params.pop("x_data") if "x_data" in params else []
        y_data = params.pop("y_data") if "y_data" in params else []
        if "xlim" in params and params["xlim"] is not None:
            params["xlim"] = tuple(params["xlim"])
        if "ylim" in params and params["ylim"] is not None:
            params["ylim"] = tuple(params["ylim"])
        if "center" in params and params["center"] is not None:
            params["center"] = tuple(params["center"])
        return cls(x_data=x_data, y_data=y_data, **params)
