from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text

from mitools.exceptions import ArgumentValueError

Color = Union[str, Sequence[float]]
Marker = Union[str, int, Path]
Cmap = Union[str, Colormap]
Norm = Union[str, Normalize]
EdgeColor = Union[Literal["face", "none", None], Color, Sequence[Color]]
FaceColor = Union[Color, Sequence[Color]]
LineStyle = Literal["solid", "dashed", "dashdot", "dotted", "-", "--", "-.", ":"]


class ScatterPlotter(ABC):
    def __init__(self, x_data: Any, y_data: Any):
        self.x_data = self._validate_data(x_data, "x_data")
        self.y_data = self._validate_data(y_data, "y_data")

        if len(self.x_data) != len(self.y_data):
            raise ArgumentValueError(
                f"'x_data' and 'y_data' must be of the same length, {len(x_data)} != {len(y_data)}."
            )
        self.size_data: Union[Sequence[float], float] = None
        self.color: Union[Sequence[Color], Color] = None
        self.marker: Union[Sequence[Marker], Marker] = "o"
        self.color_map: Cmap = None
        self.normalize: Norm = None
        self.vmin: float = None
        self.vmax: float = None
        self.alphas: Union[Sequence[float], float] = 1.0
        self.linewidths: Union[Sequence[float], float] = None
        self.linestyles: Union[Sequence[LineStyle], LineStyle] = None
        self.edgecolors: EdgeColor = None
        self.facecolors: FaceColor = None
        self.plot_non_finite: bool = False
        self.labels: Union[Sequence[str], str] = None
        self.zorders: Union[Sequence[float], float] = None
        self.figsize: Tuple[float, float] = (21, 14)
        self.title: Text = ""
        self.xlabel: Text = ""
        self.ylabel: Text = ""
        self.style: str = "default"
        self.hover: bool = False
        self.figure: Figure = None
        self.ax: Axes = None

    @abstractmethod
    def _validate_data(self, data, name):
        pass

    def set_title(self, title: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html"""
        self.title = Text(text=title, **kwargs)
        return self

    def set_xlabel(self, xlabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel"""
        self.xlabel = Text(text=xlabel, **kwargs)
        return self

    def set_ylabel(self, ylabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel"""
        self.ylabel = Text(text=ylabel, **kwargs)
        return self

    def set_axes_labels(self, xlabel: str, ylabel: str, **kwargs):
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)
        return self

    def set_style(self, style):
        if style in plt.style.available:
            self.style = style
        else:
            raise ValueError(f"Style '{style}' is not available in Matplotlib.")
        return self

    def set_color(self, color):
        self.color = color
        return self

    def set_colormap(self, cmap):
        self.color_map = cmap
        return self

    def set_alpha(self, alpha):
        if 0.0 <= alpha <= 1.0:
            self.alpha = alpha
        else:
            raise ValueError("Alpha must be between 0.0 and 1.0.")
        return self

    def set_size(self, size_data):
        if isinstance(size_data, (list, tuple, np.ndarray, pd.Series, float, int)):
            self.size_data = size_data
        else:
            raise TypeError("size_data must be array-like or a single numeric value.")
        return self

    def set_edgecolor(self, edgecolor):
        self.edgecolor = edgecolor
        return self

    def set_marker(self, marker):
        self.marker = marker
        return self

    def set_labels(self, labels):
        """Set labels for each point (for interactive plots)."""
        self.labels = self._validate_data(labels, "labels")
        if len(self.labels) != len(self.x_data):
            raise ValueError("labels must be of the same length as x_data and y_data.")
        return self

    def enable_hover(self, hover=True):
        """Enable or disable hover interaction."""
        self.hover = hover
        return self

    def set_color_data(self, color_data):
        """Set data for coloring the scatter points."""
        self.color_data = self._validate_data(color_data, "color_data")
        if len(self.color_data) != len(self.x_data):
            raise ValueError(
                "color_data must be of the same length as x_data and y_data."
            )
        return self

    def set_size_data(self, size_data):
        """Set data for sizing the scatter points."""
        return self.set_size(size_data)

    def set_figure_size(self, width, height):
        """Set the size of the figure."""
        if self.figure:
            self.figure.set_size_inches(width, height)
        else:
            self.figure = plt.figure(figsize=(width, height))
        return self

    def set_limits(self, xlim=None, ylim=None):
        """Set the limits of the axes."""
        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        return self

    def set_ticks(self, x_ticks=None, y_ticks=None):
        """Set custom ticks for the axes."""
        if x_ticks is not None:
            self.ax.set_xticks(x_ticks)
        if y_ticks is not None:
            self.ax.set_yticks(y_ticks)
        return self

    def set_tick_labels(self, x_tick_labels=None, y_tick_labels=None):
        """Set custom tick labels for the axes."""
        if x_tick_labels is not None:
            self.ax.set_xticklabels(x_tick_labels)
        if y_tick_labels is not None:
            self.ax.set_yticklabels(y_tick_labels)
        return self

    def set_grid(self, grid=True):
        """Enable or disable the grid."""
        self.ax.grid(grid)
        return self

    def set_legend(self, legend=True):
        if legend:
            self.ax.legend()
        return self

    def add_text(self, x, y, text, **kwargs):
        self.ax.text(x, y, text, **kwargs)
        return self

    def add_line(self, x_data, y_data, **kwargs):
        self.ax.plot(x_data, y_data, **kwargs)
        return self

    def set_log_scale(self, x_log=False, y_log=False):
        if x_log:
            self.ax.set_xscale("log")
        if y_log:
            self.ax.set_yscale("log")
        return self

    def invert_axes(self, x_invert=False, y_invert=False):
        if x_invert:
            self.ax.invert_xaxis()
        if y_invert:
            self.ax.invert_yaxis()
        return self

    def set_aspect_ratio(self, aspect="auto"):
        self.ax.set_aspect(aspect)
        return self

    def apply_theme(self, theme):
        if theme == "dark":
            self.set_style("dark_background")
            self.set_color("cyan")
            self.set_edgecolor("white")
        elif theme == "light":
            self.set_style("default")
            self.set_color("blue")
            self.set_edgecolor("black")
        else:
            raise ValueError(f"Theme '{theme}' is not recognized.")
        return self

    def draw(self):
        plt.style.use(self.style)
        if not self.figure or not self.ax:
            self.figure, self.ax = plt.subplots()

        scatter_kwargs = {
            "x": self.x_data,
            "y": self.y_data,
            "c": self.color_data if self.color_data is not None else self.color,
            "cmap": self.color_map,
            "alpha": self.alpha,
            "edgecolor": self.edgecolor,
            "marker": self.marker,
        }

        if self.size_data is not None:
            scatter_kwargs["s"] = self.size_data

        try:
            sc = self.ax.scatter(**scatter_kwargs)
        except Exception as e:
            self.logger.error(f"Error while creating scatter plot: {e}")
            raise

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        if self.color_data is not None and self.color_map is not None:
            cbar = self.figure.colorbar(sc, ax=self.ax)
            cbar.set_label("Color Scale")

        if self.hover and self.labels is not None:
            # Implement hover functionality
            annot = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            annot.set_visible(False)

            def update_annot(ind):
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = "{}".format(" ".join([str(self.labels[n]) for n in ind["ind"]]))
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.4)

            def hover_event(event):
                vis = annot.get_visible()
                if event.inaxes == self.ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        self.figure.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            self.figure.canvas.draw_idle()

            self.figure.canvas.mpl_connect("motion_notify_event", hover_event)

        plt.show()
        return self

    def save(self, filename, dpi=300, bbox_inches="tight"):
        if self.figure:
            try:
                self.figure.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                self.logger.error(f"Error saving figure: {e}")
                raise
        else:
            raise RuntimeError("Plot not drawn yet. Call draw() before saving.")
        return self

    def clear(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self
