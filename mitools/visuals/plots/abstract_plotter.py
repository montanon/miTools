import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)


class Plotter(ABC):
    """Abstract base class for Matplotlib plot wrappers."""

    def __init__(self, **kwargs):
        self._init_params = {
            "title": {"default": "", "type": Text},
            "xlabel": {"default": "", "type": Text},
            "ylabel": {"default": "", "type": Text},
            "figsize": {"default": (21, 14), "type": Tuple[float, float]},
            "style": {"default": "dark_background", "type": str},
            "grid": {"default": None, "type": Dict[str, Any]},
            "tight_layout": {"default": False, "type": bool},
            "texts": {"default": None, "type": Union[Sequence[Text], Text]},
            "xscale": {
                "default": None,
                "type": Literal["linear", "log", "symlog", "logit"],
            },
            "yscale": {
                "default": None,
                "type": Literal["linear", "log", "symlog", "logit"],
            },
            "background": {"default": None, "type": str},
            "figure_background": {"default": None, "type": str},
            "suptitle": {"default": None, "type": Text},
            "xlim": {"default": None, "type": Union[Tuple[float, float], None]},
            "ylim": {"default": None, "type": Union[Tuple[float, float], None]},
            "x_ticks": {
                "default": None,
                "type": Union[Sequence[Union[float, int]], None],
            },
            "y_ticks": {
                "default": None,
                "type": Union[Sequence[Union[float, int]], None],
            },
            "x_tick_labels": {"default": None, "type": Union[Sequence[str], None]},
            "y_tick_labels": {"default": None, "type": Union[Sequence[str], None]},
        }

        # Initialize attributes with default values
        for param, config in self._init_params.items():
            setattr(self, param, config["default"])
            if param in kwargs:
                setter_name = f"set_{param}"
                if hasattr(self, setter_name):
                    getattr(self, setter_name)(kwargs[param])
                else:
                    if param in ["xscale", "yscale"]:
                        self.set_scales(**{param: kwargs[param]})
                    elif param in ["xlim", "ylim"]:
                        self.set_ax_limits(**{param: kwargs[param]})
                    elif param in ["x_ticks", "y_ticks"]:
                        self.set_ticks(**{param: kwargs[param]})
                    elif param in ["x_tick_labels", "y_tick_labels"]:
                        self.set_tick_labels(**{param: kwargs[param]})
                    else:
                        raise ArgumentValueError(f"Parameter '{param}' is not valid.")

        self.figure: Figure = None
        self.ax: Axes = None

    def set_title(self, title: str, **kwargs):
        if isinstance(title, dict):
            kwargs = {k: v for k, v in title.items() if k not in ["label"]}
            title = title["label"]
        self.title = dict(label=title, **kwargs)
        return self

    def set_xlabel(self, xlabel: str, **kwargs):
        if isinstance(xlabel, dict):
            kwargs = {k: v for k, v in xlabel.items() if k not in ["xlabel"]}
            xlabel = xlabel["xlabel"]
        self.xlabel = dict(xlabel=xlabel, **kwargs)
        return self

    def set_ylabel(self, ylabel: str, **kwargs):
        if isinstance(ylabel, dict):
            kwargs = {k: v for k, v in ylabel.items() if k not in ["ylabel"]}
            ylabel = ylabel["ylabel"]
        self.ylabel = dict(ylabel=ylabel, **kwargs)
        return self

    def set_axes_labels(self, xlabel: str, ylabel: str, **kwargs):
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)
        return self

    def set_style(self, style: str):
        if style in plt.style.available or style is None:
            self.style = style
        else:
            raise ArgumentValueError(
                f"Style '{style}' is not available in Matplotlib styles: {plt.style.available}."
            )
        return self

    def set_grid(
        self,
        visible: Union[bool, Dict] = None,
        which: Literal["major", "minor", "both"] = "major",
        axis: Literal["both", "x", "y"] = "both",
        **kwargs,
    ):
        if isinstance(visible, dict):
            kwargs = {
                k: v
                for k, v in visible.items()
                if k not in ["visible", "which", "axis"]
            }
            which = visible.get("which", "major")
            axis = visible.get("axis", "both")
            visible = visible.get("visible", True)
        if visible not in [True, False]:
            raise ArgumentTypeError("visible must be a bool.")
        if which not in ["major", "minor", "both"]:
            raise ArgumentValueError("which must be one of 'major', 'minor', 'both'.")
        if axis not in ["both", "x", "y"]:
            raise ArgumentValueError("axis must be one of 'both', 'x', 'y'.")
        self.grid = dict(visible=visible, which=which, axis=axis, **kwargs)
        return self

    def set_figsize(self, figsize: Tuple[float, float]):
        if isinstance(figsize, list):
            figsize = tuple(figsize)
        if isinstance(figsize, tuple) and all(
            isinstance(val, (float, int)) for val in figsize
        ):
            self.figsize = figsize
        else:
            raise ArgumentTypeError("figsize must be a tuple of floats or ints.")
        return self

    def set_scales(self, xscale: str = None, yscale: str = None):
        _scales = ["linear", "log", "symlog", "logit"]
        if xscale is not None:
            if xscale not in _scales:
                raise ArgumentValueError(f"'xscale'={xscale} must be one of {_scales}")
            self.xscale = xscale
        if yscale is not None:
            if yscale not in _scales:
                raise ArgumentValueError(f"'yscale'={yscale} must be one of {_scales}")
            self.yscale = yscale
        return self

    def set_ax_limits(
        self, xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None
    ):
        if xlim is not None:
            if not isinstance(xlim, (list, tuple)) or len(xlim) != 2:
                raise ArgumentStructureError(
                    "xlim must be a tuple or list of two ints or floats (min, max)."
                )
            self.xlim = xlim
        if ylim is not None:
            if not isinstance(ylim, (list, tuple)) or len(ylim) != 2:
                raise ArgumentStructureError(
                    "ylim must be a tuple or list of two ints or floats (min, max)."
                )
            self.ylim = ylim
        return self

    def set_ticks(
        self,
        x_ticks: Sequence[Union[float, int]] = None,
        y_ticks: Sequence[Union[float, int]] = None,
    ):
        if x_ticks is not None:
            if not isinstance(x_ticks, (list, tuple)):
                raise ArgumentTypeError("x_ticks must be array-like")
            self.x_ticks = x_ticks
        if y_ticks is not None:
            if not isinstance(y_ticks, (list, tuple)):
                raise ArgumentTypeError("y_ticks must be array-like")
            self.y_ticks = y_ticks
        return self

    def set_tick_labels(
        self, x_tick_labels: Sequence[str] = None, y_tick_labels: Sequence[str] = None
    ):
        if x_tick_labels is not None:
            if not isinstance(x_tick_labels, (list, tuple)):
                raise ArgumentTypeError("x_tick_labels must be array-like")
            self.x_tick_labels = x_tick_labels
        if y_tick_labels is not None:
            if not isinstance(y_tick_labels, (list, tuple)):
                raise ArgumentTypeError("y_tick_labels must be array-like")
            self.y_tick_labels = y_tick_labels
        return self

    def set_background(self, background: str):
        self.background = background
        return self

    def set_figure_background(self, figure_background: str):
        self.figure_background = figure_background
        return self

    def set_tight_layout(self, tight_layout: bool = False):
        if tight_layout not in [True, False]:
            raise ArgumentValueError("tight_layout must be a bool.")
        self.tight_layout = tight_layout
        return self

    @abstractmethod
    def draw(self, show: bool = False):
        """Draw the plot. Must be implemented by subclasses."""
        pass

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
                raise Exception(f"Error while saving plot: {e}")
        else:
            raise Exception("Plot not drawn yet. Call draw() before saving.")
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
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, Path):
            return str(value)
        return value

    def save_plotter(self, file_path: Union[str, Path]) -> None:
        init_params = {}
        for param, config in self._init_params.items():
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        with open(file_path, "w") as f:
            json.dump(init_params, f, indent=4)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Plotter":
        with open(file_path, "r") as f:
            params = json.load(f)
        return cls(**params)
