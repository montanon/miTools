import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from mitools.exceptions import ArgumentValueError
from mitools.visuals.plots.matplotlib_typing import Color
from mitools.visuals.plots.plot_params import FigureParams, PlotParams
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import validate_type


class PlotComposerException(Exception):
    pass


def _get_plotter_types() -> Dict[str, Type[Plotter]]:
    plotter_types = {}
    plots_module = importlib.import_module("mitools.visuals.plots")
    for module_name in [
        "line_plotter",
        "scatter_plotter",
        "bar_plotter",
        "box_plotter",
        "histogram_plotter",
        "pie_plotter",
        "distribution_plotter",
    ]:
        try:
            importlib.import_module(f"mitools.visuals.plots.{module_name}")
        except ImportError:
            continue
    for name, obj in inspect.getmembers(plots_module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, Plotter)
            and obj != Plotter
            and not inspect.isabstract(obj)
        ):
            plotter_types[obj.__name__] = obj
    return plotter_types


class PlotComposer(PlotParams):
    def __init__(
        self, plotters: Optional[Sequence[Plotter]] = None, ax: Axes = None, **kwargs
    ):
        super().__init__(ax=ax, **kwargs)
        self.plotters: List[Plotter] = []
        if plotters:
            self.add_plotters(plotters)

    def add_plotter(self, plotter: Plotter) -> "PlotComposer":
        validate_type(plotter, Plotter, "plotter")
        self.plotters.append(plotter)
        return self

    def add_plotters(self, plotters: Sequence[Plotter]) -> "PlotComposer":
        validate_type(plotters, (list, tuple), "plotters")
        for plotter in plotters:
            self.add_plotter(plotter)
        return self

    def draw(self, show: bool = False, clear: bool = False) -> Axes:
        self._prepare_draw(clear=clear)
        try:
            for plotter in self.plotters:
                plotter.ax = self.ax
                plotter.figure = self.figure
                plotter._create_plot()
                plotter._apply_common_properties()
            self._apply_common_properties()
            self._finalize_draw(show=show)
            return self.ax
        except Exception as e:
            raise PlotComposerException(f"Error while creating composition: {e}")

    def save_plot(
        self,
        file_path: Union[str, Path],
        dpi: int = 300,
        bbox_inches: str = "tight",
        draw: bool = False,
    ) -> "PlotComposer":
        if self.figure or draw:
            if self.figure is None and draw:
                self.draw()
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise PlotComposerException(f"Error while saving composition: {e}")
        else:
            raise PlotComposerException(
                "Plot not drawn yet. Call draw() before saving."
            )
        return self

    def save_composer(
        self, file_path: Union[str, Path], return_json: bool = False
    ) -> Union[None, Dict]:
        init_params = {}
        for param, config in self._init_params.items():
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        composer_data = {"params": init_params, "plotters": []}
        for plotter in self.plotters:
            init_params = plotter.save_plotter(file_path, return_json=True)
            composer_data["plotters"].append(
                {"type": plotter.__class__.__name__, "params": init_params}
            )
        if return_json:
            return composer_data
        else:
            with open(file_path, "w") as f:
                json.dump(composer_data, f, indent=4)

    @classmethod
    def from_json(
        cls,
        file_path: Union[str, Path],
    ) -> "PlotComposer":
        with open(file_path, "r") as f:
            composer_data = json.load(f)
        plotter_types = _get_plotter_types()
        composer = cls(plotters=[], **composer_data["params"])
        for plotter_info in composer_data["plotters"]:
            plotter_type_name = plotter_info["type"]
            if plotter_type_name not in plotter_types:
                raise PlotComposerException(
                    f"Unknown plotter type: {plotter_type_name}. "
                    f"Available types: {list(plotter_types.keys())}"
                )
            plotter_type = plotter_types[plotter_type_name]
            plotter = plotter_type(**plotter_info["params"])
            composer.add_plotter(plotter)

        return composer


class AxesComposerException(Exception):
    pass


class AxesComposer(FigureParams):
    def __init__(
        self,
        axes: Sequence[Axes],
        plots: Sequence[Union[PlotComposer, Plotter]],
        **kwargs,
    ):
        validate_type(axes, (list, tuple, np.ndarray), "axes")
        validate_type(plots, (list, tuple), "plots")
        if isinstance(axes, np.ndarray):
            axes = axes.flat
        if len(axes) != len(plots):
            raise AxesComposerException(
                f"Number of axes ({len(axes)}) must match number of plots ({len(plots)})"
            )
        self.axes = list(axes)
        self.plots = list(plots)
        super().__init__(figure=axes[0].figure, **kwargs)

    def add_plot(self, ax: Axes, plot: Union[PlotComposer, Plotter]) -> "AxesComposer":
        validate_type(ax, Axes, "ax")
        validate_type(plot, (PlotComposer, Plotter), "plot")
        self.axes.append(ax)
        self.plots.append(plot)
        return self

    def draw(self, show: bool = False, clear: bool = False) -> List[Axes]:
        try:
            self._prepare_draw(clear=clear)
            for ax, plot in zip(self.axes, self.plots):
                plot.ax = ax
                plot.figure = self.figure
                plot.draw(show=False, clear=clear)
            if show:
                self.figure.show()
            self._finalize_draw(show=show)
            return self.axes
        except Exception as e:
            raise AxesComposerException(f"Error while creating composition: {e}")

    def save_plot(
        self,
        file_path: Union[str, Path],
        dpi: int = 300,
        bbox_inches: str = "tight",
        draw: bool = False,
    ) -> "AxesComposer":
        if self.figure or draw:
            if self.figure is None and draw:
                self.draw()
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise AxesComposerException(f"Error while saving composition: {e}")
        else:
            raise AxesComposerException(
                "Plot not drawn yet. Call draw() before saving."
            )
        return self

    def save_composer(
        self, file_path: Union[str, Path], return_json: bool = False
    ) -> Union[None, Dict]:
        init_params = {}
        for param, config in self._init_params.items():
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        composer_data = {"params": init_params, "axes": [], "plots": []}
        axes_data = []
        for ax in self.axes:
            x0, y0, width, height = ax.get_position().bounds
            axes_data.append([x0, y0, width, height])
        composer_data["axes"] = axes_data
        plots_data = []
        for plot in self.plots:
            if isinstance(plot, PlotComposer):
                plot_data = plot.save_composer(file_path=None, return_json=True)
                plot_type = "PlotComposer"
            else:
                plot_data = plot.save_plotter(file_path=None, return_json=True)
                plot_type = plot.__class__.__name__
            plots_data.append({"type": plot_type, "data": plot_data})
        composer_data["plots"] = plots_data
        if return_json:
            return composer_data
        else:
            with open(file_path, "w") as f:
                json.dump(composer_data, f, indent=4)

    @classmethod
    def from_json(
        cls,
        file_path: Union[str, Path],
    ) -> "AxesComposer":
        with open(file_path, "r") as f:
            composer_data = json.load(f)
        axes_data = composer_data["axes"]
        figure = plt.figure(figsize=composer_data["params"]["figsize"])
        new_axes = []
        for bbox in axes_data:
            x0, y0, width, height = bbox
            ax = figure.add_axes([x0, y0, width, height])
            new_axes.append(ax)
        axes = new_axes
        plots_info = composer_data["plots"]
        if len(plots_info) != len(axes):
            raise AxesComposerException(
                f"Number of plot objects ({len(plots_info)}) != number of Axes ({len(axes)})"
            )
        plots = []
        for plot_dict in plots_info:
            plot_type = plot_dict["type"]
            plot_data = plot_dict["data"]
            if plot_type == "PlotComposer":
                reconstructed_plot = PlotComposer(**plot_data["params"])
                for child_plotter_info in plot_data["plotters"]:
                    child_plotter_type = child_plotter_info["type"]
                    child_params = child_plotter_info["params"]
                    child_plotter_cls = _get_plotter_types()[child_plotter_type]
                    child_plotter = child_plotter_cls(**child_params)
                    reconstructed_plot.add_plotter(child_plotter)
            else:
                plotter_types = _get_plotter_types()
                if plot_type not in plotter_types:
                    raise AxesComposerException(
                        f"Unknown plot type: {plot_type}. "
                        f"Available: {list(plotter_types.keys())}"
                    )
                plot_cls = plotter_types[plot_type]
                reconstructed_plot = plot_cls(**plot_data)
            plots.append(reconstructed_plot)
        composer = cls(axes=axes, plots=plots)
        return composer
