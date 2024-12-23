import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mitools.exceptions import ArgumentValueError
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


class PlotComposer:
    def __init__(self, plotters: Sequence[Plotter], ax: Axes = None, **kwargs):
        self.ax = ax
        self.figure = None if ax is None else ax.figure
        self.plotters: List[Plotter] = []

        self._composer_params = {
            "figsize": kwargs.get("figsize", (10, 8)),
            "style": kwargs.get("style", None),
            "tight_layout": kwargs.get("tight_layout", False),
            "grid": kwargs.get("grid", None),
        }
        for param, value in self._composer_params.items():
            setattr(self, param, value)
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

    def _prepare_draw(self, clear: bool = False):
        if clear:
            self.clear()
        if self.style is not None:
            self._default_style = plt.rcParams.copy()
            plt.style.use(self.style)
        if not self.ax:
            self.figure, self.ax = plt.subplots(figsize=self.figsize)
        if self.grid is not None and self.grid["visible"]:
            self.ax.grid(**self.grid)

    def _finalize_draw(self, show: bool = False):
        if self.tight_layout:
            plt.tight_layout()
        if show:
            self.figure.show()
        if self.style is not None:
            plt.rcParams.update(self._default_style)
        return self.ax

    def draw(self, show: bool = False, clear: bool = False) -> Axes:
        self._prepare_draw(clear=clear)
        try:
            for plotter in self.plotters:
                plotter.ax = self.ax
                plotter.figure = self.figure
                plotter._create_plot()
                plotter._apply_common_properties()
            self._finalize_draw(show=show)
            return self.ax
        except Exception as e:
            raise PlotComposerException(f"Error while creating composition: {e}")

    def clear(self) -> "PlotComposer":
        if self.figure or self.ax:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self

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

    def save_composer(self, file_path: Union[str, Path]) -> None:
        composer_data = {"params": self._composer_params, "plotters": []}
        for plotter in self.plotters:
            init_params = plotter.save_plotter(file_path, return_json=True)
            composer_data["plotters"].append(
                {"type": plotter.__class__.__name__, "params": init_params}
            )
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
