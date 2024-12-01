import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.text import Text

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots import MarkerStyle, ScatterPlotter


class TestScatterPlotter(unittest.TestCase):
    def setUp(self):
        self.x_data = np.linspace(0, 10, 100)
        self.y_data = np.sin(self.x_data)
        self.valid_params = {
            "title": "Test Plot",
            "xlabel": "X Label",
            "ylabel": "Y Label",
            "size": 100,
            "color": "blue",
            "marker": "o",
            "colormap": "viridis",
            "normalization": "linear",
            "vmin": 0,
            "vmax": 1,
            "alpha": 0.7,
            "linewidth": 1.5,
            "edgecolor": "black",
            "facecolor": "red",
            "label": "Test Label",
            "zorder": 10,
            "plot_non_finite": True,
            "figsize": (10, 8),
            "style": "classic",
            "grid": True,
            "hover": False,
            "tight_layout": True,
            "texts": [{"s": "Test Text", "x": 0.5, "y": 0.5}],
            "xscale": "linear",
            "yscale": "linear",
            "background": "white",
            "figure_background": "gray",
            "suptitle": "Super Title",
            "xlim": (-1, 11),
            "ylim": (-2, 2),
            "x_ticks": np.arange(-1, 11, 1),
            "y_ticks": np.arange(-2, 2, 0.5),
            "x_tick_labels": [str(i) for i in range(-1, 11)],
            "y_tick_labels": [f"{i:.1f}" for i in np.arange(-2, 2, 0.5)],
        }

    def test_initialization_validation(self):
        with self.assertRaises(ArgumentTypeError):
            ScatterPlotter("invalid", self.y_data)
        with self.assertRaises(ArgumentTypeError):
            ScatterPlotter(self.x_data, "invalid")
        with self.assertRaises(ArgumentStructureError):
            ScatterPlotter(self.x_data, self.y_data[:50])

    def test_initialization_vs_setters(self):
        plotter1 = ScatterPlotter(self.x_data, self.y_data, **self.valid_params)
        plotter2 = ScatterPlotter(self.x_data, self.y_data)
        for param, value in self.valid_params.items():
            setter_name = f"set_{param}"
            if hasattr(plotter2, setter_name):
                getattr(plotter2, setter_name)(value)
            else:
                if param in ["xscale", "yscale"]:
                    plotter2.set_scales(**{param: value})
                elif param in ["xlim", "ylim"]:
                    plotter2.set_ax_limits(**{param: value})
                elif param in ["x_ticks", "y_ticks"]:
                    plotter2.set_ticks(**{param: value})
                elif param in ["x_tick_labels", "y_tick_labels"]:
                    plotter2.set_tick_labels(**{param: value})
                else:
                    raise ArgumentValueError(f"Parameter '{param}' is not valid.")
        for param in self.valid_params.keys():
            if param in ["marker"]:
                marker1 = getattr(plotter1, param)
                marker2 = getattr(plotter2, param)
                self.assertEqual(
                    marker1.get_marker(),
                    marker2.get_marker(),
                    f"Parameter {param} differs between init and setter methods",
                )
                self.assertEqual(
                    marker1.get_joinstyle(),
                    marker2.get_joinstyle(),
                    f"Parameter {param} differs between init and setter methods",
                )
                self.assertEqual(
                    marker1.get_marker(),
                    marker2.get_marker(),
                    f"Parameter {param} differs between init and setter methods",
                )
            elif param in ["x_ticks", "y_ticks"]:
                for val1, val2 in zip(
                    getattr(plotter1, param), getattr(plotter2, param)
                ):
                    self.assertEqual(
                        val1,
                        val2,
                        f"Parameter {param} differs between init and setter methods",
                    )
            else:
                self.assertEqual(
                    getattr(plotter1, param),
                    getattr(plotter2, param),
                    f"Parameter {param} differs between init and setter methods",
                )

    def test_color_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        valid_colors = ["red", "blue", (1, 0, 0), (1, 0, 0, 1)]
        for color in valid_colors:
            plotter.set_color(color)
        invalid_colors = [(1, 0), (1, 0, 0, 1, 1), "not_a_color"]
        for color in invalid_colors:
            with self.assertRaises((ArgumentTypeError, ArgumentValueError)):
                plotter.set_color(color)

    def test_size_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        plotter.set_size(100)
        plotter.set_size([100] * len(self.x_data))
        with self.assertRaises(ArgumentStructureError):
            plotter.set_size([100] * (len(self.x_data) - 1))

        with self.assertRaises(ArgumentTypeError):
            plotter.set_size("invalid")

    def test_marker_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        valid_markers = ["o", "s", "^", MarkerStyle("o")]
        for marker in valid_markers:
            plotter.set_marker(marker)
        with self.assertRaises(ArgumentTypeError):
            plotter.set_marker(123)

    def test_scale_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        valid_scales = ["linear", "log", "symlog", "logit"]
        for scale in valid_scales:
            plotter.set_scales(xscale=scale, yscale=scale)
        with self.assertRaises(ArgumentValueError):
            plotter.set_scales(xscale="invalid")

        with self.assertRaises(ArgumentValueError):
            plotter.set_scales(yscale="invalid")

    def test_grid_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        plotter.set_grid(visible=True, which="major", axis="both", color="gray")
        plotter.set_grid(visible=False, which="minor", axis="x")
        with self.assertRaises(ArgumentTypeError):
            plotter.set_grid(visible="invalid")

    def test_texts_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        valid_texts = [
            {"s": "Test", "x": 0.5, "y": 0.5},
            [{"s": "Test1", "x": 0.2, "y": 0.2}, {"s": "Test2", "x": 0.8, "y": 0.8}],
        ]
        for texts in valid_texts:
            plotter.set_texts(texts)
        with self.assertRaises(ArgumentTypeError):
            plotter.set_texts("invalid")

    def test_ax_limits_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        plotter.set_ax_limits(xlim=(-1, 1), ylim=(-1, 1))
        with self.assertRaises(ArgumentStructureError):
            plotter.set_ax_limits(xlim=(-1,))

        with self.assertRaises(ArgumentTypeError):
            plotter.set_ax_limits(xlim=("invalid", "invalid"))

    def test_draw_method(self):
        plotter = ScatterPlotter(self.x_data, self.y_data, **self.valid_params)
        ax = plotter.draw()
        self.assertIsInstance(ax, plt.Axes)
        self.assertIsNotNone(plotter.figure)
        self.assertIsNotNone(plotter.ax)

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
