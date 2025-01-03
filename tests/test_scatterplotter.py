import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
            "xticks": np.arange(-1, 11, 1),
            "yticks": np.arange(-2, 2, 0.5),
            "xticklabels": [str(i) for i in range(-1, 11)],
            "yticklabels": [f"{i:.1f}" for i in np.arange(-2, 2, 0.5)],
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
                if isinstance(value, dict):
                    getattr(plotter2, setter_name)(**value)
                else:
                    getattr(plotter2, setter_name)(value)
        for param in self.valid_params.keys():
            if param in ["xticks", "yticks"]:
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
        valid_colors = ["red", "b", "#FF0000", "#0000FF", (1, 0, 0), (1, 0, 0, 1)]
        for color in valid_colors:
            plotter.set_color(color)
        invalid_colors = [(1, 0), (1, 0, 0, 1, 1), "not_a_color"]
        for color in invalid_colors:
            with self.assertRaises(
                (ArgumentTypeError, ArgumentStructureError, ArgumentValueError)
            ):
                plotter.set_color(color)

    def test_size_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        plotter.set_size(100)
        plotter.set_size([100] * len(self.x_data))
        with self.assertRaises(ArgumentStructureError):
            plotter.set_size([100] * (len(self.x_data) - 1))

        with self.assertRaises(ArgumentStructureError):
            plotter.set_size("invalid")

    def test_marker_validation(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        valid_markers = ["o", "s", "^", MarkerStyle("o")]
        for marker in valid_markers:
            plotter.set_marker(marker)
        with self.assertRaises(ArgumentStructureError):
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
        plotter.set_limits(xlim=(-1, 1), ylim=(-1, 1))
        plotter.set_limits(xlim=(-1, None))
        with self.assertRaises(ArgumentTypeError):
            plotter.set_limits(xlim=("invalid", "invalid"))

    def test_draw_method(self):
        plotter = ScatterPlotter(self.x_data, self.y_data, **self.valid_params)
        ax = plotter.draw()
        self.assertIsInstance(ax, plt.Axes)
        self.assertIsNotNone(plotter.figure)
        self.assertIsNotNone(plotter.ax)

    def test_save_and_load_plotter(self):
        plotter1 = ScatterPlotter(self.x_data, self.y_data, **self.valid_params)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            plotter1.save_plotter(temp_path, data=True)
            plotter2 = ScatterPlotter.from_json(temp_path)
            for param in self.valid_params.keys():
                if param in ["xticks", "yticks"]:
                    for val1, val2 in zip(
                        getattr(plotter1, param), getattr(plotter2, param)
                    ):
                        self.assertEqual(
                            val1,
                            val2,
                            f"Parameter {param} differs between saved and loaded plotter",
                        )
                else:
                    self.assertEqual(
                        getattr(plotter1, param),
                        getattr(plotter2, param),
                        f"Parameter {param} differs between saved and loaded plotter",
                    )
            np.testing.assert_array_equal(plotter1.x_data, plotter2.x_data)
            np.testing.assert_array_equal(plotter1.y_data, plotter2.y_data)
        finally:
            temp_path.unlink()

    def test_init_params_completeness(self):
        plotter = ScatterPlotter(self.x_data, self.y_data)
        instance_attrs = set(
            attr
            for attr in dir(plotter)
            if not attr.startswith("_")
            and not callable(getattr(plotter, attr))
            and type(getattr(plotter.__class__, attr, None)) is not property
        )
        init_params = set(plotter._init_params.keys())
        required_attrs = {"x_data", "y_data", "data_size", "figure", "ax"}
        missing_params = instance_attrs - init_params - required_attrs
        extra_params = init_params - instance_attrs
        self.assertEqual(
            missing_params,
            set(),
            f"The following attributes are not included in _init_params: {missing_params}",
        )
        self.assertEqual(
            extra_params,
            set(),
            f"The following _init_params don't correspond to any attributes: {extra_params}",
        )
        required_keys = {"default", "type"}
        for param, config in plotter._init_params.items():
            self.assertTrue(
                all(key in config for key in required_keys),
                f"Parameter '{param}' is missing required keys in _init_params. "
                f"Required: {required_keys}, Found: {set(config.keys())}",
            )

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
