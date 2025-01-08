import json
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union
from unittest import TestCase
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.testing import assert_array_equal

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.plotter import Plotter, PlotterException


class DummyPlotter(Plotter):
    def _create_plot(self):
        pass


class TestPlotter(TestCase):
    def setUp(self):
        x_data = np.array([1, 2, 3, 4])
        y_data = [4, 3, 2, 1]
        self.plotter = DummyPlotter(x_data, y_data)
        self.multi_plotter = DummyPlotter([x_data, x_data], [y_data, y_data])

    def test_init_valid_inputs(self):
        x_data = np.array([1, 2, 3, 4])
        y_data = [4, 3, 2, 1]
        plotter = DummyPlotter(x_data, y_data)
        assert_array_equal(plotter.x_data, x_data.reshape(1, -1))
        assert_array_equal(plotter.y_data, np.asarray([y_data]))
        self.assertEqual(plotter._n_sequences, 1)
        self.assertFalse(plotter._multi_data)
        self.assertEqual(plotter.data_size, 4)

    def test_init_multi_sequence(self):
        x_data = np.asarray([[1, 2, 3], [4, 5, 6]])
        y_data = np.asarray([[6, 5, 4], [3, 2, 1]])
        plotter = DummyPlotter(x_data, y_data)
        assert_array_equal(plotter.x_data, x_data)
        assert_array_equal(plotter.y_data, y_data)
        self.assertEqual(plotter._n_sequences, 2)
        self.assertTrue(plotter._multi_data)
        self.assertEqual(plotter.data_size, 3)

    def test_init_y_data_none(self):
        x_data = np.array([[1, 2, 3, 4]])
        y_data = None
        plotter = DummyPlotter(x_data, y_data)
        assert_array_equal(plotter.x_data, x_data)
        self.assertIsNone(plotter.y_data)
        self.assertEqual(plotter._n_sequences, 1)
        self.assertFalse(plotter._multi_data)
        self.assertEqual(plotter.data_size, 4)

    def test_init_invalid_x_data(self):
        x_data = ["a", "b", "c"]
        y_data = [1, 2, 3]
        with self.assertRaises(ArgumentTypeError):
            DummyPlotter(x_data, y_data)

    def test_init_inconsistent_lengths(self):
        x_data = [1, 2, 3]
        y_data = [4, 3]
        with self.assertRaises(ArgumentStructureError):
            DummyPlotter(x_data, y_data)

    def test_reset_params(self):
        self.plotter.set_title("Test Title")
        self.plotter.reset_params()
        self.assertEqual(self.plotter.title, "")

    def test_set_title(self):
        self.plotter.set_title("My Title")
        self.assertEqual(self.plotter.title["label"], "My Title")

    def test_set_title_with_kwargs(self):
        self.plotter.set_title("My Title", fontsize=12)
        self.assertEqual(self.plotter.title["label"], "My Title")
        self.assertEqual(self.plotter.title["fontsize"], 12)

    def test_set_xlabel(self):
        self.plotter.set_xlabel("X Axis")
        self.assertEqual(self.plotter.xlabel["xlabel"], "X Axis")

    def test_set_ylabel(self):
        self.plotter.set_ylabel("Y Axis")
        self.assertEqual(self.plotter.ylabel["ylabel"], "Y Axis")

    def test_set_axes_labels(self):
        self.plotter.set_axes_labels("X Axis", "Y Axis")
        self.assertEqual(self.plotter.xlabel["xlabel"], "X Axis")
        self.assertEqual(self.plotter.ylabel["ylabel"], "Y Axis")

    def test_set_legend(self):
        self.plotter.set_legend(show=True)
        self.assertTrue(self.plotter.legend["show"])

    def test_set_legend_with_labels(self):
        self.plotter.set_legend(labels=["Series 1", "Series 2"])
        self.assertEqual(
            self.plotter.legend["kwargs"]["labels"], ["Series 1", "Series 2"]
        )

    def test_set_legend_invalid_ncol(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_legend(ncol="two")

    def test_set_figsize(self):
        self.plotter.set_figsize((10, 5))
        self.assertEqual(self.plotter.figsize, (10, 5))

    def test_set_figsize_invalid(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_figsize("10x5")

    def test_set_style(self):
        self.plotter.set_style("ggplot")
        self.assertEqual(self.plotter.style, "ggplot")

    def test_set_style_invalid(self):
        with self.assertRaises(ArgumentValueError):
            self.plotter.set_style("unknown_style")

    def test_set_grid(self):
        self.plotter.set_grid(visible=True, which="both", axis="both", color="gray")
        self.assertEqual(self.plotter.grid["visible"], True)
        self.assertEqual(self.plotter.grid["which"], "both")
        self.assertEqual(self.plotter.grid["axis"], "both")
        self.assertEqual(self.plotter.grid["color"], "gray")

    def test_set_tight_layout(self):
        self.plotter.set_tight_layout(True)
        self.assertTrue(self.plotter.tight_layout)

    def test_set_texts(self):
        texts = [{"x": 1, "y": 2, "s": "Point"}]
        self.plotter.set_texts(texts)
        self.assertEqual(self.plotter.texts, texts)

    def test_set_texts_single_dict(self):
        text = {"x": 1, "y": 2, "s": "Point"}
        self.plotter.set_texts(text)
        self.assertEqual(self.plotter.texts, [text])

    def test_set_xscale(self):
        self.plotter.set_xscale("log")
        self.assertEqual(self.plotter.xscale, "log")

    def test_set_xscale_invalid(self):
        with self.assertRaises(ArgumentValueError):
            self.plotter.set_xscale("unknown_scale")

    def test_set_yscale(self):
        self.plotter.set_yscale("linear")
        self.assertEqual(self.plotter.yscale, "linear")

    def test_set_yscale_invalid(self):
        with self.assertRaises(ArgumentValueError):
            self.plotter.set_yscale("unknown_scale")

    def test_set_scales(self):
        self.plotter.set_scales(xscale="log", yscale="linear")
        self.assertEqual(self.plotter.xscale, "log")
        self.assertEqual(self.plotter.yscale, "linear")

    def test_set_background(self):
        self.plotter.set_background("blue")
        self.assertEqual(self.plotter.background, "blue")

    def test_set_background_invalid(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_background(123)

    def test_set_figure_background(self):
        self.plotter.set_figure_background("white")
        self.assertEqual(self.plotter.figure_background, "white")

    def test_set_figure_background_invalid(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_figure_background(456)

    def test_set_suptitle(self):
        self.plotter.set_suptitle("Super Title", fontsize=14)
        self.assertEqual(self.plotter.suptitle["t"], "Super Title")
        self.assertEqual(self.plotter.suptitle["fontsize"], 14)

    def test_set_xlim(self):
        self.plotter.set_xlim((0, 10))
        self.assertEqual(self.plotter.xlim, (0, 10))

    def test_set_xlim_with_None(self):
        self.plotter.set_xlim((0, None))
        self.assertEqual(self.plotter.xlim, (0, None))

    def test_set_xlim_invalid(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_xlim("0 to 10")

    def test_set_ylim(self):
        self.plotter.set_ylim((0, 5))
        self.assertEqual(self.plotter.ylim, (0, 5))

    def test_set_ylim_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.plotter.set_ylim([0])

    def test_set_limits(self):
        self.plotter.set_limits(xlim=(0, 10), ylim=(0, 5))
        self.assertEqual(self.plotter.xlim, (0, 10))
        self.assertEqual(self.plotter.ylim, (0, 5))

    def test_set_xticks(self):
        self.plotter.set_xticks([0, 1, 2])
        self.assertEqual(self.plotter.xticks, [0, 1, 2])

    def test_set_xticks_invalid(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_xticks("0,1,2")

    def test_set_yticks(self):
        self.plotter.set_yticks([0, 1, 2])
        self.assertEqual(self.plotter.yticks, [0, 1, 2])

    def test_set_yticks_invalid(self):
        with self.assertRaises(ArgumentTypeError):
            self.plotter.set_yticks({"ticks": [0, 1, 2]})

    def test_set_ticks(self):
        self.plotter.set_ticks(xticks=[0, 1], yticks=[2, 3])
        self.assertEqual(self.plotter.xticks, [0, 1])
        self.assertEqual(self.plotter.yticks, [2, 3])

    def test_set_xticklabels(self):
        self.plotter.set_xticklabels(["zero", "one", "two"])
        self.assertEqual(self.plotter.xticklabels, ["zero", "one", "two"])

    def test_set_yticklabels(self):
        self.plotter.set_yticklabels(["A", "B", "C"])
        self.assertEqual(self.plotter.yticklabels, ["A", "B", "C"])

    def test_set_ticklabels(self):
        self.plotter.set_ticklabels(xticklabels=["X0", "X1"], yticklabels=["Y0", "Y1"])
        self.assertEqual(self.plotter.xticklabels, ["X0", "X1"])
        self.assertEqual(self.plotter.yticklabels, ["Y0", "Y1"])

    def test_set_xtickparams(self):
        params = {"labelsize": 10}
        self.plotter.set_xtickparams(params)
        self.assertEqual(self.plotter.xtickparams, params)

    def test_set_ytickparams(self):
        params = {"colors": "red"}
        self.plotter.set_ytickparams(params)
        self.assertEqual(self.plotter.ytickparams, params)

    def test_set_tickparams(self):
        x_params = {"labelsize": 10}
        y_params = {"colors": "red"}
        self.plotter.set_tickparams(xtickparams=x_params, ytickparams=y_params)
        self.assertEqual(self.plotter.xtickparams, x_params)
        self.assertEqual(self.plotter.ytickparams, y_params)

    def test_set_spines(self):
        left_params = {"visible": False}
        right_params = {"color": "green"}
        self.plotter.set_spines(left=left_params, right=right_params)
        self.assertEqual(self.plotter.spines["left"]["visible"], False)
        self.assertEqual(self.plotter.spines["right"]["color"], "green")

    def test_set_color_single_color(self):
        self.plotter.set_color("red")
        self.assertEqual(self.plotter.color, "red")

    def test_set_color_sequence(self):
        colors = ["red", "green", "blue", "yellow"]
        self.plotter.set_color(colors)
        self.assertEqual(self.plotter.color, colors)
        self.multi_plotter.set_color([colors, colors])
        self.assertEqual(self.multi_plotter.color, [colors, colors])

    def test_set_color_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.plotter.set_color(["asdasd"] * 3)
        with self.assertRaises(ArgumentStructureError):
            self.plotter.set_color([1, 2, 3, 4, 5])
        with self.assertRaises(ArgumentStructureError):
            self.multi_plotter.set_color(["red", "green", "blue", "yellow"])

    def test_set_alpha_single_value(self):
        self.plotter.set_alpha(0.5)
        self.assertEqual(self.plotter.alpha, 0.5)

    def test_set_alpha_sequence(self):
        alphas = [0.5, 0.6, 0.7, 0.9]
        self.plotter.set_alpha(alphas)
        assert_array_equal(self.plotter.alpha, np.array(alphas))
        self.multi_plotter.set_alpha([alphas, alphas])
        assert_array_equal(self.multi_plotter.alpha, np.array([alphas, alphas]))

    def test_set_alpha_invalid(self):
        with self.assertRaises(ArgumentValueError):
            self.plotter.set_alpha(123)
        with self.assertRaises(ArgumentStructureError):
            self.plotter.set_alpha(["asdasd"] * 3)
        with self.assertRaises(ArgumentStructureError):
            self.plotter.set_alpha([1, 2, 3, 4, 5])
        with self.assertRaises(ArgumentStructureError):
            self.multi_plotter.set_alpha([0.5, 0.6, 0.7, 0.9])

    def test_set_label_single(self):
        self.plotter.set_label("Series 1")
        self.assertEqual(self.plotter.label, "Series 1")

    def test_set_label_sequence(self):
        label = "Series 1"
        self.plotter.set_label(label)
        self.assertEqual(self.plotter.label, label)
        self.multi_plotter.set_label([label, label])
        self.assertEqual(self.multi_plotter.label, [label, label])
        self.multi_plotter.set_label(label)
        self.assertEqual(self.multi_plotter.label, label)

    def test_set_label_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.plotter.set_label(123)

    def test_set_zorder_single(self):
        self.plotter.set_zorder(2)
        self.assertEqual(self.plotter.zorder, 2)

    def test_set_zorder_sequence(self):
        zorders = [1, 2, 3, 4]
        self.plotter.set_zorder(zorders)
        assert_array_equal(self.plotter.zorder, np.asarray(zorders))
        self.multi_plotter.set_zorder([zorders, zorders])
        assert_array_equal(self.multi_plotter.zorder, np.asarray([zorders, zorders]))

    def test_prepare_draw(self):
        self.plotter.prepare_draw()
        self.assertIsNotNone(self.plotter.figure)
        self.assertIsNotNone(self.plotter.ax)

    def test_draw(self):
        self.plotter.draw(show=False)
        self.assertIsInstance(self.plotter.figure, Figure)
        self.assertIsInstance(self.plotter.ax, Axes)

    def test_save_plot(self):
        self.plotter.draw(show=False)
        with patch("matplotlib.figure.Figure.savefig") as mock_savefig:
            self.plotter.save_plot("test_plot.png", dpi=200)
            mock_savefig.assert_called_once_with(
                "test_plot.png", dpi=200, bbox_inches="tight"
            )

    def test_save_plot_without_draw(self):
        with self.assertRaises(PlotterException):
            self.plotter.save_plot("test_plot.png")

    def test_clear(self):
        self.plotter.draw(show=False)
        self.plotter.clear()
        self.assertIsNone(self.plotter.figure)
        self.assertIsNone(self.plotter.ax)

    def test_save_plotter(self):
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            self.plotter.save_plotter("plotter.json")
            mock_file.assert_called_once_with("plotter.json", "w")
            handle = mock_file()
            handle.write.assert_called()

    def test_from_json(self):
        data = {
            "x_data": [[1, 2, 3]],
            "y_data": [[4, 5, 6]],
            "title": "Test Plot",
            "xlim": [0, 10],
            "ylim": [0, 5],
        }
        with patch(
            "builtins.open", unittest.mock.mock_open(read_data=json.dumps(data))
        ):
            plotter = DummyPlotter.from_json("plotter.json")
            assert_array_equal(plotter.x_data, np.asarray(data["x_data"]))
            assert_array_equal(plotter.y_data, np.asarray(data["y_data"]))
            self.assertEqual(plotter.title, {"label": data["title"]})
            self.assertEqual(plotter.xlim, tuple(data["xlim"]))
            self.assertEqual(plotter.ylim, tuple(data["ylim"]))


if __name__ == "__main__":
    unittest.main()
