import unittest
from unittest import TestCase

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from mitools.exceptions import ArgumentTypeError
from mitools.visuals.axes_functions import (
    adjust_ax_labels_fontsize,
    adjust_axes_lims,
    adjust_text_axes_limits,
    get_axes_limits,
    is_ax_empty,
    set_axes_limits,
)


class TestIsAxEmpty(TestCase):
    def setUp(self):
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)

    def test_empty_axes(self):
        self.assertTrue(is_ax_empty(self.ax))

    def test_axes_with_lines(self):
        self.ax.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_patches(self):
        rect = Rectangle((0, 0), 1, 1)
        self.ax.add_patch(rect)
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_collections(self):
        x = np.random.rand(10)
        y = np.random.rand(10)
        self.ax.scatter(x, y)
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_texts(self):
        self.ax.text(0.5, 0.5, "Test Text")
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_images(self):
        data = np.random.rand(10, 10)
        self.ax.imshow(data)
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_xlabel(self):
        self.ax.set_xlabel("X-axis Label")
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_ylabel(self):
        self.ax.set_ylabel("Y-axis Label")
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_legend(self):
        self.ax.plot([0, 1], [0, 1], label="Test Line")
        self.ax.legend()
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_multiple_elements(self):
        self.ax.plot([0, 1], [0, 1], label="Test Line")
        self.ax.text(0.5, 0.5, "Test Text")
        self.ax.scatter(np.random.rand(10), np.random.rand(10))
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_after_clearing(self):
        self.ax.plot([0, 1], [0, 1], label="Test Line")
        self.ax.cla()  # Clear the Axes
        self.assertTrue(is_ax_empty(self.ax))

    def test_invalid_input(self):
        with self.assertRaises(ArgumentTypeError):
            is_ax_empty("Not an Axes object")
        with self.assertRaises(ArgumentTypeError):
            is_ax_empty(None)

    def test_axes_with_titles_or_labels(self):
        self.ax.set_title("Test Title")
        self.ax.set_xlabel("X-axis Label")
        self.ax.set_ylabel("Y-axis Label")
        self.assertFalse(is_ax_empty(self.ax))

    def test_axes_with_spines_and_ticks(self):
        self.assertTrue(is_ax_empty(self.ax))


class TestAdjustAxFontsize(TestCase):
    def setUp(self):
        self.figure = Figure()
        self.ax: Axes = self.figure.add_subplot(111)

    def test_valid_input(self):
        self.ax.set_xlabel("Test X")
        self.ax.set_ylabel("Test Y")
        adjust_ax_labels_fontsize(self.ax, fontsize=20)

        self.assertEqual(self.ax.xaxis.label.get_fontsize(), 20)
        self.assertEqual(self.ax.yaxis.label.get_fontsize(), 20)

    def test_no_labels_set(self):
        adjust_ax_labels_fontsize(self.ax, fontsize=15)
        self.assertEqual(self.ax.xaxis.label.get_fontsize(), 15)
        self.assertEqual(self.ax.yaxis.label.get_fontsize(), 15)
        self.assertEqual(self.ax.get_xlabel(), "")
        self.assertEqual(self.ax.get_ylabel(), "")

    def test_empty_axes(self):
        adjust_ax_labels_fontsize(self.ax, fontsize=10)
        self.assertEqual(self.ax.xaxis.label.get_fontsize(), 10)
        self.assertEqual(self.ax.yaxis.label.get_fontsize(), 10)

    def test_none_axes(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_ax_labels_fontsize(None, fontsize=12)

    def test_boundary_font_sizes(self):
        self.ax.set_xlabel("Boundary Test")
        self.ax.set_ylabel("Boundary Test")
        adjust_ax_labels_fontsize(self.ax, fontsize=1)
        self.assertEqual(self.ax.xaxis.label.get_fontsize(), 1)
        self.assertEqual(self.ax.yaxis.label.get_fontsize(), 1)
        adjust_ax_labels_fontsize(self.ax, fontsize=100)
        self.assertEqual(self.ax.xaxis.label.get_fontsize(), 100)
        self.assertEqual(self.ax.yaxis.label.get_fontsize(), 100)

    def test_invalid_fontsize_type(self):
        self.ax.set_xlabel("Invalid Fontsize Test")
        self.ax.set_ylabel("Invalid Fontsize Test")
        with self.assertRaises(ArgumentTypeError):
            adjust_ax_labels_fontsize(self.ax, fontsize="large")  # Invalid type
        with self.assertRaises(ArgumentTypeError):
            adjust_ax_labels_fontsize(self.ax, fontsize=None)  # Invalid type


if __name__ == "__main__":
    unittest.main()
