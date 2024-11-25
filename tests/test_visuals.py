import unittest
from unittest import TestCase

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from mitools.exceptions import ArgumentStructureError, ArgumentTypeError
from mitools.visuals.axes_functions import (
    adjust_ax_labels_fontsize,
    adjust_ax_text_limits,
    adjust_axes_labels_fontsize,
    adjust_axes_lims,
    are_axes_empty,
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


class TestAdjustAxLabelsFontsize(TestCase):
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
            adjust_ax_labels_fontsize(self.ax, fontsize="Invalid")  # Invalid type
        with self.assertRaises(ArgumentTypeError):
            adjust_ax_labels_fontsize(self.ax, fontsize=None)  # Invalid type

    def test_literal_fontsizes(self):
        self.ax.set_xlabel("Literal Fontsize Test")
        self.ax.set_ylabel("Literal Fontsize Test")
        adjust_ax_labels_fontsize(self.ax, fontsize="small")


class TestAdjustAxesLabelsFontsize(TestCase):
    def setUp(self):
        self.figure = Figure()
        self.ax1: Axes = self.figure.add_subplot(211)
        self.ax2: Axes = self.figure.add_subplot(212)
        self.axes = [self.ax1, self.ax2]

    def test_single_fontsize(self):
        adjust_axes_labels_fontsize(self.axes, fontsizes=15)
        for ax in self.axes:
            self.assertEqual(ax.xaxis.label.get_fontsize(), 15)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 15)

    def test_single_ax_fontsize(self):
        adjust_axes_labels_fontsize(self.axes[0], fontsizes=15)
        for ax in self.axes[:1]:
            self.assertEqual(ax.xaxis.label.get_fontsize(), 15)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 15)

    def test_multiple_fontsizes(self):
        adjust_axes_labels_fontsize(self.axes, fontsizes=[10, 20])
        self.assertEqual(self.ax1.xaxis.label.get_fontsize(), 10)
        self.assertEqual(self.ax1.yaxis.label.get_fontsize(), 10)
        self.assertEqual(self.ax2.xaxis.label.get_fontsize(), 20)
        self.assertEqual(self.ax2.yaxis.label.get_fontsize(), 20)

    def test_empty_axes_iterable(self):
        empty_axes = []
        result = adjust_axes_labels_fontsize(empty_axes, fontsizes=12)
        self.assertEqual(result, [])

    def test_invalid_fontsizes_length(self):
        with self.assertRaises(ArgumentStructureError):
            adjust_axes_labels_fontsize(self.axes, fontsizes=[10])

    def test_invalid_fontsizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_axes_labels_fontsize(self.axes, fontsizes="invalid")  # Invalid type

    def test_valid_fontsize_string(self):
        adjust_axes_labels_fontsize(self.axes, fontsizes="small")
        for ax in self.axes:
            self.assertEqual(ax.xaxis.label.get_fontsize(), 8.33)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 8.33)

    def test_invalid_axes_type(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_axes_labels_fontsize([self.ax1, "Not an Axes object"], fontsizes=12)

    def test_no_labels_set(self):
        adjust_axes_labels_fontsize(self.axes, fontsizes=14)
        for ax in self.axes:
            self.assertEqual(ax.xaxis.label.get_fontsize(), 14)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 14)

    def test_boundary_fontsize(self):
        adjust_axes_labels_fontsize(self.axes, fontsizes=[1, 100])
        self.assertEqual(self.ax1.xaxis.label.get_fontsize(), 1)
        self.assertEqual(self.ax1.yaxis.label.get_fontsize(), 1)
        self.assertEqual(self.ax2.xaxis.label.get_fontsize(), 100)
        self.assertEqual(self.ax2.yaxis.label.get_fontsize(), 100)

    def test_mixed_empty_and_non_empty_axes(self):
        self.ax1.set_xlabel("Test X")
        self.ax2.set_ylabel("Test Y")
        adjust_axes_labels_fontsize(self.axes, fontsizes=[16, 18])
        self.assertEqual(self.ax1.xaxis.label.get_fontsize(), 16)
        self.assertEqual(self.ax2.yaxis.label.get_fontsize(), 18)

    def test_already_set_fontsize(self):
        self.ax1.xaxis.label.set_fontsize(12)
        self.ax1.yaxis.label.set_fontsize(12)
        self.ax2.xaxis.label.set_fontsize(14)
        self.ax2.yaxis.label.set_fontsize(14)
        adjust_axes_labels_fontsize(self.axes, fontsizes=[16, 18])
        self.assertEqual(self.ax1.xaxis.label.get_fontsize(), 16)
        self.assertEqual(self.ax1.yaxis.label.get_fontsize(), 16)
        self.assertEqual(self.ax2.xaxis.label.get_fontsize(), 18)
        self.assertEqual(self.ax2.yaxis.label.get_fontsize(), 18)


class TestAreAxesEmpty(TestCase):
    def setUp(self):
        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.axes = [self.ax1, self.ax2]

    def test_all_empty_axes(self):
        self.assertTrue(are_axes_empty(self.axes))

    def test_one_non_empty_axes(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(are_axes_empty(self.axes))

    def test_all_non_empty_axes(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.ax2.set_xlabel("Test Label")
        self.assertFalse(are_axes_empty(self.axes))

    def test_empty_single_ax(self):
        self.assertTrue(are_axes_empty(self.ax1))

    def test_non_empty_single_ax(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(are_axes_empty(self.ax1))

    def test_axes_with_lines(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_patches(self):
        rect = Rectangle((0, 0), 1, 1)
        self.ax1.add_patch(rect)
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_collections(self):
        x = np.random.rand(10)
        y = np.random.rand(10)
        self.ax2.scatter(x, y)
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_texts(self):
        self.ax1.text(0.5, 0.5, "Test Text")
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_images(self):
        data = np.random.rand(10, 10)
        self.ax2.imshow(data)
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_labels(self):
        self.ax1.set_xlabel("X-axis Label")
        self.ax2.set_ylabel("Y-axis Label")
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_titles(self):
        self.ax1.set_title("Test Title")
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_with_legend(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.ax1.legend()
        self.assertFalse(are_axes_empty(self.axes))

    def test_axes_mixed_content(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(are_axes_empty(self.axes))

    def test_all_cleared_axes(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.ax2.set_xlabel("X-axis Label")
        self.ax1.cla()
        self.ax2.cla()
        self.assertTrue(are_axes_empty(self.axes))

    def test_invalid_input(self):
        with self.assertRaises(ArgumentTypeError):
            are_axes_empty("Not an Axes object")
        with self.assertRaises(ArgumentTypeError):
            are_axes_empty(None)
        with self.assertRaises(ArgumentTypeError):
            are_axes_empty([self.ax1, "Invalid Type"])

    def test_empty_axes_iterable(self):
        self.assertTrue(are_axes_empty([]))

    def test_mixed_empty_and_non_empty_axes(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(are_axes_empty(self.axes))

    def test_single_empty_axes(self):
        self.assertTrue(are_axes_empty(self.ax1))

    def test_single_non_empty_axes(self):
        self.ax1.plot([0, 1], [0, 1], label="Test Line")
        self.assertFalse(are_axes_empty(self.ax1))


if __name__ == "__main__":
    unittest.main()
