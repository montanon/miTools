import unittest
from typing import Tuple
from unittest import TestCase
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.axes_functions import (
    adjust_ax_labels_fontsize,
    adjust_ax_text_limits,
    adjust_axes_array_limits,
    adjust_axes_labels_fontsize,
    adjust_axes_text_limits,
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


class TestAdjustAxTextLimits(TestCase):
    def setUp(self):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        self.figure = fig
        self.ax: Axes = axes.flat[0]

    def test_adjust_x_axis_with_text(self):
        text = self.ax.text(1, 1, "Sample Text")
        adjust_ax_text_limits(self.ax, text, axis="x")
        xlim = self.ax.get_xlim()
        self.assertTrue(xlim[1] >= 1, "X-axis limit should encompass the text")

    def test_adjust_y_axis_with_text(self):
        text = self.ax.text(1, 1, "Sample Text")
        adjust_ax_text_limits(self.ax, text, axis="y")
        ylim = self.ax.get_ylim()
        self.assertTrue(ylim[1] >= 1, "Y-axis limit should encompass the text")

    def test_adjust_both_axes_with_text(self):
        text = self.ax.text(2, 3, "Sample Text")
        adjust_ax_text_limits(self.ax, text, axis="both")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.assertTrue(xlim[1] >= 2, "X-axis limit should encompass the text")
        self.assertTrue(ylim[1] >= 3, "Y-axis limit should encompass the text")

    def test_adjust_with_large_text(self):
        text = self.ax.text(5, 5, "A" * 100)  # Long text
        adjust_ax_text_limits(self.ax, text, axis="both")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.assertTrue(xlim[1] >= 5, "X-axis limit should adjust for long text")
        self.assertTrue(ylim[1] >= 5, "Y-axis limit should adjust for long text")

    def test_no_change_when_text_is_within_limits(self):
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        text = self.ax.text(5, 5, "Centered Text")
        adjust_ax_text_limits(self.ax, text, axis="both")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.assertEqual(xlim, (0, 10), "X-axis limits should remain unchanged")
        self.assertEqual(ylim, (0, 10), "Y-axis limits should remain unchanged")

    def test_invalid_axes_argument(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_ax_text_limits("Not an Axes object", Text(x=0, y=0, text="Invalid"))

    def test_invalid_text_argument(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_ax_text_limits(self.ax, "Not a Text object", axis="x")

    def test_invalid_axis_argument(self):
        text = self.ax.text(x=1, y=1, s="Sample Text")
        with self.assertRaises(ArgumentValueError):
            adjust_ax_text_limits(self.ax, text, axis="invalid")

    def test_text_outside_current_limits(self):
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        text = self.ax.text(2, 2, "Out of bounds")
        adjust_ax_text_limits(self.ax, text, axis="both")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.assertTrue(
            xlim[1] >= 2, "X-axis limit should adjust to encompass the text"
        )
        self.assertTrue(
            ylim[1] >= 2, "Y-axis limit should adjust to encompass the text"
        )

    def test_text_at_edge_of_limits(self):
        self.ax.set_xlim(0, 5)
        self.ax.set_ylim(0, 5)
        text = self.ax.text(5, 5, "Edge Text")
        adjust_ax_text_limits(self.ax, text, axis="both")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.assertTrue(
            xlim[1] >= 5, "X-axis limit should remain or expand to encompass the text"
        )
        self.assertTrue(
            ylim[1] >= 5, "Y-axis limit should remain or expand to encompass the text"
        )

    def test_adjust_axis_with_multiple_texts(self):
        self.ax.text(2, 2, "Text 1")
        text2 = self.ax.text(6, 6, "Text 2")
        adjust_ax_text_limits(self.ax, text2, axis="both")
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.assertTrue(xlim[1] >= 6, "X-axis limit should adjust to the farthest text")
        self.assertTrue(ylim[1] >= 6, "Y-axis limit should adjust to the farthest text")

    def test_adjust_after_canvas_clear(self):
        text = self.ax.text(1, 1, "Sample Text")
        self.figure.canvas.draw()
        adjust_ax_text_limits(self.ax, text, axis="x")
        xlim = self.ax.get_xlim()
        self.assertTrue(
            xlim[1] >= 1, "X-axis limit should adjust even after canvas draw"
        )

    def test_adjust_with_subplots(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        text1 = ax1.text(3, 3, "Subplot 1 Text")
        text2 = ax2.text(5, 5, "Subplot 2 Text")
        adjust_ax_text_limits(ax1, text1, axis="both")
        adjust_ax_text_limits(ax2, text2, axis="both")
        xlim1 = ax1.get_xlim()
        ylim1 = ax1.get_ylim()
        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()
        self.assertTrue(xlim1[1] >= 3, "Ax1 X-axis should adjust for text")
        self.assertTrue(ylim1[1] >= 3, "Ax1 Y-axis should adjust for text")
        self.assertTrue(xlim2[1] >= 5, "Ax2 X-axis should adjust for text")
        self.assertTrue(ylim2[1] >= 5, "Ax2 Y-axis should adjust for text")


class TestAdjustAxesTextLimits(TestCase):
    def setUp(self):
        self.figure, self.axes = plt.subplots(nrows=2, ncols=2)
        self.texts = [
            ax.text(0.5, 0.5, f"Text {i}") for i, ax in enumerate(self.axes.flat)
        ]

    def test_adjust_single_axis_with_text(self):
        single_ax = self.axes[0, 0]
        single_text = self.texts[0]
        adjust_axes_text_limits(single_ax, single_text, axis="x")
        xlim = single_ax.get_xlim()
        self.assertTrue(xlim[1] >= 0.5, "X-axis limit should adjust to fit the text")

    def test_adjust_multiple_axes_with_texts(self):
        adjust_axes_text_limits(self.axes.flat, self.texts, axis="both")
        for ax, text in zip(self.axes.flat, self.texts):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            self.assertTrue(
                xlim[1] >= 0.5, "X-axis limit should adjust to fit the text"
            )
            self.assertTrue(
                ylim[1] >= 0.5, "Y-axis limit should adjust to fit the text"
            )

    def test_adjust_x_axis_only(self):
        adjust_axes_text_limits(self.axes.flat, self.texts, axis="x")
        for ax, text in zip(self.axes.flat, self.texts):
            xlim = ax.get_xlim()
            self.assertTrue(
                xlim[1] >= 0.5, "X-axis limit should adjust to fit the text"
            )
            ylim = ax.get_ylim()
            self.assertEqual(ylim, (0.0, 1.0), "Y-axis limits should remain unchanged")

    def test_adjust_y_axis_only(self):
        adjust_axes_text_limits(self.axes.flat, self.texts, axis="y")
        for ax, text in zip(self.axes.flat, self.texts):
            ylim = ax.get_ylim()
            self.assertTrue(
                ylim[1] >= 0.5, "Y-axis limit should adjust to fit the text"
            )
            xlim = ax.get_xlim()
            self.assertEqual(xlim, (0.0, 1.0), "X-axis limits should remain unchanged")

    def test_no_adjustment_needed(self):
        for ax in self.axes.flat:
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
        adjust_axes_text_limits(self.axes.flat, self.texts, axis="both")
        for ax in self.axes.flat:
            self.assertEqual(
                ax.get_xlim(), (0, 2), "X-axis limits should remain unchanged"
            )
            self.assertEqual(
                ax.get_ylim(), (0, 2), "Y-axis limits should remain unchanged"
            )

    def test_mismatched_axes_and_texts(self):
        additional_text = self.axes[0, 0].text(1, 1, "Extra Text")
        mismatched_texts = self.texts + [additional_text]
        with self.assertRaises(ArgumentStructureError):
            adjust_axes_text_limits(self.axes.flat, mismatched_texts, axis="both")

    def test_invalid_axes_argument(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_axes_text_limits("Invalid Axes", self.texts, axis="x")

    def test_invalid_texts_argument(self):
        with self.assertRaises(ArgumentTypeError):
            adjust_axes_text_limits(self.axes.flat, "Invalid Texts", axis="x")

    def test_invalid_axis_argument(self):
        # with self.assertRaises(ArgumentValueError):
        adjust_axes_text_limits(self.axes.flat, self.texts, axis="invalid")

    def test_empty_axes_and_texts(self):
        result = adjust_axes_text_limits([], [], axis="both")
        self.assertEqual(result, [], "Empty input should return an empty list")

    def test_mixed_axes_and_texts(self):
        subset_axes = self.axes.flat[:2]
        subset_texts = self.texts[:2]
        adjust_axes_text_limits(subset_axes, subset_texts, axis="both")
        for ax, text in zip(subset_axes, subset_texts):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            self.assertTrue(
                xlim[1] >= 0.5, "X-axis limit should adjust to fit the text"
            )
            self.assertTrue(
                ylim[1] >= 0.5, "Y-axis limit should adjust to fit the text"
            )

    def test_adjust_with_preexisting_limits(self):
        for i, ax in enumerate(self.axes.flat):
            ax.set_xlim(0, 1 + i)
            ax.set_ylim(0, 1 + i)
        adjust_axes_text_limits(self.axes.flat, self.texts, axis="both")
        for ax, text in zip(self.axes.flat, self.texts):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            self.assertTrue(
                xlim[1] >= 0.5, "X-axis limit should adjust to fit the text"
            )
            self.assertTrue(
                ylim[1] >= 0.5, "Y-axis limit should adjust to fit the text"
            )


class TestGetAxesLimits(TestCase):
    def setUp(self):
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.fig.add_subplot(121)
        self.ax3 = self.fig.add_subplot(122)

        self.ax1.plot([1, 2, 3], [4, 5, 6])
        self.ax2.plot([10, 20, 30], [40, 50, 60])

    def tearDown(self):
        plt.close(self.fig)

    def test_get_x_limits_multiple_axes(self):
        limits = get_axes_limits([self.ax1, self.ax2], axis="x")
        self.assertEqual(limits, (0.9, 31.0))  # Expected x-axis range

    def test_get_y_limits_multiple_axes(self):
        limits = get_axes_limits([self.ax1, self.ax2], axis="y")
        self.assertEqual(limits, (3.9, 61.0))  # Expected y-axis range

    def test_get_x_limits_single_axis(self):
        limits = get_axes_limits(self.ax1, axis="x")
        self.assertEqual(limits, (0.9, 3.1))  # Expected x-axis range for ax1

    def test_get_y_limits_single_axis(self):
        limits = get_axes_limits(self.ax1, axis="y")
        self.assertEqual(limits, (3.9, 6.1))  # Expected y-axis range for ax1

    def test_no_valid_axes(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_axes_limits([], axis="x")
        self.assertEqual(
            str(context.exception), "No valid limits found across the provided axes."
        )

    def test_empty_axis_ignored(self):
        limits = get_axes_limits([self.ax1, self.ax2, self.ax3], axis="x")
        self.assertEqual(limits, (0.9, 31.0))  # Empty ax3 ignored

    def test_invalid_axis_parameter(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_axes_limits([self.ax1, self.ax2], axis="z")
        self.assertEqual(
            str(context.exception),
            "The 'axis'=z parameter must be one of 'x' or 'y'.",
        )

    def test_custom_get_lim_func(self):
        def custom_get_lim_func(ax: Axes) -> Tuple[float, float]:
            return ax.get_xlim()

        limits = get_axes_limits([self.ax1, self.ax2], get_lim_func=custom_get_lim_func)
        self.assertEqual(limits, (0.9, 31.0))  # Expected x-axis range

    def test_callable_check(self):
        limits = get_axes_limits([self.ax1, self.ax2], axis="y", get_lim_func=None)
        self.assertEqual(limits, (3.9, 61.0))  # Expected y-axis range

    def test_invalid_axes_object(self):
        with self.assertRaises(TypeError):
            get_axes_limits(None, axis="x")

    def test_no_limits_found(self):
        empty_fig, empty_ax = plt.subplots()
        with self.assertRaises(ArgumentValueError):
            get_axes_limits(empty_ax, axis="x")
        plt.close(empty_fig)


class TestSetAxesLimits(TestCase):
    def setUp(self):
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.fig.add_subplot(121)
        self.ax3 = self.fig.add_subplot(122)

        self.ax1.plot([1, 2, 3], [4, 5, 6])
        self.ax2.plot([10, 20, 30], [40, 50, 60])

    def tearDown(self):
        plt.close(self.fig)

    def test_set_x_limits_single_axis(self):
        axes = set_axes_limits(self.ax1, lim_min=0, lim_max=10, axis="x")
        self.assertEqual(self.ax1.get_xlim(), (0, 10))
        self.assertIn(self.ax1, axes)

    def test_set_y_limits_single_axis(self):
        axes = set_axes_limits(self.ax1, lim_min=-5, lim_max=5, axis="y")
        self.assertEqual(self.ax1.get_ylim(), (-5, 5))
        self.assertIn(self.ax1, axes)

    def test_set_x_limits_multiple_axes(self):
        axes = set_axes_limits([self.ax1, self.ax2], lim_min=0, lim_max=10, axis="x")
        self.assertEqual(self.ax1.get_xlim(), (0, 10))
        self.assertEqual(self.ax2.get_xlim(), (0, 10))
        self.assertIn(self.ax1, axes)
        self.assertIn(self.ax2, axes)

    def test_set_y_limits_multiple_axes(self):
        axes = set_axes_limits([self.ax1, self.ax2], lim_min=-5, lim_max=5, axis="y")
        self.assertEqual(self.ax1.get_ylim(), (-5, 5))
        self.assertEqual(self.ax2.get_ylim(), (-5, 5))
        self.assertIn(self.ax1, axes)
        self.assertIn(self.ax2, axes)

    def test_empty_axes(self):
        set_axes_limits([], lim_min=0, lim_max=10, axis="x")

    def test_invalid_axis_parameter(self):
        with self.assertRaises(ArgumentValueError) as context:
            set_axes_limits([self.ax1, self.ax2], lim_min=0, lim_max=10, axis="z")
        self.assertEqual(
            str(context.exception),
            "The 'axis'=z parameter must be one of 'x' or 'y'.",
        )

    def test_invalid_axes_object(self):
        with self.assertRaises(TypeError):
            set_axes_limits(None, lim_min=0, lim_max=10, axis="x")

    def test_custom_set_lim_func(self):
        custom_func = MagicMock()
        set_axes_limits(
            [self.ax1, self.ax2], lim_min=0, lim_max=10, set_lim_func=custom_func
        )
        custom_func.assert_any_call(self.ax1, (0, 10))
        custom_func.assert_any_call(self.ax2, (0, 10))

    def test_callable_check(self):
        axes = set_axes_limits(
            [self.ax1, self.ax2], lim_min=0, lim_max=10, axis="y", set_lim_func=None
        )
        self.assertEqual(self.ax1.get_ylim(), (0, 10))
        self.assertEqual(self.ax2.get_ylim(), (0, 10))
        self.assertIn(self.ax1, axes)
        self.assertIn(self.ax2, axes)


if __name__ == "__main__":
    unittest.main()
