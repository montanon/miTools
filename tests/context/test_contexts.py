import json
import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from threading import Thread
from unittest import TestCase
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from mitools.context import (
    Dev,
    SavePlotContext,
    Timing,
    get_dev_var,
    save_plot,
    store_dev_var,
)


class TestTiming(TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_timing_milliseconds(self, mock_stdout):
        with Timing("Task A: ", unit="ms"):
            time.sleep(0.01)  # Sleep for 10 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task A: " in output)
        self.assertTrue("ms" in output)
        elapsed_time = float(output.split()[2])
        self.assertAlmostEqual(
            elapsed_time, 10, delta=5
        )  # Allow delta for timing imprecision

    @patch("sys.stdout", new_callable=StringIO)
    def test_timing_seconds(self, mock_stdout):
        with Timing("Task B: ", unit="s"):
            time.sleep(0.5)  # Sleep for 500 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task B: " in output)
        self.assertTrue("s" in output)
        elapsed_time = float(output.split()[2])
        self.assertAlmostEqual(elapsed_time, 0.5, delta=0.1)

    @patch("sys.stdout", new_callable=StringIO)
    def test_timing_minutes(self, mock_stdout):
        with Timing("Task C: ", unit="m"):
            time.sleep(1)  # Sleep for 1 second
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task C: " in output)
        self.assertTrue("m" in output)
        elapsed_time = float(output.split()[2])
        self.assertAlmostEqual(elapsed_time, 1 / 60, delta=0.005)

    @patch("sys.stdout", new_callable=StringIO)
    def test_on_exit_callback(self, mock_stdout):
        def custom_on_exit(elapsed_time_ns):
            return f" - Time in nanoseconds: {elapsed_time_ns}"

        with Timing("Task D: ", unit="ms", on_exit=custom_on_exit):
            time.sleep(0.02)  # Sleep for 20 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task D: " in output)
        self.assertTrue("ms" in output)
        self.assertTrue("Time in nanoseconds" in output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_disabled_timing(self, mock_stdout):
        with Timing("Task E: ", unit="ms", enabled=False):
            time.sleep(0.01)  # Sleep for 10 milliseconds
        self.assertEqual(mock_stdout.getvalue().strip(), "")

    @patch("sys.stdout", new_callable=StringIO)
    def test_default_unit(self, mock_stdout):
        with Timing("Task F: "):
            time.sleep(0.01)  # Sleep for 10 milliseconds
        output = mock_stdout.getvalue().strip()
        self.assertTrue("Task F: " in output)
        self.assertTrue("ms" in output)

    def test_invalid_unit(self):
        with self.assertRaises(KeyError):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with Timing("Task G: ", unit="invalid"):
                    time.sleep(0.01)  # Sleep for 10 milliseconds
                mock_stdout.getvalue().strip()

    def test_short_sleep(self):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with Timing("Short Task: ", unit="ns"):
                time.sleep(0.000001)  # Sleep for 1 microsecond
            output = mock_stdout.getvalue().strip()
            self.assertTrue("ns" in output)
            elapsed_time = float(output.split()[2])
            self.assertGreater(elapsed_time, 0)


class TestDev(TestCase):
    def setUp(self):
        self.dev = Dev()  # Reference the singleton instance
        self.dev.clear_vars()  # Clear all variables before each test
        self.test_file = Path("./tests/.test_assets/dev_vars.json")

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_singleton_instance(self):
        dev1 = Dev()
        dev2 = Dev()
        self.assertIs(dev1, dev2)  # Both instances should point to the same object

    def test_store_and_get_var(self):
        self.dev.store_var("test_key", 123)
        self.assertEqual(self.dev.get_var("test_key"), 123)
        self.assertIsNone(self.dev.get_var("non_existent_key"))  # Key should not exist

    def test_store_var_invalid_key(self):
        with self.assertRaises(ValueError):
            self.dev.store_var(123, "value")  # Key must be a string

    def test_delete_var(self):
        self.dev.store_var("test_key", 123)
        self.dev.delete_var("test_key")
        with self.assertRaises(KeyError):
            self.dev.delete_var("test_key")  # Key should not exist

    def test_clear_vars(self):
        self.dev.store_var("key1", "value1")
        self.dev.store_var("key2", "value2")
        self.dev.clear_vars()
        self.assertEqual(len(self.dev.variables), 0)  # No variables should remain

    def test_save_variables(self):
        self.dev.store_var("key1", "value1")
        self.dev.save_variables(self.test_file)
        with open(self.test_file) as f:
            dev_vars = json.load(f)
        dev_vars["key1"] == "value1"

    def test_load_variables(self):
        self.dev.store_var("key1", "value1")
        self.dev.save_variables(self.test_file)
        self.dev.load_variables(self.test_file)
        self.assertEqual(self.dev.get_var("key1"), "value1")

    def test_load_variables_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.dev.load_variables("non_existent_file.json")

    def test_store_var_using_global_function(self):
        store_dev_var("global_key", "global_value")
        self.assertEqual(get_dev_var("global_key"), "global_value")

    def test_thread_safety(self):
        def store_in_thread(dev, key, value):
            dev.store_var(key, value)

        thread1 = Thread(target=store_in_thread, args=(self.dev, "key1", "value1"))
        thread2 = Thread(target=store_in_thread, args=(self.dev, "key2", "value2"))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        # Both keys should exist in the dictionary
        self.assertEqual(self.dev.get_var("key1"), "value1")
        self.assertEqual(self.dev.get_var("key2"), "value2")

    def test_save_empty_variables(self):
        self.dev.clear_vars()
        with self.assertRaises(ValueError):
            self.dev.save_variables(self.test_file)


class TestSavePlotContext(TestCase):
    def setUp(self):
        self.test_file = Path("./tests/.test_assets/test_plot.png")
        self.test_file_jpg = Path("./tests/.test_assets/test_plot.jpg")

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_file_jpg.exists():
            self.test_file_jpg.unlink()

    def test_single_axes_save(self):
        with SavePlotContext(self.test_file) as ctx:
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, label="Sine wave")
            ax.legend()
            ctx.axes = ax  # Assign axes to context
        self.assertTrue(self.test_file.exists())
        self.assertGreater(
            self.test_file.stat().st_size, 0
        )  # Ensure the file is not empty

    def test_multiple_axes_save(self):
        with SavePlotContext(self.test_file) as ctx:
            fig, axs = plt.subplots(2, 2)
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            for ax in axs.flat:
                ax.plot(x, y)
            ctx.axes = axs  # Assign the array of Axes to the context
        self.assertTrue(self.test_file.exists())
        self.assertGreater(
            self.test_file.stat().st_size, 0
        )  # Ensure the file is not empty

    def test_save_as_jpg(self):
        with SavePlotContext(self.test_file_jpg, file_format="jpg") as ctx:
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            ctx.axes = ax

        self.assertTrue(self.test_file_jpg.exists())
        self.assertGreater(
            self.test_file_jpg.stat().st_size, 0
        )  # Ensure the file is not empty

    def test_save_with_dpi(self):
        with SavePlotContext(self.test_file, dpi=150) as ctx:
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            ctx.axes = ax
        self.assertTrue(self.test_file.exists())
        self.assertGreater(
            self.test_file.stat().st_size, 0
        )  # Ensure the file is not empty

    def test_invalid_axes_type(self):
        with self.assertRaises(TypeError):
            with SavePlotContext(self.test_file) as ctx:
                ctx.axes = "invalid_axes_type"  # Assign invalid axes

    def test_no_axes_assigned(self):
        with self.assertRaises(TypeError):
            with SavePlotContext(self.test_file):
                pass  # No axes assigned

    def test_save_list_of_axes(self):
        with SavePlotContext(self.test_file) as ctx:
            fig, axs = plt.subplots(1, 2)
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            for ax in axs:
                ax.plot(x, y)
            ctx.axes = list(axs)  # Convert ndarray to list and assign it

        self.assertTrue(self.test_file.exists())
        self.assertGreater(self.test_file.stat().st_size, 0)

    def test_nonexistent_directory(self):
        with self.assertRaises(FileNotFoundError):
            with SavePlotContext("non_existent_directory/test_plot.png"):
                pass  # The directory does not exist

    def test_save_with_nonexistent_directory(self):
        with self.assertRaises(FileNotFoundError):
            with SavePlotContext("non_existent_directory/test_plot.png") as ctx:
                fig, ax = plt.subplots()
                ax.plot([0, 1, 2], [0, 1, 2])
                ctx.axes = ax


class TestSavePlot(TestCase):
    def test_save_plot_single_axes(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = Path(tmpdirname) / "test.png"
            with save_plot(file_path) as context:
                fig, ax = plt.subplots()
                context.axes = ax
            self.assertTrue(file_path.is_file())

    def test_save_plot_multiple_axes(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = Path(tmpdirname) / "test.png"
            with save_plot(file_path) as context:
                fig, axes = plt.subplots(2, 2)
                context.axes = axes
            self.assertTrue(file_path.is_file())

    def test_save_plot_invalid_axes(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = Path(tmpdirname) / "test.png"
            with self.assertRaises(TypeError):
                with save_plot(file_path) as context:
                    context.axes = "invalid"


if __name__ == "__main__":
    unittest.main()
