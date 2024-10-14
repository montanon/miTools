import tempfile
import time
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from mitools.context import Context, ContextVar, Timing
from mitools.utils import SavePlotContext, save_plot

VARIABLE = ContextVar("VARIABLE", 0)


class TestContextVars(unittest.TestCase):
    # Ensuring that the test does not modify variables outside the tests.
    ctx = Context()

    def setUp(self):
        TestContextVars.ctx.__enter__()

    def tearDown(self):
        TestContextVars.ctx.__exit__()

    def test_initial_value_is_set(self):
        _TMP = ContextVar("_TMP", 5)
        self.assertEqual(_TMP.value, 5)

    def test_multiple_creation_ignored(self):
        _TMP2 = ContextVar("_TMP2", 1)
        _TMP2 = ContextVar("_TMP2", 2)
        self.assertEqual(_TMP2.value, 1)

    def test_new_var_inside_context(self):
        # Creating a _new_ variable inside a context should not have any effect on its scope (?)
        with Context(VARIABLE=1):
            _TMP3 = ContextVar("_TMP3", 1)
        _TMP3 = ContextVar("_TMP3", 2)
        self.assertEqual(_TMP3.value, 1)

    def test_value_accross_modules(self):
        # Mocking module import by invoking the code but not in our globals().
        exec('from mitools.context import ContextVar;C = ContextVar("C", 13)', {})  # pylint:disable=exec-used
        # It should not matter that the first creation was in another module.
        C = ContextVar("C", 0)
        self.assertEqual(C.value, 13)

    def test_assignment_across_modules(self):
        B = ContextVar("B", 1)
        # local assignment
        B.value = 2
        self.assertEqual(B.value, 2)
        # Assignment in another module.
        exec(
            'from mitools.context import ContextVar;B = ContextVar("B", 0);B.value = 3;',
            {},
        )  # pylint:disable=exec-used
        # Assignment in another module should affect this one as well.
        self.assertEqual(B.value, 3)

    def test_context_assignment(self):
        with Context(VARIABLE=1):
            self.assertEqual(VARIABLE.value, 1)
        self.assertEqual(VARIABLE.value, 0)

    def test_unknown_param_to_context(self):
        with self.assertRaises(KeyError):
            with Context(SOMETHING_ELSE=1):
                pass

    def test_inside_context_assignment(self):
        with Context(VARIABLE=4):
            # What you can and cannot do inside a context.
            # 1. This type of statement has no effect.
            VARIABLE = ContextVar("VARIABLE", 0)
            self.assertTrue(
                VARIABLE >= 4,
                "ContextVars inside contextmanager may not set a new value",
            )

            # 2. The call syntax however has a local effect.
            VARIABLE.value = 13
            self.assertTrue(
                VARIABLE.value == 13,
                "Call syntax however works inside a contextmanager.",
            )

        # Related to 2. above. Note that VARIABLE is back to 0 again as expected.
        self.assertEqual(VARIABLE.value, 0)

    def test_new_var_inside_context_other_module(self):
        with Context(VARIABLE=1):
            _NEW2 = ContextVar("_NEW2", 0)
        _NEW2 = ContextVar("_NEW2", 1)
        self.assertEqual(_NEW2.value, 0)

        code = """\
from mitools.context import Context, ContextVar
with Context(VARIABLE=1):
  _NEW3 = ContextVar("_NEW3", 0)"""
        exec(code, {})  # pylint:disable=exec-used
        # While _NEW3 was created in an outside scope it should still work the same as above.
        _NEW3 = ContextVar("_NEW3", 1)
        self.assertEqual(_NEW3.value, 0)

    def test_nested_context(self):
        with Context(VARIABLE=1):
            with Context(VARIABLE=2):
                with Context(VARIABLE=3):
                    self.assertEqual(VARIABLE.value, 3)
                self.assertEqual(VARIABLE.value, 2)
            self.assertEqual(VARIABLE.value, 1)
        self.assertEqual(VARIABLE.value, 0)

    def test_decorator(self):
        @Context(VARIABLE=1, DEBUG=4)
        def test():
            self.assertEqual(VARIABLE.value, 1)

        self.assertEqual(VARIABLE.value, 0)
        test()
        self.assertEqual(VARIABLE.value, 0)

    def test_context_exit_reverts_updated_values(self):
        D = ContextVar("D", 1)
        D.value = 2
        with Context(D=3):
            ...
        assert (
            D.value == 2
        ), f"Expected D to be 2, but was {D.value}. Indicates that Context.__exit__ did not restore to the correct value."


class TestTiming(unittest.TestCase):
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


class TestSavePlotContext(unittest.TestCase):
    def setUp(self):
        self.regression_plot = Path("test_plot.png")
        self.context = SavePlotContext(self.regression_plot)

    def tearDown(self):
        if self.regression_plot.exists():
            self.regression_plot.unlink()

    def test_exit_with_axes(self):
        self.context.axes = plt.figure().subplots()
        self.context.__exit__(None, None, None)
        self.assertTrue(self.regression_plot.exists())

    def test_exit_with_axes_array(self):
        self.context.axes = np.array([plt.figure().subplots(), plt.figure().subplots()])
        self.context.__exit__(None, None, None)
        self.assertTrue(self.regression_plot.exists())

    def test_exit_with_invalid_axes(self):
        self.context.axes = "invalid"
        with self.assertRaises(TypeError):
            self.context.__exit__(None, None, None)


class TestSavePlot(unittest.TestCase):
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
