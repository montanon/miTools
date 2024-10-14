import contextlib
import time
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np


class Timing(contextlib.ContextDecorator):
    TIME_UNITS = {
        "ns": 1e0,  # Nanoseconds (default internal unit)
        "ms": 1e-6,  # Milliseconds
        "s": 1e-9,  # Seconds
        "m": 1e-9 / 60,  # Minutes
    }

    def __init__(
        self,
        prefix: Optional[str] = "",
        on_exit: Optional[Callable[[float], str]] = None,
        enabled: bool = True,
        unit: Literal["ns", "ms", "s", "m"] = "ms",
    ):
        if unit not in self.TIME_UNITS:
            raise KeyError(
                f"Unit {unit} must be in self.TIME_UNITS.keys(), {list(self.TIME_UNITS.keys())}"
            )
        self.prefix = prefix
        self.on_exit = on_exit
        self.enabled = enabled
        self.unit = unit
        self.unit_conversion = self.TIME_UNITS[unit]
        self.unit_label = unit

    def __enter__(self):
        self.start_time = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time_ns = time.perf_counter_ns() - self.start_time
        elapsed_time_converted = elapsed_time_ns * self.unit_conversion
        if self.enabled:
            output = f"{self.prefix}{elapsed_time_converted:.2f} {self.unit_label}"
            if self.on_exit:
                output += self.on_exit(elapsed_time_ns)
            print(output)


class SavePlotContext:
    def __init__(
        self,
        file_path: PathLike,
        dpi: Optional[int] = 300,
        file_format: Literal["png", "jpg", "svg"] = "png",
    ):
        self.file_path = Path(file_path)  # Ensure we have a Path object
        self.axes = None
        self.dpi = dpi
        self.file_format = file_format  # Default to "png"
        # Check if the directory exists
        if not self.file_path.parent.exists():
            raise FileNotFoundError(
                f"Directory {self.file_path.parent} does not exist."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Check if self.axes is a single Axes object
        if isinstance(self.axes, plt.Axes):
            self.axes.figure.savefig(
                self.file_path, dpi=self.dpi, format=self.file_format
            )
        # Check if self.axes is an array or list of Axes objects
        elif isinstance(self.axes, np.ndarray):
            if all(
                isinstance(ax, plt.Axes) for ax in self.axes.flat
            ):  # Handle np.ndarray of Axes
                self.axes.flat[0].figure.savefig(
                    self.file_path, dpi=self.dpi, format=self.file_format
                )
            else:
                raise TypeError(
                    "All elements of the ndarray must be matplotlib Axes objects."
                )
        elif isinstance(self.axes, list):
            if all(isinstance(ax, plt.Axes) for ax in self.axes):  # Handle list of Axes
                self.axes[0].figure.savefig(
                    self.file_path, dpi=self.dpi, format=self.file_format
                )
            else:
                raise TypeError(
                    "All elements of the list must be matplotlib Axes objects."
                )
        else:
            raise TypeError(
                "Must assign a matplotlib Axes, or an array/list of Axes to 'axes' attribute."
            )


@contextmanager
def save_plot(file_path: PathLike):
    context = SavePlotContext(file_path)
    yield context
    context.__exit__(None, None, None)
