import contextlib
import time
from typing import Callable, Literal, Optional


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
