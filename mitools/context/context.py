import contextlib
import functools
import os
import time
from typing import Callable, ClassVar, Dict, List, Literal, Optional, Tuple, TypeVar

ShapeType = Tuple[int, ...]
T = TypeVar("T")


@functools.lru_cache(maxsize=None)
def getenv(key: str, default: Optional[T] = 0) -> T:
    return type(default)(os.getenv(key, default))


class Context(contextlib.ContextDecorator):
    stack: ClassVar[List[Dict[str, int]]] = [{}]

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        Context.stack[-1] = {
            k: o.value for k, o in ContextVar._cache.items()
        }  # Store current state.
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v  # Update to new temporary state.
        Context.stack.append(
            self.kwargs
        )  # Store the temporary state so we know what to undo later.

    def __exit__(self, *args):
        for k in Context.stack.pop():
            ContextVar._cache[k].value = Context.stack[-1].get(
                k, ContextVar._cache[k].value
            )


class ContextVar:
    _cache: ClassVar[Dict[str, "ContextVar"]] = {}
    __slots__ = "value"
    value: int

    def __new__(cls, key, default_value):
        if key in ContextVar._cache:
            return ContextVar._cache[key]
        instance = ContextVar._cache[key] = super().__new__(cls)
        instance.value = getenv(key, default_value)
        return instance

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x

    def __eq__(self, x):
        return self.value == x


# * Can't be overwritten after it is defined
# * Needs to be updated through context
DEBUG = ContextVar("DEBUG", 0)  # DEBUG levels could be similar to logging.debug levels.
DISPLAY = ContextVar("DISPLAY", 0)
ASSERT = ContextVar("ASSERT", 0)
RANDOMSTATE = ContextVar("RANDOMSTATE", 0)


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
