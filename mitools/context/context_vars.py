import contextlib
import functools
import os
from typing import ClassVar, Dict, List, Optional, Tuple, TypeVar

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
ASSERT = ContextVar("ASSERT", 1)
RANDOMSTATE = ContextVar("RANDOMSTATE", 0)
RECALCULATE = ContextVar("RECALCULATE", 0)
