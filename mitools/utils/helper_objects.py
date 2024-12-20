import array
import types
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    MutableMapping,
    ValuesView,
)
from dataclasses import dataclass
from typing import Any


@dataclass
class BitArray:
    data: array.array
    size: int

    @classmethod
    def zeros(cls, n: int):
        arr_size, remainder = divmod(n, 8)
        if remainder:
            arr_size += 1
        data = array.array("B", (0 for _ in range(arr_size)))
        return cls(data=data, size=n)

    def _check_index(self, n):
        if not (0 <= n < self.size):
            raise IndexError("Index out of bounds")

    def __getitem__(self, n):
        return self.get_index(n)

    def get_index(self, n):
        self._check_index(n)
        arr_idx, bit_idx = divmod(n, 8)
        return (self.data[arr_idx] >> bit_idx) & 0b1

    def __setitem__(self, n, bit):
        self._check_index(n)
        arr_idx, bit_idx = divmod(n, 8)
        data = self.data[arr_idx]
        data &= ~(1 << bit_idx)  # clear bit
        data |= bool(bit) * (1 << bit_idx)  # set bit
        self.data[arr_idx] = data

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"

    def __len__(self):
        return self.size


class LazyDict(dict):
    def load(self):
        pass

    def _lazy(self, method, *args):
        if dict.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(dict, method), self))
        return getattr(dict, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")

    def __len__(self):
        return self._lazy("__len__")

    def __iter__(self):
        return self._lazy("__iter__")

    def __contains__(self, *args):
        return self._lazy("__contains__", *args)

    def __getitem__(self, *args):
        return self._lazy("__getitem__", *args)

    def __setitem__(self, *args):
        return self._lazy("__setitem__", *args)

    def setdefault(self, *args):
        return self._lazy("setdefault", *args)

    def get(self, *args, **kwargs):
        return self._lazy("get", *args)

    def items(self):
        return self._lazy("items")

    def keys(self):
        return self._lazy("keys")

    def values(self):
        return self._lazy("values")

    def update(self, *args):
        return self._lazy("update", *args)

    def pop(self, *args):
        return self._lazy("pop", *args)

    def popitem(self, *args):
        return self._lazy("popitem", *args)


class LazyList(list):
    def load(self):
        pass

    def _lazy(self, method, *args):
        if list.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(list, method), self))
        return getattr(list, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")

    def __len__(self):
        return self._lazy("__len__")

    def __iter__(self):
        return self._lazy("__iter__")

    def __contains__(self, *args):
        return self._lazy("__contains__", *args)

    def insert(self, *args):
        return self._lazy("insert", *args)

    def append(self, *args):
        return self._lazy("append", *args)

    def extend(self, *args):
        return self._lazy("extend", *args)

    def remove(self, *args):
        return self._lazy("remove", *args)

    def pop(self, *args):
        return self._lazy("pop", *args)

    def __getitem__(self, *args):
        return self._lazy("__getitem__", *args)

    def __setitem__(self, *args):
        return self._lazy("__setitem__", *args)


def _new_attr_dict_(*args):
    attr_dict = AttrDict()
    for k, v in args:
        attr_dict[k] = v
    return attr_dict


class AttrDict(MutableMapping):
    def update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.__private_dict__.update(*args, **kwargs)

    def clear(self) -> None:
        self.__private_dict__.clear()

    def copy(self) -> "AttrDict":
        ad = AttrDict()
        for key in self.__private_dict__.keys():
            ad[key] = self.__private_dict__[key]
        return ad

    def keys(self) -> KeysView:
        return self.__private_dict__.keys()

    def items(self) -> ItemsView:
        return self.__private_dict__.items()

    def values(self) -> ValuesView:
        return self.__private_dict__.values()

    def pop(self, key: str, default: Any = None) -> Any:
        return self.__private_dict__.pop(key, default)

    def __reduce__(
        self,
    ):
        return _new_attr_dict_, tuple((k, v) for k, v in self.items())

    def __len__(self) -> int:
        return self.__private_dict__.__len__()

    def __repr__(self) -> str:
        out = self.__private_dict__.__str__()
        return "AttrDict" + out

    def __str__(self) -> str:
        return self.__repr__()

    def __init__(self, *args, **kwargs) -> None:
        self.__dict__["__private_dict__"] = dict(*args, **kwargs)

    def __contains__(self, item: str) -> bool:
        return self.__private_dict__.__contains__(item)

    def __getitem__(self, item: str) -> Any:
        return self.__private_dict__[item]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "__private_dict__":
            raise KeyError("__private_dict__ is reserved and cannot be set.")
        self.__private_dict__[key] = value

    def __delitem__(self, key: str) -> None:
        del self.__private_dict__[key]

    def __getattr__(self, key: str) -> Any:
        if key not in self.__private_dict__:
            raise AttributeError
        return self.__private_dict__[key]

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "__private_dict__":
            raise AttributeError("__private_dict__ is invalid")
        self.__private_dict__[key] = value

    def __delattr__(self, key: str) -> None:
        del self.__private_dict__[key]

    def __dir__(self) -> Iterable:
        out = [str(key) for key in self.__private_dict__.keys()]
        out += list(super().__dir__())
        filtered = [key for key in out if key.isidentifier()]
        return sorted(set(filtered))

    def __iter__(self) -> Iterator:
        return self.__private_dict__.__iter__()
