import array
import types
from dataclasses import dataclass


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
