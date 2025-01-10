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
from typing import Any, Dict, List


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


class StringMapper:
    def __init__(
        self,
        relations: Dict[str, str],
        case_sensitive: Optional[bool] = True,
        pass_if_mapped: Optional[bool] = False,
    ):
        self.case_sensitive = case_sensitive
        self.pass_if_mapped = pass_if_mapped
        self.pretty_to_ugly = {}
        self.ugly_to_pretty = {}
        for pretty, ugly in relations.items():
            self.add_relation(pretty, ugly)

    def validate_relation(self, pretty: str, ugly: str) -> (str, str):
        if not self.case_sensitive:
            pretty, ugly = pretty.lower(), ugly.lower()
        if pretty in self.pretty_to_ugly or ugly in self.ugly_to_pretty:
            raise ValueError(
                f"Non-bijective mapping with pretty or ugly string found: {pretty} {ugly}"
            )
        return pretty, ugly

    def add_relation(self, pretty: str, ugly: str) -> None:
        pretty, ugly = self.validate_relation(pretty, ugly)
        self.pretty_to_ugly[pretty] = ugly
        self.ugly_to_pretty[ugly] = pretty

    def prettify_str(self, ugly_str: str) -> str:
        if not self.case_sensitive:
            ugly_str = ugly_str.lower()
        if ugly_str in self.ugly_to_pretty:
            return self.ugly_to_pretty[ugly_str]
        elif self.pass_if_mapped and self.is_pretty(ugly_str):
            return ugly_str
        else:
            raise ValueError(f"No pretty string found for '{ugly_str}'")

    def prettify_strs(self, ugly_strs: str) -> List[str]:
        return [self.prettify_str(ugly_str) for ugly_str in ugly_strs]

    def uglify_str(self, pretty_str: str) -> str:
        if not self.case_sensitive:
            pretty_str = pretty_str.lower()
        if pretty_str in self.pretty_to_ugly:
            return self.pretty_to_ugly[pretty_str]
        elif self.pass_if_mapped and self.is_ugly(pretty_str):
            return pretty_str
        else:
            raise ValueError(f"No ugly string found for '{pretty_str}'")

    def uglify_strs(self, pretty_strs: List[str]) -> List[str]:
        return [self.uglify_str(pretty_str) for pretty_str in pretty_strs]

    def remap_str(self, string):
        if (
            not self.case_sensitive
            and (
                string.lower() in self.pretty_to_ugly
                or string.lower() in self.ugly_to_pretty
            )
        ) or (string in self.pretty_to_ugly or string in self.ugly_to_pretty):
            if string in self.pretty_to_ugly or (
                not self.case_sensitive and string.lower() in self.pretty_to_ugly
            ):
                return self.uglify_str(string)
            else:
                return self.prettify_str(string)
        else:
            raise ValueError(f"String '{string}' is not mapped")

    def remap_strs(self, strings: List[str]) -> List[str]:
        if all(self.is_pretty(string) for string in strings):
            return [self.uglify_str(string) for string in strings]
        elif all(self.is_ugly(string) for string in strings):
            return [self.prettify_str(string) for string in strings]
        else:
            raise ValueError(
                "All strings must be either pretty or ugly before remapping"
            )

    def is_pretty(self, string: str) -> bool:
        if not self.case_sensitive:
            string = string.lower()
        return string in self.pretty_to_ugly

    def is_ugly(self, string: str) -> bool:
        if not self.case_sensitive:
            string = string.lower()
        return string in self.ugly_to_pretty

    def save_mappings(self, file_path):
        data = {
            "case_sensitive": self.case_sensitive,
            "pass_if_mapped": self.pass_if_mapped,
            "relations": {pretty: ugly for pretty, ugly in self.pretty_to_ugly.items()},
        }
        with open(file_path, "w") as file:
            json.dump(data, file)

    @staticmethod
    def load_mappings(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            case_sensitive = data["case_sensitive"]
            pass_if_mapped = data["pass_if_mapped"]
            pretty_to_ugly = data["relations"]
            return StringMapper(
                relations=pretty_to_ugly,
                case_sensitive=case_sensitive,
                pass_if_mapped=pass_if_mapped,
            )

    def __eq__(self, other):
        if isinstance(other, StringMapper):
            return (
                self.case_sensitive == other.case_sensitive
                and self.pass_if_mapped == other.pass_if_mapped
                and self.pretty_to_ugly == other.pretty_to_ugly
                and self.ugly_to_pretty == other.ugly_to_pretty
            )
        return False

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(case_sensitive={self.case_sensitive}, "
            f"pass_if_mapped={self.pass_if_mapped}, "
            f"mappings={len(self.pretty_to_ugly)})"
        )

    def __repr__(self) -> str:
        relations_repr = json.dumps(self.pretty_to_ugly, indent=4)
        return (
            f"{self.__class__.__name__}(relations={relations_repr}, "
            f"case_sensitive={self.case_sensitive}, "
            f"pass_if_mapped={self.pass_if_mapped})"
        )
