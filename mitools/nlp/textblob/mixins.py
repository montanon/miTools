import sys
from abc import ABC, abstractmethod


class ComparableMixin(ABC):
    @abstractmethod
    def comparable_key(self):
        raise NotImplementedError("comparable_key must be implemented")

    def _compare(self, other, method):
        try:
            return method(self.comparable_key(), other.comparable_key())
        except (AttributeError, TypeError):
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)


class BlobComparableMixin(ComparableMixin):
    def _compare(self, other, method):
        if isinstance(other, (str, bytes)):
            return method(self.comparable_key(), other)
        return super()._compare(other, method)


class StringlikeMixin(ABC):
    @abstractmethod
    def string_key(self):
        raise NotImplementedError("string_key must be implemented")

    def __repr__(self):
        class_name = self.__class__.__name__
        text = str(self)
        return f'{class_name}("{text}")'

    def __str__(self):
        return self.string_key()

    def __len__(self):
        return len(self.string_key())

    def __iter__(self):
        return iter(self.string_key())

    def __contains__(self, sub):
        return sub in self.string_key()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.string_key()[index]
        else:
            return self.__class__(self.string_key()[index])

    def find(self, sub, start=0, end=sys.maxsize):
        return self.string_key().find(sub, start, end)

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.string_key().rfind(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self.string_key().index(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.string_key().rindex(sub, start, end)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.string_key().startswith(prefix, start, end)

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.string_key().endswith(suffix, start, end)

    def title(self):
        return self.__class__(self.string_key().title())

    def format(self, *args, **kwargs):
        return self.__class__(self.string_key().format(*args, **kwargs))

    def split(self, sep=None, maxsplit=sys.maxsize):
        return self.string_key().split(sep, maxsplit)

    def strip(self, chars=None):
        return self.__class__(self.string_key().strip(chars))

    def upper(self):
        return self.__class__(self.string_key().upper())

    def lower(self):
        return self.__class__(self.string_key().lower())

    def join(self, iterable):
        return self.__class__(self.string_key().join(iterable))

    def replace(self, old, new, count=sys.maxsize):
        return self.__class__(self.string_key().replace(old, new, count))
