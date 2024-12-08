import sys


class ComparableMixin:
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
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
            return method(self._cmpkey(), other)
        return super()._compare(other, method)


class StringlikeMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        text = str(self)
        return f'{class_name}("{text}")'

    def __str__(self):
        return self._strkey()

    def __len__(self):
        return len(self._strkey())

    def __iter__(self):
        return iter(self._strkey())

    def __contains__(self, sub):
        return sub in self._strkey()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._strkey()[index]  # Just return a single character
        else:
            # Return a new blob object
            return self.__class__(self._strkey()[index])

    def find(self, sub, start=0, end=sys.maxsize):
        return self._strkey().find(sub, start, end)

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self._strkey().rfind(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self._strkey().index(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self._strkey().rindex(sub, start, end)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self._strkey().startswith(prefix, start, end)

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self._strkey().endswith(suffix, start, end)

    def title(self):
        return self.__class__(self._strkey().title())

    def format(self, *args, **kwargs):
        return self.__class__(self._strkey().format(*args, **kwargs))

    def split(self, sep=None, maxsplit=sys.maxsize):
        return self._strkey().split(sep, maxsplit)

    def strip(self, chars=None):
        return self.__class__(self._strkey().strip(chars))

    def upper(self):
        return self.__class__(self._strkey().upper())

    def lower(self):
        return self.__class__(self._strkey().lower())

    def join(self, iterable):
        return self.__class__(self._strkey().join(iterable))

    def replace(self, old, new, count=sys.maxsize):
        return self.__class__(self._strkey().replace(old, new, count))
