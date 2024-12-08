import csv
import json
from collections import OrderedDict

from mitools.nlp.textblob.utils import is_filelike

DEFAULT_ENCODING = "utf-8"


class BaseFormat:
    def __init__(self, fp, **kwargs):
        pass

    def to_iterable(self):
        raise NotImplementedError("Must implement to_iterable() method")

    @classmethod
    def detect(cls, stream):
        raise NotImplementedError("Must implement detect() method")


class DelimitedFormat(BaseFormat):
    def __init__(self, fp, **kwargs):
        BaseFormat.__init__(self, fp, **kwargs)
        reader = csv.reader(fp, delimiter=self.delimiter)
        self.data = [row for row in reader]

    def to_iterable(self):
        return self.data

    @classmethod
    def detect(cls, stream):
        try:
            csv.Sniffer().sniff(stream, delimiters=cls.delimiter)
            return True
        except (csv.Error, TypeError):
            return False


class CSV(DelimitedFormat):
    delimiter = ","


class TSV(DelimitedFormat):
    delimiter = "\t"


class JSON(BaseFormat):
    def __init__(self, fp, **kwargs):
        BaseFormat.__init__(self, fp, **kwargs)
        self.dict = json.load(fp)

    def to_iterable(self):
        return [(d["text"], d["label"]) for d in self.dict]

    @classmethod
    def detect(cls, stream):
        try:
            json.load(stream)
            return True
        except ValueError:
            return False


REGISTRY = OrderedDict(
    [
        ("csv", CSV),
        ("json", JSON),
        ("tsv", TSV),
    ]
)


def detect(fp, max_read: int = 1024):
    if not is_filelike(fp):
        return None
    for Format in REGISTRY.values():
        if Format.detect(fp.read(max_read)):
            fp.seek(0)
            return Format
        fp.seek(0)
    return None


def get_registry():
    return REGISTRY


def register(name, format_class):
    get_registry()[name] = format_class
