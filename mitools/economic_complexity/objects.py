import json
import statistics
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
from pandas import DataFrame


@dataclass(frozen=True)
class Product:
    code: int
    name: str
    pci: float
    value: float


@dataclass(frozen=True)
class ProductsBasket:
    products: List[Product]

    def __post_init__(self):
        # Check for unique codes and names
        codes = set()
        names = set()
        for product in self.products:
            if product.code in codes:
                raise ValueError(f"Duplicate product code found: {product.code}")
            if product.name in names:
                raise ValueError(f"Duplicate product name found: {product.name}")
            codes.add(product.code)
            names.add(product.name)
        products_data = [
            [product.code, product.name, product.pci, product.value]
            for product in self.products
        ]
        df = DataFrame(products_data, columns=["Code", "Name", "PCI", "Value"])
        object.__setattr__(self, "products_df", df)

    @property
    def mean(self):
        return statistics.mean(product.pci for product in self.products)

    @property
    def std(self):
        return statistics.stdev(product.pci for product in self.products)

    @property
    def minimum(self):
        return min(product.pci for product in self.products)

    @property
    def maximum(self):
        return max(product.pci for product in self.products)

    @property
    def median(self):
        return median(product.pci for product in self.products)

    @property
    def range(self):
        return {
            "min": self.minimum,
            "mean": self.mean,
            "median": self.median,
            "max": self.maximum,
        }

    def __len__(self):
        return len(self.products)

    def get_quantiles(self, n):
        pcis = [product.pci for product in self.products]
        if pcis:
            quantiles = [np.percentile(pcis, i * 100.0 / n) for i in range(n + 1)]
        else:
            quantiles = []
        return quantiles

    def products_closest_to_quantiles(self, n=10):
        quantiles = self.get_quantiles(n=n)
        closest_products = []
        for quantile in quantiles:
            closest_product = min(
                self.products, key=lambda product: abs(product.pci - quantile)
            )
            closest_products.append(closest_product)
        return closest_products

    def product_list(self, ascending=True):
        return DataFrame(
            [
                [product.code, product.name, product.pci, product.value]
                for product in self.products
            ],
            columns=["Code", "Name", "PCI", "Value"],
        )


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
