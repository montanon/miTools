import statistics
from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Tuple

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
