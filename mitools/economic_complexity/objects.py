import statistics
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Product:
    code: int
    name: str
    pci: float
    value: float
        
class HS2Product(Product):
    pass

class HS4Product(Product):
    pass

class HS6Product(Product):
    pass


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
    
    def __len__(self):
        return len(self.products)

    def get_quantiles(self, n):
        pcis = sorted(product.pci for product in self.products)
        if pcis:
            quantiles = [np.percentile(pcis, i * 100 / n) for i in range(1, n)]
            quantiles.insert(0, pcis[0])
            quantiles.append(pcis[-1])
        else:
            quantiles = []
        return quantiles
    
    def products_closest_to_quantiles(self, n=10):
        quantiles = self.get_quantiles(n=n)
        closest_products = []
        for quantile in quantiles:
            closest_product = min(self.products, key=lambda product: abs(product.pci - quantile))
            closest_products.append(closest_product)
        return closest_products
