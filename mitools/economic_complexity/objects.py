import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
            closest_product = min(self.products, key=lambda product: abs(product.pci - quantile))
            closest_products.append(closest_product)
        return closest_products


class StringConverter:

    def __init__(self, relations: Dict[str,str], case_sensitive: Optional[bool]=True):
        self.case_sensitive = case_sensitive
        self.pretty_to_ugly = {}
        self.ugly_to_pretty = {}
        for pretty, ugly in relations.items():
            self.add_relation(pretty, ugly)

    def validate_relation(self, pretty: str, ugly: str) -> (str, str):
        if not self.case_sensitive:
            pretty, ugly = pretty.lower(), ugly.lower()
        if pretty in self.pretty_to_ugly or ugly in self.ugly_to_pretty:
            raise ValueError(f"Non-bijective mapping with pretty or ugly string found: {pretty} {ugly}")
        return pretty, ugly
    
    def add_relation(self, pretty: str, ugly: str) -> None:
        pretty, ugly = self.validate_relation(pretty, ugly)
        self.pretty_to_ugly[pretty] = ugly
        self.ugly_to_pretty[ugly] = pretty

    def prettify_str(self, ugly_str: str) -> str:
        if not self.case_sensitive:
            ugly_str = ugly_str.lower()
        if ugly_str not in self.ugly_to_pretty:
            raise ValueError(f"No pretty string found for '{ugly_str}'")
        return self.ugly_to_pretty[ugly_str]
    
    def prettify_strs(self, ugly_strs: str) -> List[str]:
        return [self.prettify_str(ugly_str) for ugly_str in ugly_strs]
    
    def uglify_str(self, pretty_str: str) -> str:
        if not self.case_sensitive:
            pretty_str = pretty_str.lower()
        if pretty_str not in self.pretty_to_ugly:
            raise ValueError(f"No ugly string found for '{pretty_str}'")
        return self.pretty_to_ugly[pretty_str]
    
    def uglify_strs(self, pretty_strs: List[str]) -> List[str]:
        return [self.uglify_str(pretty_str) for pretty_str in pretty_strs]
    