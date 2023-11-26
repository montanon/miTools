import array
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
        data = array.array('B', (0 for _ in range(arr_size)))
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
        data &= ~(1 << bit_idx) # clear bit
        data |= (bool(bit) * (1 << bit_idx)) # set bit
        self.data[arr_idx] = data

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self)})"
    
    def __len__(self):
        return self.size
