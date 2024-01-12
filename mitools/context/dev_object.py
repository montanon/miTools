from typing import Any


class Dev:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Dev, cls).__new__(cls)
            cls._instance.variables = {}
        return cls._instance
    def store_var(self, key, value):
        self.variables[key] = value
    def get_var(self, key):
        return self.variables[key]
    def __repr__(self):
        repr_str = "Dev("
        for key, value in self.variables.items():
            repr_str += f"\n    {key}: {value},"
        repr_str += "\n)" if self.variables else ")"
        return repr_str
# Creating a global instance in the module
DEV = Dev()

# Functions for external usage
def store_dev_var(key: str, value: Any) -> None:
    DEV.store_var(key, value)
def get_dev_var(key: str) -> Any:
    return DEV.get_var(key)
