import hashlib
import re
import uuid
from abc import ABC
from dataclasses import dataclass, field, replace
from os import PathLike
from typing import Any, Dict, Iterator, List, Optional

from ..utils import iprint


def validate_hex_string(value: str) -> str:
    if not re.match(r"^[0-9a-fA-F]{16}$", value):
        raise ValueError(f"The value {value} is not a valid hex string.")
    return value

def create_notebook_cell_id(notebook_seed: str, cell_seed: str) -> str:
    seed = notebook_seed + cell_seed
    hasher = hashlib.sha256(seed.encode())
    hash = hasher.hexdigest()
    return hash[:16]

@dataclass(frozen=True)
class Notebook:
    cells: 'NotebookCells'
    metadata: 'NotebookMetadata'
    nbformat: int
    nbformat_minor: int
    name: str
    path: PathLike = field(default='')
    notebook_id: str = field(default=uuid.uuid4().hex, metadata={'validator': validate_hex_string})
    def __post_init__(self):
        new_cells = []
        for i, cell in enumerate(object.__getattribute__(self, 'cells')):
            cell_id = create_notebook_cell_id(self.notebook_id, str(i))
            new_cell = replace(cell, cell_id=validate_hex_string(cell_id))
            new_cells.append(new_cell)
        object.__setattr__(self, 'cells', new_cells)

@dataclass(frozen=True)
class NotebookCells:
    cells: List['NotebookCell']
    def __iter__(self) -> Iterator['NotebookCell']:
        return iter(self.cells)
    def __getitem__(self, index) -> 'NotebookCell':
        return self.cells[index]
    def __len__(self) -> int:
        return len(self.cells)

@dataclass(frozen=True)
class NotebookSection(NotebookCells):
    cells: NotebookCells  # This should be a NotebookCells instance
    def __post_init__(self):
        super().__init__(self.cells)
        # Ensure self.cells is a NotebookCells instance
        if not isinstance(self.cells, NotebookCells):
            raise ValueError("cells must be an instance of NotebookCells.")
        # Check if the first cell is a MarkdownCell
        try:
            first_cell = next(iter(self.cells))
        except StopIteration:
            raise ValueError("The cells list is empty.")
        if not isinstance(first_cell, MarkdownCell):
            raise ValueError("The first cell must be a MarkdownCell.")
    
@dataclass(frozen=True)
class NotebookCell:
    cell_type: str
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)
    source: List[str] = field(default_factory=list)
    cell_id: str = field(default="", metadata={'validator': validate_hex_string})

@dataclass(frozen=True)
class MarkdownCell(NotebookCell):
    def __post_init__(self):
        # Validate that cell_type is 'code', raise an error if it is not.
        if object.__getattribute__(self, 'cell_type') != 'markdown':
            raise ValueError(f"cell_type of MarkdownCell must be 'markdown', got {object.__getattribute__(self, 'cell_type')}")
        object.__setattr__(self, 'cell_type', 'markdown')

@dataclass(frozen=True)
class CodeCell(NotebookCell):
    def __post_init__(self):
        # Validate that cell_type is 'code', raise an error if it is not.
        if object.__getattribute__(self, 'cell_type') != 'code':
            raise ValueError(f"cell_type of CodeCell must be 'code', got {object.__getattribute__(self, 'cell_type')}")
        object.__setattr__(self, 'cell_type', 'code')

@dataclass(frozen=True, init=True)
class NotebookCellFactory:
    cell_types = {
        'code': CodeCell,
        'markdown': MarkdownCell
    }
    @staticmethod
    def create_cell(cell_type: str, *args, **kwargs):
        cell_class = NotebookCellFactory.cell_types.get(cell_type.lower(), NotebookCell)
        cell = cell_class(cell_type=cell_type, *args, **kwargs)
        return cell

@dataclass(frozen=True)
class NotebookMetadata:
    kernelspec: 'KernelSpec'
    language_info: 'LanguageInfo'

@dataclass
class KernelSpec:
    display_name: str
    language: str
    name: str

@dataclass
class LanguageInfo:
    codemirror_mode: 'CodeMirrorMode'
    file_extension: str
    mimetype: str
    name: str
    nbconvert_exporter: str
    pygments_lexer: str
    version: str

@dataclass
class CodeMirrorMode:
    name: str
    version: int

