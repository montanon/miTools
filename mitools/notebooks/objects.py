import re
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


def validate_hex_string(value: str) -> str:
    if not re.match(r"^[0-9a-fA-F]{8}$", value):
        raise ValueError(f"The value {value} is not a valid hex string.")
    return value

@dataclass(frozen=True)
class Notebook:
    cells: 'NotebookCells'
    metadata: 'NotebookMetadata'
    nbformat: int
    nbformat_minor: int

@dataclass(frozen=True)
class NotebookCells:
    cells: List['NotebookCell']

@dataclass(frozen=True)
class NotebookCell:
    cell_type: str
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)
    source: List[str] = field(default_factory=list)
    cell_id: str = field(default="00000000", metadata={'validator': validate_hex_string})

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

