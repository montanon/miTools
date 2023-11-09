import dataclasses
import json
from typing import Dict, List

from .objects import (CodeMirrorMode, KernelSpec, LanguageInfo, Notebook,
                      NotebookCell, NotebookMetadata)


def save_notebook_as_ipynb(notebook: Notebook, filename: str) -> None:
    # Convert the Notebook data class to a dictionary
    notebook_dict = {
        'cells': [dataclasses.asdict(cell) for cell in notebook.cells],
        'metadata': dataclasses.asdict(notebook.metadata),
        'nbformat': notebook.nbformat,
        'nbformat_minor': notebook.nbformat_minor
    }
    # Convert to a JSON string with formatting to mimic .ipynb style
    notebook_json = json.dumps(notebook_dict, indent=1)
    # Write the JSON string to a file
    with open(filename, 'w') as f:
        f.write(notebook_json)

def create_code_mirror_mode(name: str, version: int) -> CodeMirrorMode:
    return CodeMirrorMode(name=name, version=version)

def create_language_info(codemirror_mode: CodeMirrorMode, file_extension: str, mimetype: str, name: str, nbconvert_exporter: str, pygments_lexer: str, version: str) -> LanguageInfo:
    return LanguageInfo(
        codemirror_mode=codemirror_mode,
        file_extension=file_extension,
        mimetype=mimetype,
        name=name,
        nbconvert_exporter=nbconvert_exporter,
        pygments_lexer=pygments_lexer,
        version=version
    )

def create_kernel_spec(display_name: str, language: str, name: str) -> KernelSpec:
    # Assuming you have some default values for kernelspec or you might want to extend this function
    # to take those as parameters if they can vary
    return KernelSpec(display_name=display_name, language=language, name=name)

def create_notebook_metadata(language_info: LanguageInfo, kernelspec: KernelSpec = None) -> NotebookMetadata:
    # Here, kernelspec is optional in case you do not need it for some notebooks
    return NotebookMetadata(kernelspec=kernelspec, language_info=language_info)

def create_notebook_cell(cell_type: str, execution_count: None, cell_id: str, metadata: Dict, outputs: List, source: List) -> NotebookCell:
    return NotebookCell(
        cell_type=cell_type,
        execution_count=execution_count,
        cell_id=cell_id,
        metadata=metadata,
        outputs=outputs,
        source=source
    )

def create_notebook(cells: List[NotebookCell], metadata: NotebookMetadata, nbformat: int, nbformat_minor: int) -> Notebook:
    return Notebook(
        cells=cells,
        metadata=metadata,
        nbformat=nbformat,
        nbformat_minor=nbformat_minor
    )

# Example of creating the notebook structure from the JSON-like structure
def recreate_notebook_structure():
    codemirror_mode = create_code_mirror_mode(name="ipython", version=3)
    language_info = create_language_info(
        codemirror_mode=codemirror_mode,
        file_extension=".py",
        mimetype="text/x-python",
        name="python",
        nbconvert_exporter="python",
        pygments_lexer="ipython3",
        version="3.8.18"
    )
    # KernelSpec is not provided in the JSON structure so we can assume defaults or omit it
    # kernelspec = create_kernel_spec(display_name="Python 3", language="python", name="python3")
    metadata = create_notebook_metadata(language_info=language_info)
    cell = create_notebook_cell(
        cell_type="code",
        execution_count=None,
        cell_id="3fb49960",
        metadata={},
        outputs=[],
        source=[]
    )
    notebook = create_notebook(cells=[cell], metadata=metadata, nbformat=4, nbformat_minor=5)
    return notebook

