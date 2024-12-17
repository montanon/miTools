import json
import unittest
from pathlib import Path

import nbformat

from mitools.context import DEBUG
from mitools.notebooks import (
    CodeCell,
    MarkdownCell,
    NotebookCell,
    NotebookCellFactory,
    NotebookCells,
    NotebookSection,
    clear_notebook_output,
    create_code_mirror_mode,
    create_kernel_spec,
    create_language_info,
    create_notebook,
    create_notebook_cell,
    create_notebook_metadata,
    recreate_notebook_structure,
    save_notebook_as_ipynb,
)


class TestNotebookCreation(unittest.TestCase):
    def test_create_code_mirror_mode(self):
        name = "ipython"
        version = 3
        mode = create_code_mirror_mode(name, version)
        self.assertEqual(mode.name, name)
        self.assertEqual(mode.version, version)

    def test_create_language_info(self):
        mode = create_code_mirror_mode("ipython", 3)
        info = create_language_info(
            codemirror_mode=mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.18",
        )
        self.assertEqual(info.codemirror_mode, mode)
        self.assertEqual(info.file_extension, ".py")

    def test_create_kernel_spec(self):
        display_name = "Python 3 (ipykernel)"
        language = "python"
        name = "python3"
        spec = create_kernel_spec(display_name, language, name)
        self.assertEqual(spec.display_name, display_name)
        self.assertEqual(spec.language, language)
        self.assertEqual(spec.name, name)

    def test_create_notebook_metadata(self):
        mode = create_code_mirror_mode("ipython", 3)
        language_info = create_language_info(
            codemirror_mode=mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.18",
        )
        metadata = create_notebook_metadata(language_info)
        self.assertEqual(metadata.language_info, language_info)
        # Test that kernelspec is None by default if not provided
        self.assertIsNone(metadata.kernelspec)

    def test_create_notebook_cell(self):
        cell_type = "code"
        execution_count = None
        cell_id = "3fb49960"
        cell_metadata = {}
        outputs = []
        source = []
        cell = create_notebook_cell(
            cell_type, execution_count, cell_id, cell_metadata, outputs, source
        )
        self.assertEqual(cell.cell_type, cell_type)
        self.assertIsNone(cell.execution_count)
        self.assertEqual(cell.cell_id, cell_id)

    def test_create_markdown_notebook_cell(self):
        cell_type = "markdown"
        execution_count = None
        cell_id = "3fb49960"
        cell_metadata = {}
        outputs = []
        source = []
        cell = create_notebook_cell(
            cell_type, execution_count, cell_id, cell_metadata, outputs, source
        )
        self.assertEqual(cell.cell_type, cell_type)
        self.assertIsNone(cell.execution_count)
        self.assertEqual(cell.cell_id, cell_id)

    def test_create_notebook(self):
        cells = [create_notebook_cell("code", None, "5feceb66ffc86f38", {}, [], [])]
        mode = create_code_mirror_mode("ipython", 3)
        language_info = create_language_info(
            codemirror_mode=mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.18",
        )
        metadata = create_notebook_metadata(language_info)
        nb = create_notebook(cells, metadata, 4, 5)
        self.assertEqual(nb.cells, cells)
        self.assertEqual(nb.metadata, metadata)
        self.assertEqual(nb.nbformat, 4)
        self.assertEqual(nb.nbformat_minor, 5)

    def test_notebook_section_with_markdown_first(self):
        # Create a MarkdownCell and a NotebookCell
        markdown_cell = create_notebook_cell(
            cell_type="markdown",
            execution_count=None,
            metadata={},
            outputs={},
            source=["# Markdown Content"],
            cell_id="1",
        )
        code_cell = create_notebook_cell(
            cell_type="code",
            execution_count=None,
            metadata={},
            outputs=[],
            source=["print('Hello World')"],
            cell_id="2",
        )
        # Create a list of cells with the MarkdownCell first
        cells = NotebookCells(cells=[markdown_cell, code_cell])
        # Create a NotebookSection
        section = NotebookSection(cells=cells)
        # Check that the first cell is a MarkdownCell
        self.assertIsInstance(section.cells[0], MarkdownCell)

    def test_notebook_section_with_no_markdown_first(self):
        # Create a NotebookCell
        code_cell = create_notebook_cell(
            cell_type="code",
            execution_count=None,
            metadata={},
            outputs=[],
            source=["print('Hello World')"],
            cell_id="1",
        )
        # Create a list of cells with the NotebookCell first (no MarkdownCell)
        cells = [code_cell]
        # Attempt to create a NotebookSection and expect an error
        with self.assertRaises(ValueError):
            NotebookSection(cells=cells)


class TestNotebookSaving(unittest.TestCase):
    def test_save_notebook_as_ipynb(self):
        # Prepare a notebook object
        notebook = recreate_notebook_structure()
        # Define a filename for the test
        test_filename = Path("test_notebook.ipynb")
        # Save the notebook to a file
        save_notebook_as_ipynb(notebook, test_filename)
        # Read the file and check its contents
        with open(test_filename, "r") as f:
            content = json.load(f)
        # Assert the content matches the notebook structure
        self.assertEqual(content["nbformat"], 4)
        self.assertEqual(content["nbformat_minor"], 5)
        # Clean up the test file
        if not DEBUG:
            test_filename.unlink()

    def test_save_notebook_with_multiple_cells_as_ipynb(self):
        # Prepare notebook cells
        cells = [
            create_notebook_cell("markdown", None, "", {}, [], []),
            create_notebook_cell("markdown", None, "", {}, [], ["# Header"]),
            # Add more cells as needed
        ]
        # Create the rest of the notebook structure
        notebook = create_notebook(cells, recreate_notebook_structure().metadata, 4, 5)
        # Define a filename for the test
        test_filename = Path("test_notebook_multiple_cells.ipynb")
        # Save the notebook to a file
        save_notebook_as_ipynb(notebook, test_filename)
        # Read the file and check its contents
        with open(test_filename, "r") as f:
            content = json.load(f)
        # Assert the content matches the notebook structure
        self.assertEqual(content["nbformat"], 4)
        self.assertEqual(content["nbformat_minor"], 5)
        self.assertEqual(len(content["cells"]), len(cells))
        # Check if the cell contents are correct
        for cell_data, cell_obj in zip(content["cells"], cells):
            self.assertEqual(cell_data["cell_type"], cell_obj.cell_type)
            self.assertNotEqual(cell_data["cell_id"], cell_obj.cell_id)
        # Clean up the test file
        if not DEBUG:
            test_filename.unlink()


class TestCellTypes(unittest.TestCase):
    def test_markdown_cell_type(self):
        markdown_cell = MarkdownCell(
            cell_type="markdown", metadata={}, source=["# Header"], cell_id="1234abcd"
        )
        self.assertEqual(markdown_cell.cell_type, "markdown")

    def test_code_cell_type(self):
        code_cell = CodeCell(
            cell_type="code",
            metadata={},
            source=["print('Hello, world!')"],
            cell_id="1234abcd",
        )
        self.assertEqual(code_cell.cell_type, "code")

    def test_invalid_markdown_cell_type(self):
        with self.assertRaises(ValueError):
            MarkdownCell(
                cell_type="code", metadata={}, source=["# Header"], cell_id="1234abcd"
            )

    def test_invalid_code_cell_type(self):
        with self.assertRaises(ValueError):
            CodeCell(
                cell_type="markdown",
                metadata={},
                source=["print('Hello, world!')"],
                cell_id="1234abcd",
            )


class TestClearNotebookOutput(unittest.TestCase):
    def setUp(self):
        cells = NotebookCells(
            [
                create_notebook_cell("markdown", None, "", {}, [], ["## Tile"]),
                create_notebook_cell("code", None, "", {}, [], ["## Code"]),
                create_notebook_cell("markdown", None, "", {}, [], ["## Tile"]),
            ]
        )
        notebook = create_notebook(cells, recreate_notebook_structure().metadata, 4, 5)
        # Define a filename for the test
        self.test_filename = Path("test_notebook.ipynb")
        # Save the notebook to a file
        save_notebook_as_ipynb(notebook, self.test_filename)

    def test_clear_notebook_output(self):
        # Define the paths for the test notebook and the cleaned notebook
        clean_notebook_path = Path("clean_test_notebook.ipynb")
        # Assume test_notebook_path has a notebook with outputs
        # Now call the function to clear the outputs
        clear_notebook_output(self.test_filename, clean_notebook_path)
        # Load the cleaned notebook to check if outputs have been cleared
        with open(clean_notebook_path, "r", encoding="utf-8") as f:
            clean_nb = nbformat.read(f, as_version=4)
        # Check all cells to ensure outputs are cleared
        for cell in clean_nb.cells:
            if "outputs" in cell:
                self.assertEqual(cell["outputs"], [])
        # Clean up the test file
        if not DEBUG:
            self.test_filename.unlink()
            clean_notebook_path.unlink()


class TestNotebookCellFactory(unittest.TestCase):
    def test_create_code_cell(self):
        code_cell = NotebookCellFactory.create_cell(
            "code", source=["print('Hello, World!')"]
        )
        self.assertIsInstance(code_cell, CodeCell)
        self.assertEqual(code_cell.source, ["print('Hello, World!')"])

    def test_create_markdown_cell(self):
        markdown_cell = NotebookCellFactory.create_cell("markdown", source=["# Header"])
        self.assertIsInstance(markdown_cell, MarkdownCell)
        self.assertEqual(markdown_cell.source, ["# Header"])

    def test_create_default_cell(self):
        # Assuming 'text' is not a recognized cell type, it should default to NotebookCell
        default_cell = NotebookCellFactory.create_cell("text", source=["Text content"])
        self.assertIsInstance(default_cell, NotebookCell)
        self.assertEqual(default_cell.source, ["Text content"])


if __name__ == "__main__":
    unittest.main()
