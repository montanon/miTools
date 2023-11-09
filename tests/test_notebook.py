import os
import unittest

from mitools.notebooks import *


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
            version="3.8.18"
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
            version="3.8.18"
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
        cell = create_notebook_cell(cell_type, execution_count, cell_id, cell_metadata, outputs, source)
        self.assertEqual(cell.cell_type, cell_type)
        self.assertIsNone(cell.execution_count)
        self.assertEqual(cell.cell_id, cell_id)

    def test_create_notebook(self):
        cells = [create_notebook_cell("code", None, "3fb49960", {}, [], [])]
        mode = create_code_mirror_mode("ipython", 3)
        language_info = create_language_info(
            codemirror_mode=mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.18"
        )
        metadata = create_notebook_metadata(language_info)
        nb = create_notebook(cells, metadata, 4, 5)
        self.assertEqual(nb.cells, cells)
        self.assertEqual(nb.metadata, metadata)
        self.assertEqual(nb.nbformat, 4)
        self.assertEqual(nb.nbformat_minor, 5)

class TestNotebookSaving(unittest.TestCase):

    def test_save_notebook_as_ipynb(self):
        # Prepare a notebook object
        notebook = recreate_notebook_structure()
        # Define a filename for the test
        test_filename = 'test_notebook.ipynb'
        # Save the notebook to a file
        save_notebook_as_ipynb(notebook, test_filename)
        # Read the file and check its contents
        with open(test_filename, 'r') as f:
            content = json.load(f)
        # Assert the content matches the notebook structure
        self.assertEqual(content['nbformat'], 4)
        self.assertEqual(content['nbformat_minor'], 5)
        # Clean up the test file
        os.remove(test_filename)

class TestNotebookSaving(unittest.TestCase):

    def test_save_notebook_with_multiple_cells_as_ipynb(self):
        # Prepare notebook cells
        cells = [
            create_notebook_cell("code", None, "3fb49960", {}, [], []),
            create_notebook_cell("markdown", None, "4ec49961", {}, [], ["# Header"]),
            # Add more cells as needed
        ]
        # Create the rest of the notebook structure
        notebook = create_notebook(cells, recreate_notebook_structure().metadata, 4, 5)
        # Define a filename for the test
        test_filename = 'test_notebook_multiple_cells.ipynb'
        # Save the notebook to a file
        save_notebook_as_ipynb(notebook, test_filename)
        # Read the file and check its contents
        with open(test_filename, 'r') as f:
            content = json.load(f)
        # Assert the content matches the notebook structure
        self.assertEqual(content['nbformat'], 4)
        self.assertEqual(content['nbformat_minor'], 5)
        self.assertEqual(len(content['cells']), len(cells))
        # Check if the cell contents are correct
        for cell_data, cell_obj in zip(content['cells'], cells):
            self.assertEqual(cell_data['cell_type'], cell_obj.cell_type)
            self.assertEqual(cell_data['cell_id'], cell_obj.cell_id)
        # Clean up the test file
        os.remove(test_filename)

class TestCellTypes(unittest.TestCase):

    def test_markdown_cell_type(self):
        markdown_cell = MarkdownCell(cell_type='markdown', metadata={}, source=["# Header"], cell_id="1234abcd")
        self.assertEqual(markdown_cell.cell_type, 'markdown')

    def test_code_cell_type(self):
        code_cell = CodeCell(cell_type='code', metadata={}, source=["print('Hello, world!')"], cell_id="1234abcd")
        self.assertEqual(code_cell.cell_type, 'code')

    def test_invalid_markdown_cell_type(self):
        with self.assertRaises(ValueError):
            MarkdownCell(cell_type='code', metadata={}, source=["# Header"], cell_id="1234abcd")

    def test_invalid_code_cell_type(self):
        with self.assertRaises(ValueError):
            CodeCell(cell_type='markdown', metadata={}, source=["print('Hello, world!')"], cell_id="1234abcd")


if __name__ == '__main__':
    unittest.main()
