import unittest
from pathlib import Path
from unittest import TestCase

from mitools.files import folder_in_subtree, folder_is_subfolder


class TestFolderIsSubfolder(TestCase):
    def setUp(self):
        self.root_folder = Path("/home/user/documents")
        self.sub_folder = Path("/home/user/documents/reports")
        self.non_sub_folder = Path("/home/user/pictures")

    def test_valid_subfolder(self):
        result = folder_is_subfolder(self.root_folder, self.sub_folder)
        self.assertTrue(result)

    def test_non_subfolder(self):
        result = folder_is_subfolder(self.root_folder, self.non_sub_folder)
        self.assertFalse(result)

    def test_same_folder(self):
        result = folder_is_subfolder(self.root_folder, self.root_folder)
        self.assertFalse(result)

    def test_non_existent_paths(self):
        result = folder_is_subfolder(
            Path("/non_existent_folder"), Path("/non_existent_folder/subfolder")
        )
        self.assertTrue(result)

    def test_string_input(self):
        result = folder_is_subfolder("/home/user", "/home/user/documents")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
