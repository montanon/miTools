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


class TestFolderInSubtree(TestCase):
    def setUp(self):
        self.root = Path("/home/user/documents")
        self.branch = Path("/home/user/documents/reports")
        self.subfolder = Path("/home/user/documents/reports/2024")
        self.outside_folder = Path("/home/user/pictures")

    def test_folder_in_subtree_found(self):
        result = folder_in_subtree(self.root, self.subfolder, [self.branch])
        self.assertEqual(
            result, Path("/home/user/documents/reports").resolve(strict=False)
        )

    def test_no_folder_in_subtree(self):
        result = folder_in_subtree(self.root, self.subfolder, [self.outside_folder])
        self.assertIsNone(result)

    def test_branch_not_in_root_subtree(self):
        result = folder_in_subtree(
            self.root, self.outside_folder, [self.outside_folder]
        )
        self.assertIsNone(result)

    def test_same_folder_as_root(self):
        result = folder_in_subtree(self.root, self.root, [self.root])
        self.assertIsNone(result)  # Same folder should not be considered a subfolder

    def test_string_input(self):
        result = folder_in_subtree(
            str(self.root), str(self.subfolder), [str(self.branch)]
        )
        self.assertEqual(
            result, Path("/home/user/documents/reports").resolve(strict=False)
        )


if __name__ == "__main__":
    unittest.main()
