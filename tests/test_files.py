import shutil
import unittest
from pathlib import Path
from unittest import TestCase

from mitools.files import (
    folder_in_subtree,
    folder_is_subfolder,
    rename_folders_in_folder,
)


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


class TestRenameFoldersInFolder(TestCase):
    def setUp(self):
        self.test_dir = Path("./test_folder")
        self.test_dir.mkdir(exist_ok=True)
        (self.test_dir / "folder 1").mkdir()
        (self.test_dir / "folder 2").mkdir()
        (self.test_dir / "already_exists").mkdir()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_default_rename(self):
        rename_folders_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "folder_1").exists())
        self.assertTrue((self.test_dir / "folder_2").exists())

    def test_attempt_mode(self):
        rename_folders_in_folder(self.test_dir, attempt=True)
        self.assertTrue((self.test_dir / "folder 1").exists())
        self.assertTrue((self.test_dir / "folder 2").exists())

    def test_custom_char_replacement(self):
        rename_folders_in_folder(
            self.test_dir, char_replacement=lambda name: name.replace(" ", "-")
        )
        self.assertTrue((self.test_dir / "folder-1").exists())
        self.assertTrue((self.test_dir / "folder-2").exists())

    def test_existing_target_folder(self):
        (self.test_dir / "folder_1").mkdir()  # Create a conflicting folder
        rename_folders_in_folder(self.test_dir)
        self.assertTrue((self.test_dir / "folder 1").exists())  # Original remains

    def test_no_change_for_identical_name(self):
        (self.test_dir / "folder_1").mkdir()
        rename_folders_in_folder(
            self.test_dir,
            char_replacement=lambda name: name,  # No change
        )
        self.assertTrue((self.test_dir / "folder_1").exists())

    def test_invalid_directory(self):
        with self.assertRaises(ValueError):
            rename_folders_in_folder("./non_existent_folder")


if __name__ == "__main__":
    unittest.main()
