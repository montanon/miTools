import shutil
import tempfile
import unittest
from pathlib import Path

from mitools.project import Project


class TestProject(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_name = 'test_project'
        self.version = 'v0'
        self.project = Project(self.test_dir, self.project_name, self.version)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        self.assertEqual(self.project.root, Path(self.test_dir))
        self.assertEqual(self.project.name, self.project_name)
        self.assertEqual(self.project.folder, Path(self.test_dir) / self.project_name)
        self.assertEqual(self.project.version, self.version)
        self.assertEqual(self.project.version_folder, Path(self.test_dir) / self.project_name / self.version)

    def test_create_main_folder(self):
        self.assertTrue((Path(self.test_dir) / self.project_name).exists())

    def test_create_version_folder(self):
        self.assertTrue((Path(self.test_dir) / self.project_name / self.version).exists())

    def test_update_version(self):
        new_version = 'v1'
        self.project.update_version(new_version)
        self.assertEqual(self.project.version, new_version)
        self.assertEqual(self.project.version_folder, self.project.folder / new_version)
        self.assertTrue(self.project.version_folder.exists())

    def test_get_all_versions(self):
        versions = self.project.get_all_versions()
        self.assertIn(self.version, versions)

    def test_create_subfolder(self):
        subfolder_name = 'subfolder0'
        self.project.create_subfolder(subfolder_name)
        subfolder_path = self.project.version_folder / subfolder_name
        self.assertTrue(subfolder_path.exists())
        self.assertTrue(subfolder_path.is_dir())

    def test_list_subfolders(self):
        subfolder_name = 'subfolder1'
        self.project.create_subfolder(subfolder_name)
        subfolders = self.project.list_subfolders()
        self.assertIn(subfolder_name, subfolders)

    def test_delete_subfolder(self):
        subfolder_name = "subfolder2"
        self.project.create_subfolder(subfolder_name)
        self.project.delete_subfolder(subfolder_name)
        self.assertNotIn(subfolder_name, self.project.list_subfolders())

    def test_reset_version(self):
        # Reset version and check if the new version folder is created
        new_version = "v2"
        self.project.update_version(new_version)
        self.project.reset_version(new_version)
        self.assertIn(new_version, self.project.get_all_versions())

    def test_delete_version(self):
        # Add a new version and then delete it
        new_version = "v2"
        self.project.update_version(new_version)
        self.assertTrue(self.project.version_folder.exists())
        self.project.delete_version(new_version)
        self.assertNotIn(new_version, self.project.get_all_versions())
    
    def test_clear_version(self):
        # Add a dummy file to clear
        dummy_file = self.project.version_folder / "dummy.txt"
        dummy_file.touch()
        self.project.clear_version()
        self.assertFalse(dummy_file.exists())

    def test_clear_project(self):
        # Add a dummy file to clear
        dummy_file1 = self.project.folder / "dummy1.txt"
        dummy_file1.touch()
        dummy_file2 = self.project.version_folder / "dummy2.txt"
        dummy_file2.touch()
        self.project.clear_project()
        self.assertFalse(dummy_file1.exists())
        self.assertFalse(dummy_file2.exists()) 

    def test_delete_file(self):
        # Create a file and then delete it
        subfolder_name = "subfolder"
        self.project.create_subfolder(subfolder_name)
        file_name = "testfile.txt"
        file_path = self.project.version_folder / subfolder_name / file_name
        file_path.touch()  # Create the file
        self.project.delete_file(file_name, subfolder_name)
        self.assertFalse(file_path.exists())

    def test_store_and_load_project(self):
        self.project.store_project()
        loaded_project = Project.load_project(self.project.folder)
        self.assertEqual(loaded_project.name, self.project.name)
        self.assertEqual(loaded_project.version, self.project.version)
        self.assertEqual(loaded_project.versions, self.project.versions)

    def test_repr_method(self):
        # Check the __repr__ method for correct format
        expected_repr = f"Project({self.test_dir}, {self.project_name}, {self.version})"
        self.assertEqual(self.project.__repr__(), expected_repr)

    def test_str_method(self):
        # Check the __str__ method for correct format
        expected_str = (f"Project {self.project.name}\n\nCurrent Version: {self.project.version},\nRoot: {self.project.root},\n" +
            f"Folder: {self.project.folder},\nVersions: {self.project.versions}\n")
        self.assertEqual(self.project.__str__(), expected_str)

    def test_update_info(self):
        # Check if versions and subfolders are correctly updated
        new_version = "v2"
        self.project.update_version(new_version)
        subfolder_name = "subfolder"
        self.project.create_subfolder(subfolder_name)
        self.project.update_info()
        self.assertIn(new_version, self.project.versions)
        self.assertIn(subfolder_name, self.project.subfolders)

    def test_get_info(self):
        # Check if get_info method returns correct info
        info = self.project.get_info()
        self.assertEqual(info["name"], self.project_name)
        self.assertEqual(info["root"], str(self.test_dir))
        self.assertEqual(info["folder"], str(self.project.folder))
        self.assertEqual(info["version"], self.version)
        self.assertIn(self.version, info["versions"])
        # Initially, no subfolders are created
        self.assertEqual(info["subfolders"], [])

    def test_version_tree(self):
        # Test if directory_tree method executes without error (not checking output)
        try:
            self.project.version_tree()
            execution_successful = True
        except Exception:
            execution_successful = False
        self.assertTrue(execution_successful)

    def test_project_tree(self):
        # Test if directory_tree method executes without error (not checking output)
        try:
            self.project.project_tree()
            execution_successful = True
        except Exception:
            execution_successful = False
        self.assertTrue(execution_successful)

    def test_directory_tree(self):
        # Test if directory_tree method executes without error (not checking output)
        try:
            self.project.directory_tree(self.project.version_folder)
            execution_successful = True
        except Exception:
            execution_successful = False
        self.assertTrue(execution_successful)

    def test_invalid_folder_path(self):
        with self.assertRaises(ValueError):
            Project("invalid/path", self.project_name)

    def test_store_project_with_invalid_path(self):
        self.project.folder = Path("invalid/path")
        with self.assertRaises(Exception):
            self.project.store_project()

    def test_load_project_with_invalid_folder(self):
        with self.assertRaises(Exception):
            Project.load_project("invalid/path")

    def test_delete_nonexistent_version(self):
        with self.assertRaises(ValueError):
            self.project.delete_version("nonexistent_version")

    def test_delete_nonexistent_subfolder(self):
        with self.assertRaises(ValueError):
            self.project.delete_subfolder("nonexistent_subfolder")

    def test_delete_nonexistent_file(self):
        # Trying to delete a file that doesn't exist
        subfolder = 'subfolder'
        self.project.create_subfolder(subfolder)
        with self.assertRaises(FileNotFoundError):
            self.project.delete_file("nonexistent_file.txt", subfolder)

    def test_delete_nonexistent_file_subfolder(self):
        # Trying to delete a file that doesn't exist
        with self.assertRaises(ValueError):
            self.project.delete_file("file.txt", "non_existent_subfolder")

    def test_create_existing_subfolder(self):
        subfolder_name = "subfolder"
        self.project.create_subfolder(subfolder_name)
        self.project.create_subfolder(subfolder_name)

    
if __name__ == '__main__':
    unittest.main()