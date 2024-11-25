import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from mitools.exceptions import ProjectError, ProjectFolderError, ProjectVersionError
from mitools.project import Project, VersionInfo, VersionState


class TestProject(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.project_name = "TestProject"
        self.version = "v0"
        self.project = Project(
            project_name=self.project_name, root=self.root_path, version=self.version
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_project_initialization(self):
        self.assertTrue(self.project.folder.exists(), "Project folder was not created.")
        self.assertTrue(
            self.project.version_folder.exists(), "Version folder was not created."
        )
        self.assertTrue(
            self.project.project_folder.exists(),
            "Internal project folder was not created.",
        )
        self.assertTrue(
            self.project.project_file.exists(),
            "Project file (project.pkl) was not created.",
        )
        self.assertTrue(
            self.project.project_notebook.exists(),
            "Default project notebook was not created.",
        )

    def test_invalid_root_directory(self):
        with self.assertRaises(ProjectFolderError):
            Project(
                project_name=self.project_name, root=self.root_path / "invalid_root"
            )

    def test_invalid_project_name(self):
        with self.assertRaises(ProjectError):
            Project(project_name="")  # Invalid empty project name

    def test_create_new_version(self):
        new_version = "v1"
        self.project.create_version(new_version)
        self.assertIn(new_version, self.project.get_all_versions())
        self.assertTrue(
            (self.project.folder / new_version).exists(),
            "Version folder was not created.",
        )

    def test_create_existing_version(self):
        with self.assertRaises(ProjectVersionError):
            self.project.create_version(self.version)

    def test_update_version(self):
        new_version = "v1"
        self.project.create_version(new_version)
        self.project.update_version(new_version)
        self.assertEqual(
            self.project.version, new_version, "Active version was not updated."
        )
        self.assertEqual(self.project.version, new_version)
        self.assertEqual(self.project.version_folder, self.project.folder / new_version)
        self.assertTrue(self.project.version_folder.exists())

    def test_delete_version(self):
        new_version = "v1"
        self.project.create_version(new_version)
        self.assertTrue(self.project.version_folder.exists())
        self.project.delete_version(new_version)
        self.assertNotIn(new_version, self.project.get_all_versions())
        self.assertFalse(
            (self.project.folder / new_version).exists(),
            "Version folder was not deleted.",
        )

    def test_delete_only_version(self):
        with self.assertRaises(ProjectError):
            self.project.delete_version(self.version)

    def test_create_version_subfolder(self):
        subfolder_name = "data"
        self.project.create_version_subfolder(subfolder_name)
        subfolder_path = self.project.version_folder / subfolder_name
        self.assertTrue(subfolder_path.exists(), "Subfolder was not created.")
        self.assertIn(subfolder_name, self.project.list_version_subfolders())

    def test_delete_version_subfolder(self):
        subfolder_name = "data"
        self.project.create_version_subfolder(subfolder_name)
        self.project.delete_subfolder(subfolder_name)
        subfolder_path = self.project.version_folder / subfolder_name
        self.assertFalse(subfolder_path.exists(), "Subfolder was not deleted.")
        self.assertNotIn(subfolder_name, self.project.list_version_subfolders())

    def test_create_project_notebook(self):
        self.assertTrue(
            self.project.project_notebook.exists(), "Project notebook was not created."
        )
        self.assertTrue(
            self.project.project_notebook.suffix == ".ipynb",
            "Project notebook is not a Jupyter Notebook.",
        )

    def test_restore_archived_version(self):
        new_version = "v1"
        self.project.create_version(new_version)
        self.project.archive_version(new_version)
        self.project.restore_version(new_version)
        self.assertTrue(
            (self.project.folder / new_version).exists(), "Version was not restored."
        )
        self.assertEqual(
            self.project.versions_metadata[new_version].state, VersionState.ACTIVE
        )

    def test_archive_version(self):
        new_version = "v1"
        self.project.create_version(new_version)
        self.project.archive_version(new_version)
        archive_path = self.project.folder / ".archive" / new_version
        print(archive_path)
        self.assertTrue(archive_path.exists(), "Version was not moved to archive.")
        self.assertEqual(
            self.project.versions_metadata[new_version].state, VersionState.ARCHIVED
        )

    def test_add_variable(self):
        self.project.add_var("key1", "value1")
        self.assertIn("key1", self.project.vars)
        self.assertEqual(self.project.vars["key1"], "value1")

    def test_add_existing_variable(self):
        self.project.add_var("key1", "value1")
        with self.assertRaises(ProjectError):
            self.project.add_var("key1", "value2")

    def test_update_variable(self):
        self.project.add_var("key1", "value1")
        self.project.update_var("key1", "new_value")
        self.assertEqual(self.project.vars["key1"], "new_value")

    def test_store_and_load_project(self):
        self.project.store_project()
        project_file = self.project.project_file
        self.assertTrue(project_file.exists(), "Project file was not created.")
        loaded_project = Project.load(project_folder=self.project.folder)
        self.assertEqual(loaded_project.name, self.project.name)
        self.assertEqual(loaded_project.version, self.project.version)
        self.assertEqual(loaded_project.versions, self.project.versions)

    def test_version_context(self):
        new_version = "v1"
        self.project.create_version(new_version)
        with self.project.version_context(new_version):
            self.assertEqual(self.project.version, new_version)
        self.assertEqual(
            self.project.version, "v0", "Version was not reverted after context."
        )

    def test_init(self):
        self.assertEqual(self.project.root, Path(self.temp_dir.name))
        self.assertEqual(self.project.name, self.project_name)
        self.assertEqual(
            self.project.folder, Path(self.temp_dir.name) / self.project_name
        )
        self.assertEqual(self.project.version, self.version)
        self.assertEqual(
            self.project.version_folder,
            Path(self.temp_dir.name) / self.project_name / self.version,
        )

    def test_create_main_folder(self):
        self.assertTrue((Path(self.temp_dir.name) / self.project_name).exists())

    def test_create_version_folder(self):
        self.assertTrue(
            (Path(self.temp_dir.name) / self.project_name / self.version).exists()
        )

    def test_get_all_versions(self):
        versions = self.project.get_all_versions()
        self.assertIn(self.version, versions)

    def test_create_subfolder(self):
        subfolder_name = "subfolder0"
        self.project.create_version_subfolder(subfolder_name)
        subfolder_path = self.project.version_folder / subfolder_name
        self.assertTrue(subfolder_path.exists())
        self.assertTrue(subfolder_path.is_dir())

    def test_list_subfolders(self):
        subfolder_name = "subfolder1"
        self.project.create_version_subfolder(subfolder_name)
        subfolders = self.project.list_version_subfolders()
        self.assertIn(subfolder_name, subfolders)

    def test_delete_subfolder(self):
        subfolder_name = "subfolder2"
        self.project.create_version_subfolder(subfolder_name)
        self.project.delete_subfolder(subfolder_name)
        self.assertNotIn(subfolder_name, self.project.list_version_subfolders())

    def test_reset_version(self):
        new_version = "v2"
        self.project.update_version(new_version)
        self.project.reset_version(new_version)
        self.assertIn(new_version, self.project.get_all_versions())

    def test_clear_version(self):
        dummy_file = self.project.version_folder / "dummy.txt"
        dummy_file.touch()
        self.project.clear_version()
        self.assertFalse(dummy_file.exists())

    def test_clear_project(self):
        dummy_file1 = self.project.folder / "dummy1.txt"
        dummy_file1.touch()
        dummy_file2 = self.project.version_folder / "dummy2.txt"
        dummy_file2.touch()
        self.project.clear_project()
        self.assertFalse(dummy_file1.exists())
        self.assertFalse(dummy_file2.exists())

    def test_delete_file(self):
        subfolder_name = "subfolder"
        self.project.create_version_subfolder(subfolder_name)
        file_name = "testfile.txt"
        file_path = self.project.version_folder / subfolder_name / file_name
        file_path.touch()  # Create the file
        self.project.delete_file(file_name, subfolder_name)
        self.assertFalse(file_path.exists())

    def test_repr_method(self):
        expected_repr = (
            f"Project({self.temp_dir.name}, {self.project_name}, {self.version})"
        )
        self.assertEqual(self.project.__repr__(), expected_repr)

    def test_str_method(self):
        expected_str = (
            f"Project {self.project.name}\n\nCurrent Version: {self.project.version},\nRoot: {self.project.root},\n"
            + f"Folder: {self.project.folder},\nVersions: {self.project.versions}\n"
        )
        self.assertEqual(self.project.__str__(), expected_str)

    def test_update_info(self):
        new_version = "v2"
        self.project.update_version(new_version)
        subfolder_name = "subfolder"
        self.project.create_version_subfolder(subfolder_name)
        self.project.update_info()
        self.assertIn(new_version, self.project.versions)
        self.assertIn(subfolder_name, self.project.subfolders)

    def test_get_info(self):
        info = self.project.get_info()
        self.assertEqual(info["name"], self.project_name)
        self.assertEqual(info["root"], str(self.temp_dir.name))
        self.assertEqual(info["folder"], str(self.project.folder))
        self.assertEqual(info["version"], self.version)
        self.assertIn(self.version, info["versions"])
        self.assertEqual(info["subfolders"], [])

    def test_version_tree(self):
        try:
            self.project.version_tree()
            execution_successful = True
        except Exception:
            execution_successful = False
        self.assertTrue(execution_successful)

    def test_project_tree(self):
        try:
            self.project.project_tree()
            execution_successful = True
        except Exception:
            execution_successful = False
        self.assertTrue(execution_successful)

    def test_temp_directory_tree(self):
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
            Project.load("invalid/path")

    def test_delete_nonexistent_version(self):
        with self.assertRaises(ValueError):
            self.project.delete_version("nonexistent_version")

    def test_delete_nonexistent_subfolder(self):
        with self.assertRaises(ValueError):
            self.project.delete_subfolder("nonexistent_subfolder")

    def test_delete_nonexistent_file(self):
        subfolder = "subfolder"
        self.project.create_version_subfolder(subfolder)
        with self.assertRaises(ValueError):
            self.project.delete_file("nonexistent_file.txt", subfolder)

    def test_delete_nonexistent_file_subfolder(self):
        with self.assertRaises(ValueError):
            self.project.delete_file("file.txt", "non_existent_subfolder")

    def test_create_existing_subfolder(self):
        subfolder_name = "subfolder"
        self.project.create_version_subfolder(subfolder_name)
        self.project.create_version_subfolder(subfolder_name)


if __name__ == "__main__":
    unittest.main()
