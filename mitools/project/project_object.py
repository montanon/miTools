import pickle
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional

from mitools.exceptions import ProjectError, ProjectFolderError, ProjectVersionError
from mitools.files import folder_in_subtree, folder_is_subfolder
from mitools.notebooks import recreate_notebook_structure, save_notebook_as_ipynb
from mitools.utils import build_dir_tree

PROJECT_FILENAME = Path("project.pkl")
PROJECT_FOLDER = Path(".project")
PROJECT_NOTEBOOK = "Project.ipynb"
PROJECT_ARCHIVE = ".archive"
PROJECT_BACKUP = ".backup"
NOT_IN_NOTEBOOK = (
    "Project object can only be initialized inside the respective Project Notebook={}."
)


class VersionState(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class VersionInfo:
    version: str
    creation: float
    state: VersionState = VersionState.ACTIVE
    description: str = ""


class Project:
    def __init__(
        self,
        project_name: str,
        root: PathLike = ".",
        version: str = "v0",
        logger: Logger = None,
    ):
        if not isinstance(project_name, str) or not project_name:
            raise ProjectError(f"Project 'root'={root} must be a non-empty string.")
        self.root = Path(root).absolute()
        if self.root.exists() and not self.root.is_dir():
            raise ProjectFolderError(f"{self.root} is not a directory")
        elif not self.root.exists():
            raise ProjectFolderError(f"{self.root} does not exist")
        self.logger = logger
        self.name = project_name
        self.folder = self.root / self.name
        self.project_folder = self.folder / PROJECT_FOLDER
        self.project_file = self.folder / PROJECT_FILENAME
        self.project_notebook = self.folder / PROJECT_NOTEBOOK
        self.backup_folder = self.folder / PROJECT_BACKUP
        self.create_main_folder()
        self.version = version
        self.version_folder = self.folder / self.version
        self.create_version_folder()
        self.versions = self.get_all_versions()
        self.versions_metadata: Dict[str, VersionInfo] = {}
        self.vars: Dict[str, Any] = {}
        self.paths: Dict[str, Path] = {}
        self.tree = build_dir_tree(self.folder)
        self.update_info()
        self.store_project()

    def create_main_folder(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.create_project_notebook()  # TODO

    def create_version_folder(self) -> None:
        self.version_folder.mkdir(parents=True, exist_ok=True)

    def get_all_versions(self) -> List[str]:
        return [
            d.name
            for d in self.folder.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def folder_path_dict(self) -> List[str]:
        return {
            subfolder: self.version_folder / subfolder for subfolder in self.subfolders
        }

    def update_info(self) -> None:
        self.versions = self.get_all_versions()
        self.version_folders = [
            Path(self.folder) / version for version in self.versions
        ]
        self.subfolders = self.list_version_subfolders()
        self.paths.update(self.folder_path_dict())

    def create_version(self, version: str, description: str = "") -> None:
        version_path = self.folder / version
        if not version_path.exists():
            version_path.mkdir(parents=True, exist_ok=True)
            self.update_info()
            self.add_version_metadata(version, description)
        else:
            raise ProjectVersionError(
                f"Version {version} already exists in Project {self.name}. Existing versions: [{self.versions}]"
            )

    def update_version(self, version: str) -> None:
        self.version = version
        self.version_folder = self.folder / self.version
        self.create_version_folder()
        self.update_info()

    def add_version_metadata(self, version: str, description: str = "") -> None:
        if version not in self.versions:
            raise ProjectError(
                f"Version {version} does not exist in Project {self.name} with version {self.versions}"
            )
        self.versions_metadata[version] = VersionInfo(
            version=version,
            creation=datetime.now(),
            description=description,
        )
        self.store_project()

    def create_version_subfolder(self, subfolder_name: str) -> None:
        subfolder_path = self.version_folder / subfolder_name
        subfolder_path.mkdir(parents=True, exist_ok=True)
        self.update_info()

    def list_version_subfolders(self) -> List[str]:
        return [d.name for d in self.version_folder.iterdir() if d.is_dir()]

    @contextmanager
    def version_context(self, version: str):
        original_version = self.version
        try:
            self.update_version(version)
            yield
        finally:
            self.update_version(original_version)

    def delete_subfolder(self, subfolder_name: str) -> None:
        subfolder_path = self.version_folder / subfolder_name
        if not subfolder_path.exists():
            raise ProjectFolderError(
                f"Subfolder {subfolder_name} does not exist in Project {self.name} version {self.version}"
            )
        for child in subfolder_path.iterdir():
            if child.is_file():
                child.unlink()
        subfolder_path.rmdir()
        self.update_info()

    def delete_version(self, version: str) -> None:
        if version in self.versions and len(self.versions) == 1:
            raise ProjectError(
                f"Cannot delete, {version} is the only version of Project {self.name}"
            )
        elif version not in self.versions:
            raise ProjectVersionError(
                f"Version {version} does not exist in Project {self.name}, with versions: {self.versions}"
            )
        version_path: Path = self.folder / version
        print(f"About to remove version {version} of Project {self.name}...")
        if version_path.exists() and version_path.is_dir():
            for item in version_path.glob("**/*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            version_path.rmdir()
        print(f"Removed version {version} of Project {self.name}")
        if self.version == version:
            print(
                f"Changing current version of Project {self.name} to {self.versions[0]}"
            )
            self.update_version(self.versions[0])

    def archive_version(self, version: str) -> None:
        if version not in self.versions_metadata:
            raise ProjectError(
                f"Version {version} not found int Project {self.name} with versions {self.versions}"
            )
        self.versions_metadata[version].state = VersionState.ARCHIVED
        archive_path = self.folder / PROJECT_ARCHIVE / version
        version_path = self.folder / version
        archive_path.parent.mkdir(exist_ok=True)
        shutil.move(str(version_path), str(archive_path))
        self.store_project()

    def restore_version(self, version: str) -> None:
        archive_path = self.folder / PROJECT_ARCHIVE / version
        if not archive_path.exists():
            raise ProjectError(
                f"Archived version {version} not found in Archive: {self.folder / PROJECT_ARCHIVE}"
            )
        version_path = self.folder / version
        shutil.move(str(archive_path), str(version_path))
        self.versions_metadata[version].state = VersionState.ACTIVE
        self.store_project()

    def reset_version(self, version: str) -> None:
        self.delete_version(version)
        self.update_version(version)

    def get_version_data(self, version: str) -> Dict[str, Any]:
        if version not in self.versions_metadata:
            raise ProjectVersionError(
                f"Version {version} not found in Porject {self.name} with versions {self.versions}"
            )
        metadata = self.versions_metadata[version]
        return {
            "name": version,
            "created_at": metadata.created_at,
            "state": metadata.state.value,
            "description": metadata.description,
            "subfolders": self.list_version_subfolders(version),
        }

    def clear_version(self) -> None:
        for path in self.version_folder.rglob("*"):
            if path.is_file():
                path.unlink()

    def clear_project(self) -> None:
        for path in self.folder.rglob("*"):
            if (
                path.is_file()
                and path.name != PROJECT_FILENAME
                and path.name != PROJECT_NOTEBOOK
                and path.parent != PROJECT_FOLDER
                and path.parent != PROJECT_ARCHIVE
                and path.parent != PROJECT_BACKUP
            ):
                path.unlink()

    def delete_file(self, file_name: str, subfolder: str = None) -> None:
        subfolder_path = self.version_folder / subfolder
        if not subfolder_path.exists():
            raise ProjectFolderError(
                f"Subfolder {subfolder} does not exist in Project {self.name} version {self.version}"
            )
        file_path = subfolder_path / file_name
        if not file_path.exists():
            raise ProjectError(
                f"File {file_name} does not exist in subfolder {subfolder} of Project {self.name}"
            )
        file_path.unlink()

    def store_project(self) -> None:
        self.update_info()
        with open(self.project_file, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def find_project(
        cls,
        project_folder: PathLike = None,
        max_depth: int = 3,
        auto_load: bool = False,
    ):
        if project_folder is None and auto_load:
            current_path = Path.cwd().resolve()
            for _ in range(max_depth):
                project_path = current_path / PROJECT_FILENAME
                if project_path.exists():
                    break
                if current_path.parent == current_path:  # reached the root directory
                    break
                current_path = current_path.parent
            else:
                raise ProjectError(
                    f"No {PROJECT_FILENAME} found in the current or {max_depth} parent directories."
                )
        else:
            project_path = Path(project_folder) / PROJECT_FILENAME
            if not project_path.exists():
                raise ProjectError(
                    f"{PROJECT_FILENAME} does not exist in the specified directory {project_folder}"
                )
        return project_path

    @classmethod
    def load(
        cls,
        project_folder: Optional[Path] = None,
        auto_load: bool = False,
        max_depth: int = 3,
    ) -> "Project":
        project_path = cls.find_project(
            project_folder=project_folder, max_depth=max_depth, auto_load=auto_load
        )
        with project_path.open("rb") as file:
            obj = pickle.load(file)
        obj.update_info()
        # Retro-compatibility
        obj.vars = obj.vars if hasattr(obj, "vars") else {}
        obj.paths = obj.paths if hasattr(obj, "paths") else {}
        obj.versions_metadata = (
            obj.versions_metadata
            if hasattr(obj, "versions_metadata")
            else {v: obj.add_version_metadata(v) for v in obj.versions}
        )
        current_path = Path.cwd().resolve()
        if folder_is_subfolder(obj.root, current_path):
            version_folder = folder_in_subtree(
                obj.root, current_path, obj.version_folders
            )
            if version_folder:
                obj.update_version(version_folder.stem)
                print(f"Updated Project version to current {obj.version} version.")
        return obj

    def __repr__(self) -> str:
        return f"Project({self.root}, {self.name}, {self.version})"

    def __str__(self) -> str:
        return (
            f"Project {self.name}\n\nCurrent Version: {self.version},\nRoot: {self.root},\n"
            + f"Folder: {self.folder},\nVersions: {self.versions}\n"
        )

    def get_info(self) -> Dict:
        self.update_info()
        return {
            "name": self.name,
            "root": str(self.root),
            "folder": str(self.folder),
            "version": self.version,
            "versions": self.versions,
            "subfolders": self.subfolders,
        }

    def version_tree(self) -> None:
        self.directory_tree(self.version_folder)

    def project_tree(self) -> None:
        self.directory_tree(self.folder)

    def directory_tree(self, directory: PathLike) -> None:
        self.tree = build_dir_tree(directory)
        self.tree.show()

    def clone_version(self, source_version: str, new_version: str) -> None:
        source_version_folder = self.folder / source_version
        new_version_folder = self.folder / new_version

        if not source_version_folder.exists():
            raise ProjectVersionError(
                f"Version {source_version} does not exists in Project {self.name}. Existing versions: [{self.versions}]"
            )

        if new_version_folder.exists():
            raise ProjectVersionError(
                f"Version {new_version} already exists in Project {self.name}. Existing versions: [{self.versions}]"
            )

        shutil.copytree(source_version_folder, new_version_folder)
        self.update_version(new_version)
        self.update_info()
        self.store_project()

    def add_var(self, key: str, value: Any, overwrite: bool = False) -> None:
        if key in self.vars and not overwrite:
            raise ProjectError(
                f"Key '{key}' already exists in self.vars. Use update_var() to modify existing variables."
            )
        self.vars[key] = value
        self.store_project()
        print(f"Added '{key}' to project variables and stored the project.")

    def update_var(self, key: str, value: Any) -> None:
        if key not in self.vars:
            raise ProjectError(
                f"Key {key} does not exist in self.vars. Cannot update non-existing variable."
            )
        self.vars[key] = value
        self.store_project()
        print(f"Updated '{key}' of project variables and stored the project.")

    def add_path(self, key: str, path: Path, overwrite: bool = False) -> None:
        if not isinstance(path, Path):
            raise ProjectError(
                f"Provided 'path'={path} of type {type(path)} must be of type pathlib.Path"
            )
        if key in self.paths and not overwrite:
            raise ProjectError(
                f"Key '{key}' already exists in self.paths. Use update_path() to modify existing variables."
            )
        self.paths[key] = path
        self.store_project()
        print(f"Added '{key}' to project paths and stored the project.")

    def update_path(self, key: str, path: Path) -> None:
        if not isinstance(path, Path):
            raise ProjectError(
                f"Provided 'path'={path} of type {type(path)} must be of type pathlib.Path"
            )
        if key not in self.paths:
            raise ProjectError(
                f"Key {key} does not exist in self.paths. Cannot update non-existing path."
            )
        self.paths[key] = path
        self.store_project()
        print(f"Updated '{key}' of project paths and stored the project.")

    def create_project_notebook(self) -> None:
        notebook = recreate_notebook_structure()
        save_notebook_as_ipynb(notebook, self.project_notebook)
