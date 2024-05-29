import pickle
import shutil
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional

from ..files import folder_in_subtree, folder_is_subfolder
from ..utils import build_dir_tree

PROJECT_FILENAME = "project.pkl"


class Project:
    def __init__(
        self, root: PathLike, project_name: str, version: Optional[str] = "v0"
    ):
        self.root = Path(root).absolute()
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"{self.root} is not a directory")
        elif not self.root.exists():
            raise ValueError(f"{self.root} does not exist")
        self.name = project_name
        self.folder = self.root / self.name
        self.create_main_folder()
        self.version = version
        self.version_folder = self.folder / self.version
        self.create_version_folder()
        self.vars = {}
        self.update_info()

    def create_main_folder(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

    def create_version_folder(self) -> None:
        self.version_folder.mkdir(parents=True, exist_ok=True)

    def update_version(self, version: str) -> None:
        self.version = version
        self.version_folder = self.folder / self.version
        self.create_version_folder()
        self.update_info()

    def get_all_versions(self) -> List[str]:
        return [d.name for d in self.folder.iterdir() if d.is_dir()]

    def create_subfolder(self, subfolder_name: str) -> None:
        subfolder_path = self.version_folder / subfolder_name
        subfolder_path.mkdir(parents=True, exist_ok=True)
        self.update_info()

    def list_subfolders(self) -> List[str]:
        return [d.name for d in self.version_folder.iterdir() if d.is_dir()]

    def delete_subfolder(self, subfolder_name: str) -> None:
        subfolder_path = self.version_folder / subfolder_name
        if not subfolder_path.exists():
            raise ValueError(
                f"Subfolder {subfolder_name} does not exist in Project {self.name} version {self.version}"
            )
        for child in subfolder_path.iterdir():
            if child.is_file():
                child.unlink()
        subfolder_path.rmdir()
        self.update_info()

    def reset_version(self, version: str) -> None:
        self.delete_version(version)
        self.update_version(version)

    def delete_version(self, version: str) -> None:
        if version in self.versions and len(self.versions) == 1:
            raise ValueError(
                f"Cannot delete, {version} is the only version of Project {self.name}"
            )
        elif version not in self.versions:
            raise ValueError(f"Version {version} does not exist in Project {self.name}")
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

    def clear_version(self) -> None:
        for path in self.version_folder.rglob("*"):
            if path.is_file():
                path.unlink()

    def clear_project(self) -> None:
        for path in self.folder.rglob("*"):
            if path.is_file() and path.name != PROJECT_FILENAME:
                path.unlink()

    def delete_file(self, file_name: str, subfolder: str = None) -> None:
        subfolder_path = self.version_folder / subfolder
        if not subfolder_path.exists():
            raise ValueError(
                f"Subfolder {subfolder} does not exist in Project {self.name} version {self.version}"
            )
        file_path = subfolder_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"File {file_name} does not exist in subfolder {subfolder} of Project {self.name}"
            )
        file_path.unlink()

    def store_project(self) -> None:
        self.update_info()
        with open(self.folder / PROJECT_FILENAME, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_project(
        cls, project_folder: Optional[Path] = None, auto_load: bool = False, n: int = 3
    ) -> "Project":
        if project_folder is None and auto_load:
            current_path = Path.cwd().resolve()
            for _ in range(n):
                project_path = current_path / PROJECT_FILENAME
                if project_path.exists():
                    break
                if current_path.parent == current_path:  # reached the root directory
                    break
                current_path = current_path.parent
            else:
                raise FileNotFoundError(
                    f"No {PROJECT_FILENAME} found in the current or {n} parent directories."
                )
        else:
            project_path = Path(project_folder) / PROJECT_FILENAME
            if not project_path.exists():
                raise FileNotFoundError(
                    f"{PROJECT_FILENAME} does not exist in the specified directory {project_folder}"
                )

        with project_path.open("rb") as file:
            obj = pickle.load(file)
        obj.update_info()
        obj.vars = obj.vars if hasattr(obj, "vars") else {}

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

    def update_info(self) -> None:
        self.versions = self.get_all_versions()
        self.version_folders = [
            Path(self.folder) / version for version in self.versions
        ]
        self.subfolders = self.list_subfolders()
        self.vars.update(self.folder_path_dict())

    def folder_path_dict(self) -> List[str]:
        return {
            subfolder: self.version_folder / subfolder for subfolder in self.subfolders
        }

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
        tree = build_dir_tree(directory)
        tree.show()

    def clone_version(self, source_version: str, new_version: str) -> None:
        source_version_folder = self.folder / source_version
        new_version_folder = self.folder / new_version

        if not source_version_folder.exists():
            raise ValueError(f"Source version {source_version} does not exist.")

        if new_version_folder.exists():
            raise ValueError(f"New version {new_version} already exists.")

        shutil.copytree(source_version_folder, new_version_folder)
        self.update_version(new_version)
        self.update_info()
        self.store_project()
