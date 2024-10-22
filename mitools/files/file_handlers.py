import shutil
from os import PathLike
from pathlib import Path
from typing import Callable, List, Union

from mitools.exceptions import ArgumentValueError


def folder_is_subfolder(root_folder: PathLike, folder_to_check: PathLike) -> bool:
    root_folder = Path(root_folder)
    folder_to_check = Path(folder_to_check)
    try:
        root_folder = root_folder.resolve(strict=False)
    except Exception as e:
        raise ArgumentValueError(f"Invalid {root_folder} path provided: {e}")
    try:
        folder_to_check = folder_to_check.resolve(strict=False)
    except Exception as e:
        raise ArgumentValueError(f"Invalid {folder_to_check} path provided: {e}")
    if root_folder == folder_to_check:
        return False
    return root_folder in folder_to_check.parents


def folder_in_subtree(
    root_folder: PathLike, branch_folder: PathLike, folders_to_check: List[PathLike]
) -> Union[Path, None]:
    root_folder = Path(root_folder).resolve(strict=False)
    branch_folder = Path(branch_folder).resolve(strict=False)
    folders_to_check = {
        Path(folder).resolve(strict=False) for folder in folders_to_check
    }
    if not folder_is_subfolder(root_folder, branch_folder):
        return None
    for folder in branch_folder.parents:
        if folder in folders_to_check:
            return folder
        if folder == root_folder:
            break
    return None


def rename_folders_in_folder(
    folder_path: PathLike,
    char_replacement: Callable[[str], str] = None,
    attempt: bool = False,
) -> None:
    folder_path = Path(folder_path).resolve(strict=False)
    if not folder_path.is_dir():
        raise ArgumentValueError(f"{folder_path} is not a valid directory.")
    char_replacement = char_replacement or (lambda name: name.replace(" ", "_"))
    for folder in folder_path.iterdir():
        if folder.is_dir():
            new_name = char_replacement(folder.name)
            new_path = folder_path / new_name
            if folder == new_path:
                continue
            if new_path.exists():
                print(
                    f"Skipping '{folder.name}' → '{new_name}' (target already exists)"
                )
                continue
            if attempt:
                print(f"[Attempt] Renaming '{folder.name}' to '{new_name}'")
            else:
                print(f"Renaming '{folder.name}' to '{new_name}'")
                shutil.move(str(folder), str(new_path))
