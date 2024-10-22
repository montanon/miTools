from os import PathLike
from pathlib import Path
from typing import List, Union

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
