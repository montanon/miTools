import pathlib
from pathlib import Path
from typing import List, Union


def folder_is_subfolder(root_folder: Path, folder_to_check: Path) -> bool:
    root_folder = root_folder.resolve()
    folder_to_check = folder_to_check.resolve()
    return root_folder in folder_to_check.parents


def folder_in_subtree(
    root_folder: Path, branch_folder: Path, folders_to_check: List[Path]
) -> Union[Path, None]:
    root_folder = root_folder.resolve()
    branch_folder = branch_folder.resolve()
    folders_to_check = [folder.resolve() for folder in folders_to_check]
    if not folder_is_subfolder(root_folder, branch_folder):
        return None
    intermediate_folders = list(branch_folder.parents)
    for folder in intermediate_folders:
        if folder in folders_to_check:
            return folder
        if folder == root_folder:
            break
    return None
