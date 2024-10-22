import re
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
        raise ArgumentValueError(
            f"Invalid 'root_folder'={root_folder} path provided: {e}"
        )
    try:
        folder_to_check = folder_to_check.resolve(strict=False)
    except Exception as e:
        raise ArgumentValueError(
            f"Invalid 'folder_to_check'={folder_to_check} path provided: {e}"
        )
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


def remove_characters_from_string(string: str, characters: str = None) -> str:
    if characters is None:
        characters = r'[\\/*?:"<>|]'
    return re.sub(characters, "", string)


def remove_characters_from_filename(file_path: PathLike, characters: str = None) -> str:
    file_path = Path(file_path)
    filename = remove_characters_from_string(
        string=file_path.stem, characters=characters
    )
    return file_path.with_name(f"{filename}{file_path.suffix}")


def handle_duplicated_filenames(file_path: Path) -> Path:
    counter = 1
    new_file = file_path
    while new_file.exists():
        new_file = file_path.with_name(f"{file_path.stem}_{counter}{file_path.suffix}")
        counter += 1
    return new_file


def rename_file(file: PathLike, new_name: str = None) -> None:
    file = Path(file)
    sanitized_name = (
        remove_characters_from_filename(file) if new_name is None else new_name
    )
    new_file = handle_duplicated_filenames(file.with_name(sanitized_name))
    shutil.move(str(file), str(new_file))
    print(f"Renamed '{file.name}' to '{new_file.name}'")


def rename_files_in_folder(
    folder_path: PathLike,
    file_types: List[str] = None,
    renaming_function: Callable[[str], str] = None,
) -> None:
    folder = Path(folder_path).resolve(strict=True)
    for file in folder.iterdir():
        if not file.is_file():
            continue  # Skip non-files
        if file_types and file.suffix.lower() not in file_types:
            continue  # Skip files not in the specified types
        try:
            rename_file(
                file, None if renaming_function is None else renaming_function(file)
            )
        except Exception as e:
            print(f"Error processing '{file.name}': {e}")


if __name__ == "__main__":
    pass
