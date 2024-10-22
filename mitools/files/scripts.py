import argparse
import os
import re
import shutil
import sys
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, List

import PyPDF2

sys.path.append("/Users/sebastian/Desktop/MontagnaInc/miTools/mitools")
from mitools.utils import fuzz_string_in_string

PATTERN = "^([A-Za-z0-9.]+-)+[A-Za-z0-9]+.pdf$"


def extract_pdf_metadata(pdf_filename: PathLike) -> Dict:
    metadata = {}
    with open(pdf_filename, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        doc_info = pdf_reader.metadata
        for key in doc_info:
            metadata[key[1:]] = doc_info[key]
    return metadata


def extract_pdf_title(pdf_filename: PathLike) -> str:
    metadata = extract_pdf_metadata(pdf_filename)
    if "Title" in metadata:
        return metadata["Title"]
    else:
        raise Exception(f"{os.path.basename(pdf_filename)} has no title in metadata")


def set_pdf_filename_as_title(pdf_filename: PathLike, title: str) -> None:
    title = re.sub("[:\)\(\/]*", "", title)
    os.rename(pdf_filename, os.path.join(os.path.dirname(pdf_filename), f"{title}.pdf"))


def set_folder_pdf_filenames_as_title(folder: PathLike) -> None:
    pdfs = [f for f in os.listdir(folder) if f.endswith("pdf")]
    for pdf in pdfs:
        title = extract_pdf_title(pdf)
        title_in_name = fuzz_string_in_string(title, pdf, 85)
        if (re.match(PATTERN, pdf) or not title_in_name) and not pdf.startswith(
            "RELEVANT"
        ):
            try:
                set_pdf_filename_as_title(os.path.join(folder, pdf), title)
            except Exception as e:
                print(e)


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
    parser = argparse.ArgumentParser(description="PDF operations.")
    parser.add_argument("path", type=str, help="Path to the PDF file or folder.")
    subparsers = parser.add_subparsers(dest="command")

    # Command to set filename as title for a single PDF
    rename_parser = subparsers.add_parser(
        "rename", help="Rename a single PDF based on its title."
    )
    rename_parser.set_defaults(func=set_pdf_filename_as_title)

    # Command to set filenames as titles within a folder
    rename_folder_parser = subparsers.add_parser(
        "rename_folder", help="Rename all PDFs in a folder based on their titles."
    )
    rename_folder_parser.set_defaults(func=set_folder_pdf_filenames_as_title)

    # Command to rename files in a folder, leveraging the title if available
    rename_files_parser = subparsers.add_parser(
        "rename_files", help="Rename files in a folder, using PDF titles if available."
    )
    rename_files_parser.set_defaults(func=rename_files_in_folder)

    args = parser.parse_args()
    if args.command:
        if args.command in ["rename", "rename_folder"]:
            args.func(args.path)  # These commands don't return a value to print
        elif args.command == "rename_files":
            args.func(args.path)  # No return value to print
    else:
        parser.print_help()
