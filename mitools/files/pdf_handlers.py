import os
import re
from os import PathLike
from pathlib import Path
from typing import Dict, Union

import PyPDF2

from mitools.exceptions import ArgumentTypeError, ArgumentValueError
from mitools.files import remove_characters_from_string, rename_file
from mitools.utils import fuzz_string_in_string

PATTERN = "^([A-Za-z0-9.]+-)+[A-Za-z0-9]+.pdf$"


def extract_pdf_metadata(pdf_filename: PathLike) -> Union[Dict[str, str], None]:
    pdf_filename = Path(pdf_filename)
    if not pdf_filename.is_file():
        raise ArgumentValueError(f"'{pdf_filename}' is not a valid file path.")
    metadata = {}
    try:
        with open(pdf_filename, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            doc_info = pdf_reader.metadata or {}  # Handle case where no metadata exists
            for key, value in doc_info.items():
                sanitized_key = key.lstrip("/")  # Remove prefix if it exists
                metadata[sanitized_key] = (
                    str(value).encode("utf-8", errors="ignore").decode("utf-8")
                )
    except (FileNotFoundError, PyPDF2.errors.PdfReadError) as e:
        raise ArgumentValueError(f"Error reading PDF '{pdf_filename}': {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
    return metadata


def extract_pdf_title(pdf_filename: PathLike) -> str:
    pdf_filename = Path(pdf_filename)
    if not pdf_filename.is_file():
        raise ArgumentValueError(f"'{pdf_filename}' is not a valid file path.")
    if not pdf_filename.suffix.lower() == ".pdf":
        raise ArgumentTypeError(f"'{pdf_filename}' is not a valid PDF file.")
    metadata = extract_pdf_metadata(pdf_filename)
    if "Title" in metadata:
        return metadata["Title"]
    else:
        raise ArgumentValueError(f"'{pdf_filename}' has no title in its metadata.")


def set_pdf_title_as_filename(
    pdf_filename: PathLike, attempt: bool = False, overwrite: bool = False
) -> None:
    pdf_filename = Path(pdf_filename).resolve(strict=True)
    if pdf_filename.suffix.lower() != ".pdf":
        raise ArgumentTypeError(f"'{pdf_filename}' is not a valid PDF file.")
    title = extract_pdf_title(pdf_filename)
    title = remove_characters_from_string(title).replace(" ", "_")
    new_filename = pdf_filename.with_name(f"{title}.pdf")
    rename_file(
        file=pdf_filename, new_name=new_filename, overwrite=overwrite, attempt=attempt
    )


def set_folder_pdfs_titles_as_filenames(
    folder_path: PathLike, overwrite: bool = False
) -> None:
    try:
        folder = Path(folder_path).resolve(strict=True)
    except FileNotFoundError as e:
        raise ArgumentValueError(f"Invalid 'folder_path'={folder_path} provided. {e}")
    for file in folder.iterdir():
        if not file.is_file() or file.suffix.lower() != ".pdf":
            continue  # Skip non-PDF files
        try:
            set_pdf_title_as_filename(file, overwrite=overwrite)
        except Exception as e:
            print(f"Error processing '{file.name}': {e}")


if __name__ == "__main__":
    pass
