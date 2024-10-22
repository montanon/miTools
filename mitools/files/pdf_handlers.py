import os
import re
from os import PathLike
from pathlib import Path
from typing import Dict, Union

import PyPDF2

from mitools.exceptions import ArgumentValueError
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


if __name__ == "__main__":
    pass
