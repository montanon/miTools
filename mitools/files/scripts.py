import argparse
import os
import re
import shutil
import sys
from os import PathLike
from typing import Dict

import PyPDF2

sys.path.append("/Users/sebastian/Desktop/MontagnaInc/miTools/mitools")
from mitools.utils import fuzz_string_in_string

PATTERN = '^([A-Za-z0-9.]+-)+[A-Za-z0-9]+.pdf$'

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
    if 'Title' in metadata:
        return metadata['Title']
    else:
        raise Exception(f'{os.path.basename(pdf_filename)} has no title in metadata')
    
def set_pdf_filename_as_title(pdf_filename: PathLike, title: str) -> None:
    title = re.sub('[:\)\(\/]*', '', title)
    os.rename(pdf_filename, os.path.join(os.path.dirname(pdf_filename), f'{title}.pdf'))

def set_folder_pdf_filenames_as_title(folder: PathLike) -> None:
    pdfs = [f for f in os.listdir(folder) if f.endswith('pdf')]
    for pdf in pdfs:
        title = extract_pdf_title(pdf)
        title_in_name = fuzz_string_in_string(title, pdf, 85)
        if (re.match(PATTERN, pdf) or not title_in_name) and not pdf.startswith('RELEVANT'):
            try:
                set_pdf_filename_as_title(os.path.join(folder, pdf), title)
            except Exception as e:
                print(e)

def rename_folders_in_folder(folder_path: PathLike) -> None:
    for name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, name)):
            new_name = name.replace(' ', '_')
            shutil.move(os.path.join(folder_path, name), os.path.join(folder_path, new_name))

def rename_files_in_folder(folder_path: PathLike) -> None:
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.pdf'):
                old_file = os.path.join(root, filename)
                try:
                    title = extract_pdf_title(old_file)
                    new_filename = re.sub(r'[\\/*?:"<>|]', '', title) + '.pdf'  # Remove illegal characters for filenames
                except Exception as e:
                    print(f"Error extracting title for {filename}: {e}")
                    continue  # Skip renaming if title extraction fails
                max_length = 255 - len(".pdf") - 10  # 10 characters reserved for a counter and separator
                if len(new_filename) > max_length:
                    new_filename = f"{new_filename[:max_length]}.pdf"
                new_file = os.path.join(root, new_filename)
                counter = 1
                while os.path.exists(new_file):  # Handle name conflicts
                    new_file = os.path.join(root, new_filename.replace(".pdf", f"_{counter}.pdf"))
                    counter += 1
                shutil.move(old_file, new_file)
                print(f"Renamed '{filename}' to '{new_filename}'")
            else:
                continue  # Skip non-PDF files
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PDF operations.")
    parser.add_argument("path", type=str, help="Path to the PDF file or folder.")
    subparsers = parser.add_subparsers(dest="command")

    # Command to set filename as title for a single PDF
    rename_parser = subparsers.add_parser("rename", help="Rename a single PDF based on its title.")
    rename_parser.set_defaults(func=set_pdf_filename_as_title)

    # Command to set filenames as titles within a folder
    rename_folder_parser = subparsers.add_parser("rename_folder", help="Rename all PDFs in a folder based on their titles.")
    rename_folder_parser.set_defaults(func=set_folder_pdf_filenames_as_title)

    # Command to rename files in a folder, leveraging the title if available
    rename_files_parser = subparsers.add_parser("rename_files", help="Rename files in a folder, using PDF titles if available.")
    rename_files_parser.set_defaults(func=rename_files_in_folder)

    args = parser.parse_args()
    if args.command:
        if args.command in ["rename", "rename_folder"]:
            args.func(args.path)  # These commands don't return a value to print
        elif args.command == "rename_files":
            args.func(args.path)  # No return value to print
    else:
        parser.print_help()