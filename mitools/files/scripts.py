import PyPDF2
import os
from os import PathLike
import re
import argparse

import sys
sys.path.append("/Users/sebastian/Desktop/MontagnaInc/miTools/mitools")
from ..utils import fuzz_string_in_string

PATTERN = '^([A-Za-z0-9.]+-)+[A-Za-z0-9]+.pdf$'

def extract_pdf_metadata(pdf_filename: PathLike):
    metadata = {}
    with open(pdf_filename, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        doc_info = pdf_reader.metadata
        for key in doc_info:
            metadata[key[1:]] = doc_info[key]
    return metadata

def extract_pdf_title(pdf_filename: PathLike):
    metadata = extract_pdf_metadata(pdf_filename)
    if 'Title' in metadata:
        return metadata['Title']
    else:
        raise Exception(f'{os.path.basename(pdf_filename)} has no title in metadata')
    
def set_pdf_filename_as_title(pdf_filename: PathLike, title: str):
    title = re.sub('[:\)\(\/]*', '', title)
    os.rename(pdf_filename, os.path.join(os.path.dirname(pdf_filename), f'{title}.pdf'))

def set_folder_pdf_filenames_as_title(folder: PathLike):
    pdfs = [f for f in os.listdir(folder) if f.endswith('pdf')]
    for pdf in pdfs:
        title = extract_pdf_title(pdf)
        title_in_name = fuzz_string_in_string(title, pdf, 85)
        if (re.match(PATTERN, pdf) or not title_in_name) and not pdf.startswith('RELEVANT'):
            try:
                set_pdf_filename_as_title(os.path.join(folder, pdf), title)
            except Exception as e:
                print(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PDF operations.")
    parser.add_argument("pdf_filename", type=str, help="Path to the PDF file.")
    subparsers = parser.add_subparsers(dest="command")

    # Command to set filename as title
    rename_parser = subparsers.add_parser("rename", help="Rename PDF based on its title.")
    rename_parser.set_defaults(func=set_pdf_filename_as_title)

    # Command to set filename as title
    rename_parser = subparsers.add_parser("rename_folder", help="Rename PDF folder based on its title.")
    rename_parser.set_defaults(func=set_folder_pdf_filenames_as_title)

    args = parser.parse_args()
    if args.command:
        result = args.func(args.pdf_filename)
        if args.command in ["metadata", "title"]:
            print(result)
    else:
        parser.print_help()