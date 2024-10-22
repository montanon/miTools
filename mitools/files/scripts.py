import argparse
import sys

sys.path.append("/Users/sebastian/Desktop/MontagnaInc/miTools/mitools")
from mitools.files import (
    rename_files_in_folder,
    set_folder_pdf_filenames_as_title,
    set_pdf_filename_as_title,
)

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
