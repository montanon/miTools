import argparse
import sys
from pathlib import Path

sys.path.append("/Users/sebastian/Desktop/MontagnaInc/miTools/mitools")
from mitools.files import (
    convert_file,
    rename_files_in_folder,
    set_folder_pdfs_titles_as_filenames,
    set_pdf_title_as_filename,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF operations.")
    parser.add_argument("path", type=str, help="Path to the PDF file or folder.")
    subparsers = parser.add_subparsers(dest="command")

    # Command to set filename as title for a single PDF
    rename_parser = subparsers.add_parser(
        "rename", help="Rename a single PDF based on its title."
    )
    rename_parser.set_defaults(func=set_pdf_title_as_filename)

    # Command to set filenames as titles within a folder
    rename_folder_parser = subparsers.add_parser(
        "rename_folder", help="Rename all PDFs in a folder based on their titles."
    )
    rename_folder_parser.set_defaults(func=set_folder_pdfs_titles_as_filenames)

    # Command to rename files in a folder, leveraging the title if available
    rename_files_parser = subparsers.add_parser(
        "rename_files", help="Rename files in a folder, using PDF titles if available."
    )
    rename_files_parser.set_defaults(func=rename_files_in_folder)

    # Command to convert a file
    convert_parser = subparsers.add_parser(
        "convert_file", help="Convert a file to a specified format."
    )
    convert_parser.add_argument(
        "source_file", type=Path, help="Path to the source file."
    )
    convert_parser.add_argument(
        "output_file", type=Path, help="Path for the output file."
    )
    convert_parser.add_argument(
        "output_format",
        type=str,
        help="Desired output format (e.g., 'pdf', 'html', etc.).",
    )
    convert_parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow operation if the output file already exists.",
    )
    convert_parser.add_argument(
        "--overwrite",
        action="store_false",
        help="Overwrite the output file if it exists.",
    )
    convert_parser.set_defaults(func=convert_file)

    args = parser.parse_args()
    if args.command:
        if args.command in ["rename", "rename_folder", "convert_file"]:
            # Call the function with the specified arguments
            if args.command == "convert_file":
                args.func(
                    source_file=args.source_file,
                    output_file=args.output_file,
                    output_format=args.output_format,
                    exist_ok=args.exist_ok,
                    overwrite=args.overwrite,
                )
            else:
                args.func(args.path)
        elif args.command == "rename_files":
            args.func(args.path)
    else:
        parser.print_help()
