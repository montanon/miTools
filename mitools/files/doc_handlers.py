from os import PathLike
from pathlib import Path
from typing import List, Union

import pypandoc
from docx import Document

from mitools.exceptions import ArgumentValueError


def read_docx_file(file_path: Union[str, Path], indent: str = "-") -> List[str]:
    document = Document(file_path)
    paragraphs = []
    previous_style = None
    previous_punctuation = False
    n_tabs = 0
    for n, para in enumerate(document.paragraphs):
        if para.text:
            text = para.text.strip()
            formatted_text = ""
            if previous_style is None:
                previous_style = para.style.name
            if (
                para.style.name != previous_style
                or text.endswith(":")
                and not previous_style.find("Heading")
            ):
                if previous_style == "Normal" and para.style.name == "List Paragraph":
                    n_tabs += 1
                elif previous_style == "List Paragraph" and para.style.name == "Normal":
                    n_tabs -= 1
                if text.endswith(":"):
                    n_tabs += 1
            previous_style = para.style.name
            style_name = para.style.name
            previous_punctuation = text.endswith(":")
            if "Heading" in style_name:
                formatted_text += f"{'#'*int(style_name[-1])} "
            highlighted = False
            for run in para.runs:
                run_text = run.text
                if run.bold:
                    run_text = f"**{run_text}**"  # Mark bold text
                if run.italic:
                    run_text = f"*{run_text}*"  # Mark italic text
                if run.font.highlight_color:
                    highlighted = True
                formatted_text += run_text
            if highlighted:
                formatted_text = f"<highlight>{formatted_text}<\highlight>"
            indentation = "\t" * n_tabs + indent
            paragraphs.append(
                f"{indentation if n_tabs != 0 else ''}{formatted_text[:20]} -- ({para.style.name}, {previous_style}, {previous_punctuation})"
            )
    return paragraphs


def convert_docx_to_pdf(
    source_file: PathLike,
    output_file: PathLike,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    convert_file(
        source_file=source_file,
        output_file=output_file,
        output_format="pdf",
        exist_ok=exist_ok,
        overwrite=overwrite,
    )


def convert_file(
    source_file: PathLike,
    output_file: PathLike,
    output_format: str,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    if Path(output_file).exists() and not exist_ok and not overwrite:
        raise ArgumentValueError(f"'{output_file}' already exists.")
    elif Path(output_file).exists() and exist_ok and not overwrite:
        return
    output = pypandoc.convert_file(
        source_file=source_file, to=output_format, outputfile=output_file
    )
    assert output == "", f"Conversion failed: {output}"


def convert_directory_files(
    source_directory: PathLike,
    output_directory: PathLike,
    input_format: str,
    output_format: str,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    for file in source_directory.glob(f"*.{input_format}"):
        output_file = (
            output_directory / file.with_suffix(f".{output_format}").name
            if output_directory
            else file.with_suffix(f".{output_format}")
        )
        convert_file(
            source_file=file,
            output_file=output_file,
            output_format=output_format,
            exist_ok=exist_ok,
            overwrite=overwrite,
        )


def convert_directory_docxs_to_pdfs(
    source_directory: PathLike,
    output_directory: PathLike = None,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    convert_directory_files(
        source_directory=source_directory,
        output_directory=output_directory,
        input_format="docx",
        output_format="pdf",
        exist_ok=exist_ok,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    pass
