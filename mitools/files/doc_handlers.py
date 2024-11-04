from pathlib import Path
from typing import List, Union

from docx import Document


def read_docx_file(file_path: Union[str, Path]) -> List[str]:
    document = Document(file_path)
    text = []
    previous_style = None
    indent = "- "
    for n, para in enumerate(document.paragraphs):
        if para.text:
            formatted_text = ""
            if previous_style is None:
                previous_style = para.style.name
            if para.style.name != previous_style:
                if previous_style == "Normal" and para.style.name == "List Paragraph":
                    indent = "\t- "
                elif previous_style == "List Paragraph" and para.style.name == "Normal":
                    indent = "- "
            previous_style = para.style.name
            style_name = para.style.name
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
            text.append(f"{indent if n != 0 else ''}{formatted_text}")
    return text
