import ast
import base64
import os
from concurrent.futures import ThreadPoolExecutor

from IPython import get_ipython
from IPython.core.magic import register_cell_magic, register_line_magic
from IPython.display import HTML, Markdown, display
from playsound import playsound

from mitools.exceptions import ArgumentValueError

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_FOLDER = "alarms"
ALARM_FILENAME = "short-success-sound-glockenspiel-treasure-video-game-6346.mp3"
ALARM_FILE_PATH = os.path.join(CURRENT_DIR, ALARM_FOLDER, ALARM_FILENAME)

executor = ThreadPoolExecutor(max_workers=1)


@register_cell_magic
def execute(line, cell=None):
    """Executes the current cell if line evaluates to True."""
    try:
        condition = ast.literal_eval(line)
        if not isinstance(condition, bool):
            raise ArgumentValueError("Condition must evaluate to a boolean value.")
    except Exception as e:
        display(Markdown(f"**Error in condition:** {e}"))
        return
    if condition:
        local_ns = {}
        global_ns = get_ipython().user_ns
        exec(cell, global_ns, local_ns)


@register_line_magic
def notify(line):
    """Plays an alarm sound."""
    try:
        executor.submit(playsound, ALARM_FILE_PATH)
    except FileNotFoundError:
        display(HTML("<p style='color:red'>Error: Alarm file not found.</p>"))
    except Exception as e:
        display(HTML(f"<p style='color:red'>Error: {e}</p>"))


def load_ipython_extension(shell):
    """Registers the magics when the extension loads."""
    shell.register_magic_function(execute, "line_cell")
    shell.register_magic_function(notify, "line")


def unload_ipython_extension(shell):
    """Unregisters the magics when the extension unloads."""
    del shell.magics_manager.magics["cell"]["execute"]
    del shell.magics_manager.magics["line"]["alarm"]
