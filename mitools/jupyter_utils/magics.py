import os
import base64
from IPython.core.magic import register_cell_magic, register_line_magic
from IPython.display import HTML, display


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_FOLDER = 'alarms'
ALARM_FILENAME = 'short-success-sound-glockenspiel-treasure-video-game-6346.mp3'
ALARM_FILE_PATH = os.path.join(CURRENT_DIR, ALARM_FOLDER, ALARM_FILENAME)

@register_cell_magic
def execute(line, cell=None):
    '''Executes the current cell if line evaluates to True.'''
    if not eval(line):
        return
    global_ns = get_ipython().user_ns
    exec(cell, global_ns)

@register_line_magic
def notify(line):
    '''Plays an alarm sound.'''
    play_sound(ALARM_FILE_PATH)

def play_sound(filename):
    '''Plays sound from given filename in Jupyter.'''
    src = """
    <audio controls autoplay style="display:none">
      <source src="data:audio/mp3;base64,{}" type="audio/mp3" />
    </audio>
    """.format(base64.b64encode(open(filename, "rb").read()).decode())
    display(HTML(src))

def load_ipython_extension(shell):
    '''Registers the magics when the extension loads.'''
    shell.register_magic_function(execute, 'line_cell')
    shell.register_magic_function(notify, 'line')

def unload_ipython_extension(shell):
    '''Unregisters the magics when the extension unloads.'''
    del shell.magics_manager.magics['cell']['execute']
    del shell.magics_manager.magics['line']['alarm']
