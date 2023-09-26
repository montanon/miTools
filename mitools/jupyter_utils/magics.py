def execute(line, cell=None):
    '''Executes the current line/cell if line evaluates to True.'''
    if not eval(line):
        return

    get_ipython().ex(cell)

def load_ipython_extension(shell):
    '''Registers the execute magic when the extension loads.'''
    get_ipython().register_magic_function(execute, 'line_cell')

def unload_ipython_extension(shell):
    '''Unregisters the execute magic when the extension unloads.'''
    del get_ipython().magics_manager.magics['cell']['execute']