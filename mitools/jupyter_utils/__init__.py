def is_ipython():
    try:
        get_ipython()
        return True
    except NameError:
        return False


if is_ipython():
    from mitools.jupyter_utils.magics import load_ipython_extension

    load_ipython_extension(get_ipython())
