def _setup_autoreload():
    """
    Private function to setup autoreload in an IPython environment.
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython:
            ipython.run_line_magic("load_ext", "autoreload")
            ipython.run_line_magic("autoreload", "2")
    except ImportError:
        pass  # Fails silently if not in an IPython environment


# Automatically set up autoreload when this module is imported
_setup_autoreload()

import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mitools import clustering as clustering
from mitools import economic_complexity as ec
from mitools import etl, files, google_utils, jupyter_utils, nlp, scraping, utils
from mitools import notebooks as nb
from mitools import oldregressions as oreg
from mitools import pandas_utils as pdf
from mitools import regressions as reg
from mitools.context import DEV
from mitools.country_utils import name_converter
from mitools.project import Project
from mitools.utils import iprint
