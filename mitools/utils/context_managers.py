from contextlib import contextmanager
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np


class SavePlotContext:
    def __init__(self, regression_plot: PathLike):
        self.regression_plot = regression_plot
        self.axes = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.axes, plt.Axes):
            self.axes.figure.savefig(self.regression_plot)
        elif isinstance(self.axes, (np.ndarray, list)) and all(isinstance(ax, plt.Axes) for ax in self.axes.flat):
            self.axes.flat[0].figure.savefig(self.regression_plot)
        else:
            raise TypeError("Must assign a matplotlib Axes or array/list of Axes to 'axes' attribute.")

@contextmanager
def save_plot(regression_plot: PathLike):
    context = SavePlotContext(regression_plot)
    yield context
    context.__exit__(None, None, None)
    