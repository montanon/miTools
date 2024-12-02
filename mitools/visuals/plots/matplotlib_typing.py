from pathlib import Path
from typing import Literal, Sequence, Union

import matplotlib.colors as mcolors
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle

_colors = list(mcolors.get_named_colors_mapping().keys())
Color = Union[str, Sequence[float]]
Marker = Union[str, int, Path, MarkerStyle]
Markers = Union[Marker, Sequence[Marker]]
Cmap = Union[
    Literal[
        "magma",
        "inferno",
        "plasma",
        "viridis",
        "cividis",
        "twilight",
        "twilight_shifted",
        "turbo",
    ],
    Colormap,
]
Norm = Union[str, Normalize]
EdgeColor = Union[Literal["face", "none", None], Color, Sequence[Color]]
FaceColor = Union[Color, Sequence[Color]]
LineStyle = Union[
    Literal["-", "--", "-.", ":", "None", "none", " ", ""],
    Sequence[Literal["-", "--", "-.", ":", "None", "none", " ", ""]],
]
Scale = Literal["linear", "log", "symlog", "logit"]
