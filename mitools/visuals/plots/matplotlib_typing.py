from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Union

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
TickParams = Dict[str, Any]
_tickparams = [
    "size",
    "width",
    "color",
    "tickdir",
    "pad",
    "labelsize",
    "labelcolor",
    "zorder",
    "gridOn",
    "tick1On",
    "tick2On",
    "label1On",
    "label2On",
    "length",
    "direction",
    "left",
    "bottom",
    "right",
    "top",
    "labelleft",
    "labelbottom",
    "labelright",
    "labeltop",
    "labelrotation",
    "grid_agg_filter",
    "grid_alpha",
    "grid_animated",
    "grid_antialiased",
    "grid_clip_box",
    "grid_clip_on",
    "grid_clip_path",
    "grid_color",
    "grid_dash_capstyle",
    "grid_dash_joinstyle",
    "grid_dashes",
    "grid_data",
    "grid_drawstyle",
    "grid_figure",
    "grid_fillstyle",
    "grid_gapcolor",
    "grid_gid",
    "grid_in_layout",
    "grid_label",
    "grid_linestyle",
    "grid_linewidth",
    "grid_marker",
    "grid_markeredgecolor",
    "grid_markeredgewidth",
    "grid_markerfacecolor",
    "grid_markerfacecoloralt",
    "grid_markersize",
    "grid_markevery",
    "grid_mouseover",
    "grid_path_effects",
    "grid_picker",
    "grid_pickradius",
    "grid_rasterized",
    "grid_sketch_params",
    "grid_snap",
    "grid_solid_capstyle",
    "grid_solid_joinstyle",
    "grid_transform",
    "grid_url",
    "grid_visible",
    "grid_xdata",
    "grid_ydata",
    "grid_zorder",
    "grid_aa",
    "grid_c",
    "grid_ds",
    "grid_ls",
    "grid_lw",
    "grid_mec",
    "grid_mew",
    "grid_mfc",
    "grid_mfcalt",
    "grid_ms",
]