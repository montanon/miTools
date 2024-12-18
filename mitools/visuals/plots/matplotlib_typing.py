from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

from matplotlib.colors import Colormap, Normalize, get_named_colors_mapping
from matplotlib.markers import MarkerStyle
from numpy import integer

NumericType = Union[int, float, integer]
NumericTuple = Tuple[NumericType, NumericType]
NumericTupleSequence = Sequence[NumericTuple]
NumericTupleSequences = Sequence[NumericTupleSequence]
NumericSequence = Sequence[NumericType]
NumericSequences = Sequence[NumericSequence]
DictSequence = Sequence[Dict[str, Any]]
DictSequences = Sequence[DictSequence]
Color = Union[str, Sequence[float]]
StrSequence = Sequence[str]
StrSequences = Sequence[StrSequence]
ColorSequence = Sequence[Color]
ColorSequences = Sequence[ColorSequence]
COLORS = set(get_named_colors_mapping().keys())
MARKERS = set(MarkerStyle.markers.keys()).union(set(MarkerStyle.filled_markers))
MARKERS_FILLSTYLES = set(MarkerStyle.fillstyles)
Marker = Union[str, int, Path, MarkerStyle, dict]
MarkerSequence = Union[Marker, Sequence[Marker]]
MarkerSequences = Sequence[MarkerSequence]
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
KERNELS = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
ORIENTATIONS = ["horizontal", "vertical"]
BANDWIDTH_METHODS = ["scott", "silverman"]
BARS_ALIGN = ["center", "edge"]
NORMALIZATIONS = [
    "linear",
    "log",
    "symlog",
    "asinh",
    "logit",
    "function",
    "functionlog",
]
CMAPS = [
    "magma",
    "inferno",
    "plasma",
    "viridis",
    "cividis",
    "twilight",
    "twilight_shifted",
    "turbo",
]
BINS = ["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
Bins = Union[int, str]
BinsSequence = Sequence[Bins]
BoolSequence = Sequence[bool]
BinsSequences = Sequence[BinsSequence]
LiteralSequence = Sequence[Literal["literal"]]
LiteralSequences = Sequence[LiteralSequence]
CmapSequence = Sequence[Cmap]
Norm = Union[str, Normalize]
NormSequence = Sequence[Norm]
EdgeColor = Union[Literal["face", "none", None], Color, Sequence[Color]]
EdgeColorSequence = Sequence[EdgeColor]
EdgeColorSequences = Sequence[EdgeColorSequence]
LineStyle = Union[
    Literal["-", "--", "-.", ":", "None", "none", " ", ""],
    Sequence[Literal["-", "--", "-.", ":", "None", "none", " ", ""]],
]
Scale = Literal["linear", "log", "symlog", "logit"]
TickParams = Dict[str, Any]
LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    "None",
    "none",
    " ",
    "",
    "dotted",
    "dashed",
    "dashdot",
    "solid",
]
HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
HIST_ALIGN = ["left", "mid", "right"]
HIST_HISTTYPE = ["bar", "barstacked", "step", "stepfilled"]
TICKPARAMS = [
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
