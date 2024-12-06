import re
from typing import Any, Dict, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    COLORS,
    ORIENTATIONS,
    BoolSequence,
    Color,
    DictSequence,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericTupleSequence,
    NumericTupleSequences,
    NumericType,
    StrSequence,
    StrSequences,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_bool,
    is_bool_sequence,
    is_literal,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_numeric_tuple,
    is_numeric_tuple_sequence,
    is_numeric_tuple_sequences,
    is_sequence,
    is_str,
    is_str_sequence,
    validate_length,
    validate_literal,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
)


class BoxPlotterException(Exception):
    pass


class BoxPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence, None] = None,
        **kwargs,
    ):
        self._box_params = {
            # General Axes Parameters that are independent of the number of data sequences
            "bootstrap": {"default": None, "type": Union[NumericType, None]},
            "orientation": {
                "default": "vertical",
                "type": Literal["vertical", "horizontal"],
            },
            # Specific Parameters that are based on the number of data sequences
            "notch": {"default": False, "type": Union[BoolSequence, bool]},
            "sym": {"default": None, "type": Union[StrSequence, str]},
            "whis": {"default": 1.5, "type": Union[NumericSequence, NumericType]},
            "usermedians": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence],
            },
            "conf_intervals": {
                "default": None,
                "type": Union[NumericTupleSequences, NumericTupleSequence],
            },
            "positions": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence],
            },
            "widths": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "patch_artist": {"default": False, "type": Union[BoolSequence, bool]},
            "box_labels": {
                "default": None,
                "type": Union[StrSequences, StrSequence, str],
            },
            "manage_ticks": {"default": True, "type": Union[BoolSequence, bool]},
            "autorange": {"default": False, "type": Union[BoolSequence, bool]},
            "meanline": {"default": False, "type": Union[BoolSequence, bool]},
            "showcaps": {"default": True, "type": Union[BoolSequence, bool]},
            "showbox": {"default": True, "type": Union[BoolSequence, bool]},
            "showfliers": {"default": True, "type": Union[BoolSequence, bool]},
            "showmeans": {"default": False, "type": Union[BoolSequence, bool]},
            "capprops": {"default": None, "type": Union[DictSequence, Dict]},
            "capwidths": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "boxprops": {"default": None, "type": Union[DictSequence, Dict]},
            "whiskerprops": {"default": None, "type": Union[DictSequence, Dict]},
            "flierprops": {"default": None, "type": Union[DictSequence, Dict]},
            "medianprops": {"default": None, "type": Union[DictSequence, Dict]},
            "meanprops": {"default": None, "type": Union[DictSequence, Dict]},
        }
        super().__init__(x_data=x_data, y_data=None, **kwargs)
        self._init_params.update(self._box_params)
        self._data_size = len(self.x_data)
        self._set_init_params(**kwargs)
        self.set_positions(
            y_data if y_data is not None else list(range(1, self.n_sequences + 1))
        )
        self.figure: Figure = None
        self.ax: Axes = None

    def set_bootstrap(self, bootstrap: Union[NumericType, None]):
        validate_type(bootstrap, (*NUMERIC_TYPES, type(None)), "bootstrap")
        self.bootstrap = bootstrap
        return self

    def set_orientation(self, orientation: Literal["vertical", "horizontal"]):
        validate_literal(orientation, ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_notch(self, notch: Union[BoolSequence, bool]):
        return self.set_bool_sequence(notch, "notch")

    def set_sym(self, sym: str):
        if is_str_sequence(sym):
            self.sym = sym
            self.multi_params_structure["sym"] = "sequence"
            return self
        elif is_str(sym):
            self.sym = sym
            self.multi_params_structure["sym"] = "value"
            return self
        raise ArgumentStructureError("sym must be a string or a sequence of strings")

    def set_whis(
        self,
        whis: Union[NumericSequence, NumericTupleSequence, NumericType, NumericTuple],
    ):
        if (
            is_numeric_tuple_sequences(whis)
            or is_numeric_tuple_sequence(whis)
            or is_numeric_tuple(whis, 2)
        ):
            return self.set_numeric_tuple_sequence(whis, 2, "whis")
        elif (
            is_numeric_sequences(whis) or is_numeric_sequence(whis) or is_numeric(whis)
        ):
            return self.set_numeric_sequence(whis, "whis")
        raise ArgumentStructureError(
            "whis must be a number or tuple, or sequence of numbers or tuples, or sequence of sequences of numbers or tuples."
        )

    def set_usermedians(
        self, usermedians: Union[NumericSequences, NumericSequence, None]
    ):
        if usermedians is None:
            return self
        return self.set_numeric_sequences(
            usermedians, "usermedians", single_value=False
        )

    def set_conf_intervals(
        self, conf_intervals: Union[NumericTupleSequences, NumericTupleSequence]
    ):
        if conf_intervals is None:
            return self
        return self.set_numeric_tuple_sequences(conf_intervals, 2, "conf_intervals")

    def set_positions(self, positions: Union[NumericSequences, NumericSequence]):
        return self.set_numeric_sequences(positions, "positions", single_value=False)

    def set_widths(self, widths: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(widths, "widths")

    def set_patch_artist(self, patch_artist: Union[BoolSequence, bool]):
        return self.set_bool_sequence(patch_artist, "patch_artist")

    def set_box_labels(self, box_labels: Union[StrSequences, StrSequence, str]):
        return self.set_str_sequences(box_labels, "box_labels")

    def set_manage_ticks(self, manage_ticks: Union[BoolSequence, bool]):
        return self.set_bool_sequence(manage_ticks, "manage_ticks")

    def set_autorange(self, autorange: Union[BoolSequence, bool]):
        return self.set_bool_sequence(autorange, "autorange")

    def set_meanline(self, meanline: Union[BoolSequence, bool]):
        return self.set_bool_sequence(meanline, "meanline")

    def set_showcaps(self, showcaps: Union[BoolSequence, bool]):
        return self.set_bool_sequence(showcaps, "showcaps")

    def set_showbox(self, showbox: Union[BoolSequence, bool]):
        return self.set_bool_sequence(showbox, "showbox")

    def set_showfliers(self, showfliers: Union[BoolSequence, bool]):
        return self.set_bool_sequence(showfliers, "showfliers")

    def set_showmeans(self, showmeans: Union[BoolSequence, bool]):
        return self.set_bool_sequence(showmeans, "showmeans")

    def set_capprops(self, capprops: Union[DictSequence, Dict]):
        return self.set_dict_sequence(capprops, "capprops")

    def set_capwidths(
        self, capwidths: Union[NumericSequences, NumericSequence, NumericType]
    ):
        return self.set_numeric_sequences(capwidths, "capwidths")

    def set_boxprops(self, boxprops: Union[DictSequence, Dict]):
        return self.set_dict_sequence(boxprops, "boxprops")

    def set_whiskerprops(self, whiskerprops: Union[DictSequence, Dict]):
        return self.set_dict_sequence(whiskerprops, "whiskerprops")

    def set_flierprops(self, flierprops: Union[DictSequence, Dict]):
        return self.set_dict_sequence(flierprops, "flierprops")

    def set_medianprops(self, medianprops: Union[DictSequence, Dict]):
        return self.set_dict_sequence(medianprops, "medianprops")

    def set_meanprops(self, meanprops: Union[DictSequence, Dict]):
        return self.set_dict_sequence(meanprops, "meanprops")

    def _create_box_kwargs(self, n_sequence: int):
        box_kwargs = {
            "x": [self.x_data[n_sequence]],
            "notch": self.get_sequences_param("notch", n_sequence),
            "sym": self.get_sequences_param("sym", n_sequence),
            "vert": self.orientation == "vertical",
            "whis": self.get_sequences_param("whis", n_sequence),
            "bootstrap": self.get_sequences_param("bootstrap", n_sequence),
            "usermedians": self.get_sequences_param("usermedians", n_sequence),
            "conf_intervals": self.get_sequences_param("conf_intervals", n_sequence),
            "positions": self.get_sequences_param("positions", n_sequence),
            "widths": self.get_sequences_param("widths", n_sequence),
            "patch_artist": self.get_sequences_param("patch_artist", n_sequence),
            "labels": self.get_sequences_param("box_labels", n_sequence),
            "manage_ticks": self.get_sequences_param("manage_ticks", n_sequence),
            "autorange": self.get_sequences_param("autorange", n_sequence),
            "meanline": self.get_sequences_param("meanline", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
            "showcaps": self.get_sequences_param("showcaps", n_sequence),
            "showbox": self.get_sequences_param("showbox", n_sequence),
            "showfliers": self.get_sequences_param("showfliers", n_sequence),
            "showmeans": self.get_sequences_param("showmeans", n_sequence),
            "capprops": self.get_sequences_param("capprops", n_sequence),
            "capwidths": self.get_sequences_param("capwidths", n_sequence),
            "boxprops": self.get_sequences_param("boxprops", n_sequence),
            "whiskerprops": self.get_sequences_param("whiskerprops", n_sequence),
            "flierprops": self.get_sequences_param("flierprops", n_sequence),
            "medianprops": self.get_sequences_param("medianprops", n_sequence),
            "meanprops": self.get_sequences_param("meanprops", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
        }
        box_kwargs = {k: v for k, v in box_kwargs.items() if v is not None}
        if box_kwargs["positions"] is not None and is_numeric(box_kwargs["positions"]):
            box_kwargs["positions"] = [box_kwargs["positions"]]
        return box_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            try:
                box_kwargs = self._create_box_kwargs(n_sequence)
                self.ax.boxplot(**box_kwargs)
                if self.patch_artist and self.color is not None:
                    boxes = [
                        box
                        for box in self.ax.get_children()
                        if isinstance(box, PathPatch)
                    ]
                    for box in boxes:
                        box.set_facecolor(self.color)
            except Exception as e:
                raise BoxPlotterException(f"Error while creating box plot: {e}")
