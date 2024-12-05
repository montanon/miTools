import re
from typing import Any, Dict, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import (
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    COLORS,
    HATCHES,
    BoolSequence,
    Color,
    ColorSequence,
    ColorSequences,
    DictSequence,
    LiteralSequence,
    LiteralSequences,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericTupleSequence,
    NumericType,
    StrSequence,
    StrSequences,
)
from mitools.visuals.plots.plotter import Plotter
from mitools.visuals.plots.validations import (
    NUMERIC_TYPES,
    SEQUENCE_TYPES,
    is_sequence,
    validate_length,
    validate_sequence_length,
    validate_sequence_type,
    validate_type,
    validate_value_in_options,
)


class PiePlotterException(Exception):
    pass


class PiePlotter(Plotter):
    def __init__(
        self,
        data: Union[NumericSequences, NumericSequence],
        pos_data: Union[NumericSequences, NumericSequence, None] = None,
        **kwargs,
    ):
        self._pie_params = {
            # Specific Parameters that are based on the number of data sequences
            "explode": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "labels": {"default": None, "type": Union[StrSequences, StrSequence]},
            "hatch": {
                "default": None,
                "type": Union[LiteralSequences, LiteralSequence, Literal["hatches"]],
            },
            "autopct": {
                "default": None,
                "type": Union[Sequence[Union[str, callable]], Union[str, callable]],
            },
            "pctdistance": {
                "default": 0.6,
                "type": Union[NumericSequence, NumericType],
            },
            "labeldistance": {
                "default": 1.1,
                "type": Union[NumericSequence, NumericType],
            },
            "shadow": {"default": False, "type": Union[BoolSequence, bool]},
            "startangle": {
                "default": None,
                "type": Union[NumericSequence, NumericType],
            },
            "radius": {"default": None, "type": Union[NumericSequence, NumericType]},
            "counterclock": {"default": True, "type": Union[BoolSequence, bool]},
            "wedgeprops": {"default": None, "type": Union[DictSequence, Dict]},
            "textprops": {"default": None, "type": Union[DictSequence, Dict]},
            "center": {
                "default": (0, 0),
                "type": Union[NumericTupleSequence, NumericTuple],
            },
            "frame": {"default": False, "type": Union[BoolSequence, bool]},
            "rotatelabels": {"default": False, "type": Union[BoolSequence, bool]},
            "normalize": {"default": True, "type": Union[BoolSequence, bool]},
        }
        super().__init__(x_data=data, y_data=pos_data, **kwargs)
        self._init_params.update(self._pie_params)
        self._set_init_params(**kwargs)
        self.figure: Figure = None
        self.ax: Axes = None

    def set_color(self, color: Union[ColorSequences, ColorSequence, Color]):
        return self.set_color_sequences(color, "color")

    def set_explode(
        self, explode: Union[NumericSequences, NumericSequence, NumericType]
    ):
        return self.set_numeric_sequences(explode, "explode")

    def set_labels(self, labels: Union[StrSequences, StrSequence, str]):
        return self.set_str_sequences(labels, "labels")

    def set_hatch(
        self, hatch: Union[LiteralSequences, LiteralSequence, Literal["hatches"]]
    ):
        return self.set_literal_sequences(hatch, HATCHES, "hatch")

    def set_autopct(
        self, autopct: Union[Sequence[Union[str, callable]], Union[str, callable]]
    ):
        if is_sequence(autopct):
            validate_sequence_type(autopct, (str, callable), "autopct")
        else:
            validate_type(autopct, (str, callable), "autopct")
        self.autopct = autopct
        return self

    def set_pctdistance(self, pctdistance: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(pctdistance, "pctdistance")

    def set_labeldistance(self, labeldistance: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(labeldistance, "labeldistance", min_value=0)

    def set_shadow(self, shadow: Union[BoolSequence, bool]):
        return self.set_bool_sequence(shadow, "shadow")

    def set_startangle(self, startangle: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(startangle, "startangle")

    def set_radius(self, radius: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequence(radius, "radius", min_value=0)

    def set_counterclock(self, counterclock: Union[BoolSequence, bool]):
        return self.set_bool_sequence(counterclock, "counterclock")

    def set_wedgeprops(self, **kwargs):
        self.wedgeprops = kwargs
        return self

    def set_textprops(self, kwargs: Union[DictSequence, Dict]):
        self.textprops = kwargs
        return self

    def set_center(self, center: Union[NumericTupleSequence, NumericTuple]):
        return self.set_numeric_tuple_sequence(center, "center")

    def set_frame(self, frame: Union[BoolSequence, bool]):
        return self.set_bool_sequence(frame, "frame")

    def set_rotatelabels(self, rotatelabels: Union[BoolSequence, bool]):
        return self.set_bool_sequence(rotatelabels, "rotatelabels")

    def set_normalize(self, normalize: Union[BoolSequence, bool]):
        return self.set_bool_sequence(normalize, "normalize")

    def _create_plot(self):
        pie_kwargs = {
            "x": self.x_data,
            "explode": self.explode,
            "hatch": self.hatch,
            "labels": self.label,
            "colors": self.color,
            "autopct": self.autopct,
            "pctdistance": self.pctdistance,
            "labeldistance": self.labeldistance,
            "shadow": self.shadow,
            "startangle": self.startangle,
            "radius": self.radius,
            "counterclock": self.counterclock,
            "wedgeprops": self.wedgeprops,
            "textprops": self.textprops,
            "center": self.center,
            "frame": self.frame,
            "rotatelabels": self.rotatelabels,
            "normalize": self.normalize,
        }
        pie_kwargs = {k: v for k, v in pie_kwargs.items() if v is not None}

        try:
            self.ax.pie(**pie_kwargs)
            self.ax.axis("equal")
        except Exception as e:
            raise PiePlotterException(f"Error while creating pie plot: {e}")
