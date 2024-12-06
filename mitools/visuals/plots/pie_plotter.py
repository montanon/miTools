from typing import Callable, Dict, Literal, Sequence, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mitools.exceptions import ArgumentStructureError
from mitools.visuals.plots.matplotlib_typing import (
    HATCHES,
    BoolSequence,
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
    is_dict_sequence,
    is_sequence,
    validate_sequence_type,
    validate_type,
)


class PiePlotterException(Exception):
    pass


class PiePlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence, None] = None,
        ax: Axes = None,
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
        super().__init__(x_data=x_data, y_data=None, **kwargs)
        self._init_params.update(self._pie_params)
        self._set_init_params(**kwargs)
        if y_data is not None:
            self.set_center(y_data)

    def set_frame(self, frame: bool):
        if isinstance(frame, bool):
            self.frame = frame
            self.multi_params_structure["frame"] = "value"
            return self

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
        self,
        autopct: Union[
            Sequence[Union[str, Callable, None]], Union[str, Callable, None]
        ],
    ):
        if is_sequence(autopct):
            validate_sequence_type(
                autopct,
                (str, Callable),
                "autopct",
            )
            self.autopct = autopct
            self.multi_params_structure["autopct"] = "sequence"
        else:
            validate_type(autopct, (str, Callable), "autopct")
            self.autopct = autopct
            self.multi_params_structure["autopct"] = "value"
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

    def set_wedgeprops(self, kwargs: Union[DictSequence, Dict]):
        if is_dict_sequence(kwargs):
            self.multi_params_structure["wedgeprops"] = "sequence"
        elif isinstance(kwargs, dict) or kwargs is None:
            self.multi_params_structure["wedgeprops"] = "value"
        else:
            raise ArgumentStructureError("Invalid wedgeprops")
        self.wedgeprops = kwargs
        return self

    def set_textprops(self, kwargs: Union[DictSequence, Dict]):
        if is_dict_sequence(kwargs):
            self.multi_params_structure["textprops"] = "sequence"
        elif isinstance(kwargs, dict) or kwargs is None:
            self.multi_params_structure["textprops"] = "value"
        else:
            raise ArgumentStructureError("Invalid textprops")
        self.textprops = kwargs
        return self

    def set_center(self, center: Union[NumericTupleSequence, NumericTuple]):
        return self.set_numeric_tuple_sequence(center, 2, "center")

    def set_rotatelabels(self, rotatelabels: Union[BoolSequence, bool]):
        return self.set_bool_sequence(rotatelabels, "rotatelabels")

    def set_normalize(self, normalize: Union[BoolSequence, bool]):
        return self.set_bool_sequence(normalize, "normalize")

    def _create_pie_kwargs(self, n_sequence: int):
        pie_kwargs = {
            "x": self.x_data[n_sequence],
            "explode": self.get_sequences_param("explode", n_sequence),
            "labels": self.get_sequences_param("labels", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "colors": self.get_sequences_param("color", n_sequence),
            "autopct": self.get_sequences_param("autopct", n_sequence),
            "pctdistance": self.get_sequences_param("pctdistance", n_sequence),
            "labeldistance": self.get_sequences_param("labeldistance", n_sequence),
            "shadow": self.get_sequences_param("shadow", n_sequence),
            "startangle": self.get_sequences_param("startangle", n_sequence),
            "radius": self.get_sequences_param("radius", n_sequence),
            "counterclock": self.get_sequences_param("counterclock", n_sequence),
            "wedgeprops": self.get_sequences_param("wedgeprops", n_sequence),
            "textprops": self.get_sequences_param("textprops", n_sequence),
            "center": self.get_sequences_param("center", n_sequence),
            "frame": self.get_sequences_param("frame", n_sequence),
            "rotatelabels": self.get_sequences_param("rotatelabels", n_sequence),
            "normalize": self.get_sequences_param("normalize", n_sequence),
        }
        return pie_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            pie_kwargs = self._create_pie_kwargs(n_sequence)
            pie_kwargs = {k: v for k, v in pie_kwargs.items() if v is not None}
            try:
                self.ax.pie(**pie_kwargs)
                self.ax.axis("equal")
            except Exception as e:
                raise PiePlotterException(f"Error while creating pie plot: {e}")
