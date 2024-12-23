import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from numpy import ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
)
from mitools.visuals.plots.matplotlib_typing import (
    Color,
    ColorSequence,
    ColorSequences,
    NumericSequence,
    NumericSequences,
    NumericType,
    StrSequence,
)
from mitools.visuals.plots.plot_params import PlotParams
from mitools.visuals.plots.setter import Setter
from mitools.visuals.plots.validations import (
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_str_sequence,
    validate_consistent_len,
    validate_numeric_sequences,
    validate_same_length,
    validate_sequence_length,
)


class PlotterException(Exception):
    pass


class Plotter(PlotParams, Setter, ABC):
    def __init__(
        self,
        x_data: Union[NumericSequence, NumericSequences],
        y_data: Union[NumericSequence, NumericSequences, None],
        ax: Axes = None,
        **kwargs,
    ):
        self.x_data = self._validate_data(x_data, "x_data")
        self.y_data = self._validate_data(y_data, "y_data")
        validate_same_length(
            self.x_data[0],
            self.y_data[0] if self.y_data is not None else self.x_data[0],
            "x_data",
            "y_data",
        )
        self._n_sequences = len(self.x_data)
        self._multi_data = self._n_sequences > 1
        self._data_size = len(self.x_data[0])
        # Specific Parameters that are based on the number of data sequences
        self._multi_data_params = {
            "color": {
                "default": None,
                "type": Union[ColorSequences, ColorSequence, Color],
            },
            "alpha": {
                "default": 1.0,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
            "label": {"default": None, "type": Union[StrSequence, str]},
            "zorder": {
                "default": None,
                "type": Union[NumericSequences, NumericSequence, NumericType],
            },
        }
        self._multi_params_structure = {}
        super().__init__(ax=ax, **kwargs)
        self._init_params.update(self._multi_data_params)
        self._set_init_params(**kwargs)

    @property
    def data_size(self) -> int:
        return self._data_size

    @property
    def n_sequences(self) -> int:
        return self._n_sequences

    @property
    def multi_data(self) -> bool:
        return self._multi_data

    @property
    def multi_params_structure(self) -> dict:
        return self._multi_params_structure

    def _validate_data(
        self,
        data: Union[NumericSequence, NumericSequences, None],
        name: Literal["x_data", "y_data"],
    ) -> NumericSequences:
        if name == "y_data" and data is None:
            return data
        if is_numeric_sequence(data):
            data = [data]
        validate_numeric_sequences(data, name)
        validate_consistent_len(data, name)
        return np.asarray(data)

    def set_color(self, color: Union[ColorSequences, ColorSequence, Color]):
        return self.set_color_sequences(color, param_name="color")

    def set_alpha(self, alpha: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(
            alpha, param_name="alpha", min_value=0, max_value=1
        )

    def set_label(self, labels: Union[Sequence[str], str]):
        if self._multi_data and is_str_sequence(labels):
            validate_sequence_length(labels, self._n_sequences, "labels")
            self.label = labels
            self._multi_params_structure["label"] = "sequence"
            return self
        if isinstance(labels, str):
            self.label = labels
            self._multi_params_structure["label"] = "value"
            return self
        raise ArgumentStructureError(
            "Invalid label, must be a string or sequence of strings."
        )

    def set_zorder(self, zorder: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(zorder, param_name="zorder")

    def get_sequences_param(self, param_name: str, n_sequence: int):
        param_value = getattr(self, param_name)
        if self._multi_data:
            param_structure = self._multi_params_structure.get(param_name)
            if param_structure in ["sequences", "sequence"]:
                return param_value[n_sequence]
            elif param_structure == "value":
                return param_value
        return param_value

    @abstractmethod
    def _create_plot(self):
        raise NotImplementedError

    def draw(self, show: bool = False, clear: bool = False):
        self._prepare_draw(clear=clear)
        try:
            self._create_plot()
        except Exception as e:
            raise PlotterException(f"Error while creating plot: {e}")
        self._apply_common_properties()
        return self._finalize_draw(show)

    def save_plot(
        self,
        file_path: Path,
        dpi: int = 300,
        bbox_inches: str = "tight",
        draw: bool = False,
    ):
        if self.figure or draw:
            if self.figure is None and draw:
                self.draw()
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise PlotterException(f"Error while saving scatter plot: {e}")
        else:
            raise PlotterException("Plot not drawn yet. Call draw() before saving.")
        return self

    def _to_serializable(self, value: Any) -> Any:
        if value is None:
            return None
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, ndarray):
            return value.tolist()
        elif isinstance(value, Series):
            return value.to_list()
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, Colormap):
            return value.name
        elif isinstance(value, Normalize):
            return value.__class__.__name__.lower()
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, MarkerStyle):
            marker = dict(
                marker=value.get_marker(),
                fillstyle=value.get_fillstyle(),
                capstyle=value.get_capstyle(),
                joinstyle=value.get_joinstyle(),
            )
            return marker

        return value

    def save_plotter(
        self, file_path: Union[str, Path], data: bool = True, return_json: bool = False
    ) -> None:
        init_params = {}
        for param, config in self._init_params.items():
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        if data:
            init_params["x_data"] = self._to_serializable(self.x_data)
            init_params["y_data"] = self._to_serializable(self.y_data)
        if return_json:
            return init_params
        with open(file_path, "w") as f:
            json.dump(init_params, f, indent=4)

    @classmethod
    def _convert_list_to_tuple(
        cls,
        value: Union[NumericSequences, NumericSequence, None],
        expected_size: Union[Tuple[NumericType], NumericType] = None,
    ) -> Any:
        if value is None:
            return None
        if expected_size is not None and is_numeric(expected_size):
            expected_size = (expected_size,)
        if is_numeric_sequences(value):
            if expected_size is not None:
                if all(len(item) in expected_size for item in value):
                    return [tuple(val) for val in value]
        elif is_numeric_sequence(value):
            if expected_size is not None:
                if len(value) in expected_size:
                    return tuple(value)
        return value

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Plotter":
        with open(file_path, "r") as f:
            params = json.load(f)
        x_data = params.pop("x_data") if "x_data" in params else None
        y_data = params.pop("y_data") if "y_data" in params else None
        # Convert lists to tuples where needed
        if "xlim" in params:
            params["xlim"] = cls._convert_list_to_tuple(params["xlim"], 2)
        if "ylim" in params:
            params["ylim"] = cls._convert_list_to_tuple(params["ylim"], 2)
        if "figsize" in params:
            params["figsize"] = cls._convert_list_to_tuple(params["figsize"], 2)
        if "center" in params:
            params["center"] = cls._convert_list_to_tuple(params["center"], 2)
        if "range" in params:
            params["range"] = cls._convert_list_to_tuple(params["range"], 2)
        if "color" in params:
            params["color"] = cls._convert_list_to_tuple(params["color"], (3, 4))
        if "whis" in params:
            params["whis"] = cls._convert_list_to_tuple(params["whis"], 2)
        return cls(x_data=x_data, y_data=y_data, **params)
