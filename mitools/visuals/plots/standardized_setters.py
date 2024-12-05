from typing import Any, Dict, Literal, Union

import numpy as np

from mitools.exceptions import ArgumentStructureError
from mitools.visuals.plots import Plotter
from mitools.visuals.plots.matplotlib_typing import (
    Color,
    ColorSequence,
    ColorSequences,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericType,
)
from mitools.visuals.plots.validations import (
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    validate_color,
    validate_consistent_len,
    validate_sequence_length,
    validate_subsequences_length,
)


def set_color_sequences(
    plotter: Plotter,
    colors: Union[ColorSequences, ColorSequence, Color],
    param_name: str,
) -> Any:
    if plotter._multi_data:
        if is_color_sequences(colors):
            validate_sequence_length(colors, plotter._n_sequences, param_name)
            validate_subsequences_length(colors, [1, plotter.data_size], param_name)
            setattr(plotter, param_name, colors)
            plotter._multi_params_structure[param_name] = "sequences"
            return plotter
        elif is_color_sequence(colors):
            validate_sequence_length(colors, plotter._n_sequences, param_name)
            setattr(plotter, param_name, colors)
            plotter._multi_params_structure[param_name] = "sequence"
            return plotter
        elif is_color(colors):
            setattr(plotter, param_name, colors)
            plotter._multi_params_structure[param_name] = "value"
            return plotter
    else:
        if is_color_sequence(colors):
            validate_sequence_length(colors, plotter.data_size, param_name)
            setattr(plotter, param_name, colors)
            plotter._multi_params_structure[param_name] = "sequence"
            return plotter
        elif is_color(colors):
            setattr(plotter, param_name, colors)
            plotter._multi_params_structure[param_name] = "value"
            return plotter
    raise ArgumentStructureError(
        f"Invalid {param_name}, must be a color, sequence of colors, or sequences of colors."
    )


def set_numeric_sequences(
    plotter: Plotter,
    sequences: Union[NumericSequences, NumericSequence, NumericType],
    param_name: str,
):
    if plotter._multi_data:
        if is_numeric_sequences(sequences):
            validate_sequence_length(sequences, plotter._n_sequences, param_name)
            validate_subsequences_length(sequences, [1, plotter.data_size], param_name)
            setattr(plotter, param_name, np.asarray(sequences))
            plotter._multi_params_structure[param_name] = "sequences"
            return plotter
        elif is_numeric_sequence(sequences):
            validate_sequence_length(sequences, plotter._n_sequences, param_name)
            setattr(plotter, param_name, sequences)
            plotter._multi_params_structure[param_name] = "sequence"
            return plotter
        elif is_numeric(sequences):
            setattr(plotter, param_name, sequences)
            plotter._multi_params_structure[param_name] = "value"
            return plotter
    else:
        if is_numeric_sequence(sequences):
            validate_sequence_length(sequences, plotter.data_size, param_name)
            setattr(plotter, param_name, sequences)
            plotter._multi_params_structure[param_name] = "sequence"
            return plotter
        elif is_numeric(sequences):
            setattr(plotter, param_name, sequences)
            plotter._multi_params_structure[param_name] = "value"
            return plotter
    raise ArgumentStructureError(
        f"Invalid {param_name}, must be a numeric value, numeric sequences, or sequence of numeric sequences."
    )
