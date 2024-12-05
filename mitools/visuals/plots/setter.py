from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np

from mitools.exceptions import ArgumentStructureError
from mitools.visuals.plots.matplotlib_typing import (
    Bins,
    BinsSequence,
    Cmap,
    CmapSequence,
    Color,
    ColorSequence,
    ColorSequences,
    DictSequence,
    EdgeColor,
    EdgeColorSequence,
    EdgeColorSequences,
    LiteralSequence,
    LiteralSequences,
    Marker,
    MarkerSequence,
    MarkerSequences,
    Norm,
    NormSequence,
    NumericSequence,
    NumericSequences,
    NumericTuple,
    NumericTupleSequence,
    NumericType,
    StrSequence,
    StrSequences,
)
from mitools.visuals.plots.validations import (
    is_bins,
    is_bins_sequence,
    is_bool,
    is_bool_sequence,
    is_color,
    is_color_sequence,
    is_color_sequences,
    is_colormap,
    is_colormap_sequence,
    is_dict,
    is_dict_sequence,
    is_edgecolor,
    is_edgecolor_sequence,
    is_edgecolor_sequences,
    is_literal,
    is_literal_sequence,
    is_literal_sequences,
    is_marker,
    is_marker_sequence,
    is_marker_sequences,
    is_normalization,
    is_normalization_sequence,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_numeric_tuple,
    is_numeric_tuple_sequence,
    is_str,
    is_str_sequence,
    is_str_sequences,
    validate_sequence_length,
    validate_sequence_values_in_range,
    validate_sequences_values_in_range,
    validate_subsequences_length,
    validate_value_in_range,
)


class Setter(ABC):
    @property
    @abstractmethod
    def data_size(self) -> int:
        pass

    @property
    @abstractmethod
    def n_sequences(self) -> int:
        pass

    @property
    @abstractmethod
    def multi_data(self) -> bool:
        pass

    @property
    @abstractmethod
    def multi_params_structure(self) -> dict:
        pass

    def set_color_sequences(
        self,
        colors: Union[ColorSequences, ColorSequence, Color],
        param_name: str,
    ) -> Any:
        if self.multi_data:
            if is_color_sequences(colors):
                validate_sequence_length(colors, self.n_sequences, param_name)
                validate_subsequences_length(colors, [1, self.data_size], param_name)
                setattr(self, param_name, colors)
                self.multi_params_structure[param_name] = "sequences"
                return self
            elif is_color_sequence(colors):
                validate_sequence_length(colors, self.n_sequences, param_name)
                setattr(self, param_name, colors)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_color(colors):
                setattr(self, param_name, colors)
                self.multi_params_structure[param_name] = "value"
                return self
        else:
            if is_color_sequence(colors):
                validate_sequence_length(colors, self.data_size, param_name)
                setattr(self, param_name, colors)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_color(colors):
                setattr(self, param_name, colors)
                self.multi_params_structure[param_name] = "value"
                return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a color, sequence of colors, or sequences of colors."
        )

    def set_color_sequence(self, colors: Union[ColorSequence, Color], param_name: str):
        if self.multi_data and is_color_sequence(colors):
            validate_sequence_length(colors, self.n_sequences, param_name)
            setattr(self, param_name, colors)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_color(colors):
            setattr(self, param_name, colors)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a color or sequence of colors."
        )

    def set_numeric_sequences(
        self,
        sequences: Union[NumericSequences, NumericSequence, NumericType],
        param_name: str,
        min_value: NumericType = None,
        max_value: NumericType = None,
    ):
        if self.multi_data:
            if is_numeric_sequences(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                expanded_sequences = [
                    np.repeat(seq, self.data_size) if len(seq) == 1 else seq
                    for seq in sequences
                ]
                validate_subsequences_length(
                    expanded_sequences, self.data_size, param_name
                )
                validate_sequences_values_in_range(
                    expanded_sequences, min_value, max_value, param_name
                )
                setattr(self, param_name, np.asarray(expanded_sequences))
                self.multi_params_structure[param_name] = "sequences"
                return self
            elif is_numeric_sequence(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_sequence_values_in_range(
                    sequences, min_value, max_value, param_name
                )
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_numeric(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        else:
            if is_numeric_sequence(sequences):
                validate_sequence_length(sequences, self.data_size, param_name)
                validate_sequence_values_in_range(
                    sequences, min_value, max_value, param_name
                )
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_numeric(sequences):
                validate_value_in_range(sequences, min_value, max_value, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a numeric value, numeric sequences, or sequence of numeric sequences."
        )

    def set_numeric_sequence(
        self,
        sequence: Union[NumericSequence, NumericType],
        param_name: str,
        min_value: NumericType = None,
        max_value: NumericType = None,
    ):
        if self.multi_data and is_numeric_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            validate_sequence_values_in_range(
                sequence, min_value, max_value, param_name
            )
            setattr(self, param_name, np.asarray(sequence))
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_numeric(sequence):
            validate_value_in_range(sequence, min_value, max_value, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a numeric value or sequence of numbers."
        )

    def set_literal_sequences(
        self,
        sequences: Union[LiteralSequences, LiteralSequence, Literal["options"]],
        options: Sequence[str],
        param_name: str,
    ):
        if self.multi_data:
            if is_literal_sequences(sequences, options):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequences"
                return self
            elif is_literal_sequence(sequences, options):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_literal(sequences, options):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        else:
            if is_literal_sequence(sequences, options):
                validate_sequence_length(sequences, self.data_size, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_literal(sequences, options):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a literal or sequence of literals."
        )

    def set_literal_sequence(
        self,
        sequence: Union[LiteralSequence, Literal["options"]],
        options: Sequence[str],
        param_name: str,
    ):
        if self.multi_data and is_literal_sequence(sequence, options):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_literal(sequence, options):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a literal or sequence of literals."
        )

    def set_marker_sequences(
        self,
        sequences: Union[MarkerSequences, MarkerSequence, Marker],
        param_name: str,
    ):
        if self.multi_data:
            if is_marker_sequences(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequences"
                return self
            elif is_marker_sequence(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_marker(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        else:
            if is_marker_sequence(sequences):
                validate_sequence_length(sequences, self.data_size, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_marker(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a marker, sequence of markers, or sequences of markers."
        )

    def set_marker_sequence(
        self, sequence: Union[MarkerSequence, Marker], param_name: str
    ):
        if self.multi_data and is_marker_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_marker(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a marker or sequence of markers."
        )

    def set_edgecolor_sequences(
        self,
        sequences: Union[EdgeColorSequences, EdgeColorSequence, EdgeColor],
        param_name: str,
    ):
        if self.multi_data:
            if is_edgecolor_sequences(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequences"
                return self
            elif is_edgecolor_sequence(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_edgecolor(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        else:
            if is_edgecolor_sequence(sequences):
                validate_sequence_length(sequences, self.data_size, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_edgecolor(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be an edgecolor, sequence of edgecolors, or sequences of edgecolors."
        )

    def set_edgecolor_sequence(
        self, sequence: Union[EdgeColorSequence, EdgeColor], param_name: str
    ):
        if self.multi_data and is_edgecolor_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_edgecolor(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be an edgecolor or sequence of edgecolors."
        )

    def set_colormap_sequence(
        self, sequence: Union[CmapSequence, Cmap], param_name: str
    ):
        if self.multi_data and is_colormap_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_colormap(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a colormap, sequence of colormaps, or sequences of colormaps."
        )

    def set_norm_sequence(self, sequence: Union[NormSequence, Norm], param_name: str):
        if self.multi_data and is_normalization_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_normalization(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a normalization, sequence of normalizations, or sequences of normalizations."
        )

    def set_str_sequences(
        self, sequences: Union[StrSequences, StrSequence], param_name: str
    ):
        if self.multi_data:
            if is_str_sequences(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequences"
                return self
            elif is_str_sequence(sequences):
                validate_sequence_length(sequences, self.n_sequences, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_str(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        else:
            if is_str_sequence(sequences):
                validate_sequence_length(sequences, self.data_size, param_name)
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "sequence"
                return self
            elif is_str(sequences):
                setattr(self, param_name, sequences)
                self.multi_params_structure[param_name] = "value"
                return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a string, sequence of strings, or sequences of strings."
        )

    def set_str_sequence(self, sequence: Union[StrSequence, str], param_name: str):
        if self.multi_data and is_str_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_str(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a string or sequence of strings."
        )

    def set_numeric_tuple_sequence(
        self,
        sequence: Union[NumericTupleSequence, NumericTuple],
        sizes: Sequence[int],
        param_name: str,
    ):
        if self.multi_data and is_numeric_tuple_sequence(sequence, sizes):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_numeric_tuple(sequence, sizes):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a numeric tuple, sequence of numeric tuples, or sequences of numeric tuples."
        )

    def set_bins_sequence(self, sequence: Union[BinsSequence, Bins], param_name: str):
        if self.multi_data and is_bins_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_bins(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a bin, sequence of bins, or sequences of bins."
        )

    def set_bool_sequence(self, sequence: Union[Sequence[bool], bool], param_name: str):
        if self.multi_data and is_bool_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_bool(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a boolean or sequence of booleans."
        )

    def set_dict_sequence(self, sequence: Union[DictSequence, Dict], param_name: str):
        if self.multi_data and is_dict_sequence(sequence):
            validate_sequence_length(sequence, self.n_sequences, param_name)
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "sequence"
            return self
        elif is_dict(sequence):
            setattr(self, param_name, sequence)
            self.multi_params_structure[param_name] = "value"
            return self
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a dictionary or sequence of dictionaries."
        )
