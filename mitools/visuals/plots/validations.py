import re
from pathlib import Path
from typing import Any, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import (
    BINS,
    CMAPS,
    COLORS,
    MARKERS,
    MARKERS_FILLSTYLES,
    NORMALIZATIONS,
    NumericSequences,
    NumericType,
)

T = TypeVar("T")

NUMERIC_TYPES = (float, int, integer)
SEQUENCE_TYPES = (list, tuple, ndarray, Series)


def is_dict_sequences(sequences: Sequence[Sequence[Any]]) -> bool:
    return is_sequences(sequences) and all(is_dict_sequence(seq) for seq in sequences)


def is_dict_sequence(sequence: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(
        isinstance(item, dict) or item is None for item in sequence
    )


def is_literal(value: Any, options: Sequence[Any]) -> bool:
    return (isinstance(value, str) and value in options) or value is None


def validate_literal(value: Any, options: Sequence[Any]) -> None:
    if not is_literal(value, options):
        raise ArgumentTypeError(f"Invalid literal: {value}")


def is_literal_sequence(sequence: Sequence[Any], options: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(is_literal(val, options) for val in sequence)


def validate_literal_sequence(sequence: Sequence[Any], options: Sequence[Any]) -> None:
    if not is_literal_sequence(sequence, options):
        raise ArgumentTypeError(f"Invalid literal sequence: {sequence}")


def is_literal_sequences(
    sequences: Sequence[Sequence[Any]], options: Sequence[Any]
) -> bool:
    return is_sequence(sequences) and all(
        is_literal_sequence(seq, options) for seq in sequences
    )


def is_str_sequences(sequences: Sequence[Sequence[Any]]) -> bool:
    return is_sequence(sequences) and all(is_str_sequence(seq) for seq in sequences)


def validate_same(value1: Any, value2: Any, param_name1: str, param_name2: str):
    if value1 != value2:
        raise ArgumentValueError(
            f"{param_name1}={value1} and {param_name2}={value2} must be the same."
        )


def is_numeric_tuple(value: Any, sizes: Union[Sequence[int], int] = None) -> bool:
    if isinstance(sizes, int):
        sizes = [sizes]
    return (
        isinstance(value, tuple)
        and all(isinstance(item, NUMERIC_TYPES) for item in value)
        and len(value) in sizes
        if sizes
        else True
    )


def validate_numeric_tuple(value: Any, sizes: Sequence[int]) -> None:
    if not is_numeric_tuple(value, sizes):
        raise ArgumentTypeError(f"Invalid numeric tuple: {value}")


def is_numeric_tuple_sequence(
    sequence: Sequence[Any], sizes: Sequence[int] = None
) -> bool:
    return (
        is_sequence(sequence)
        and not isinstance(sequence, tuple)
        and all(is_numeric_tuple(val, sizes) for val in sequence)
    )


def is_numeric_tuple_sequences(
    sequences: Sequence[Sequence[Any]], sizes: Sequence[int] = None
) -> bool:
    return is_sequence(sequences) and all(
        is_numeric_tuple_sequence(seq, sizes) for seq in sequences
    )


def validate_numeric_tuple_sequence(
    sequence: Sequence[Any], sizes: Sequence[int]
) -> None:
    if not is_numeric_tuple_sequence(sequence, sizes):
        raise ArgumentTypeError(f"Invalid numeric tuple sequence: {sequence}")


def is_bins_sequences(sequences: Sequence[Sequence[Any]]) -> bool:
    return is_sequence(sequences) and all(is_bins_sequence(seq) for seq in sequences)


def is_bins_sequence(sequence: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(is_bins(val) for val in sequence)


def validate_bins(bins: Any) -> None:
    if not is_bins(bins):
        raise ArgumentTypeError(f"Invalid bins: {bins}")


def is_bins(value: Any) -> bool:
    if isinstance(value, (int, str)):
        if isinstance(value, int):
            return is_value_in_range(value, 0, 1_000_000)
        if isinstance(value, str):
            return value in BINS
    return False


def is_normalization(value: Any) -> bool:
    return (isinstance(value, str) and value in NORMALIZATIONS) or isinstance(
        value, Normalize
    )


def is_normalization_sequence(sequence: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(is_normalization(val) for val in sequence)


def is_colormap(value: Any) -> bool:
    return (isinstance(value, str) and value in CMAPS) or isinstance(value, Colormap)


def is_colormap_sequence(sequence: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(is_colormap(val) for val in sequence)


def is_facecolor(value: Any) -> bool:
    return is_color(value)


def is_edgecolor(value: Any) -> bool:
    return value in ["face", "none", None] or is_color(value)


def validate_edgecolor(value: Any) -> None:
    if not is_edgecolor(value):
        raise ArgumentTypeError(f"Invalid edgecolor: {value}")


def is_edgecolor_sequence(sequence: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(is_edgecolor(val) for val in sequence)


def validate_edgecolor_sequence(sequence: Sequence[Any]) -> None:
    if not is_edgecolor_sequence(sequence):
        raise ArgumentTypeError(f"Invalid edgecolor sequence: {sequence}")


def is_edgecolor_sequences(sequences: Sequence[Sequence[Any]]) -> bool:
    return is_sequence(sequences) and all(
        is_edgecolor_sequence(seq) for seq in sequences
    )


def validate_edgecolor_sequences(sequences: Sequence[Sequence[Any]]) -> None:
    if not is_edgecolor_sequences(sequences):
        raise ArgumentTypeError(f"Invalid edgecolor sequences: {sequences}")


def validate_marker(value: Any):
    if not is_marker(value):
        raise ArgumentTypeError(f"Invalid marker: {value}")


def is_marker(value: Any) -> bool:
    if isinstance(value, (str, int, Path, MarkerStyle, dict)):
        if isinstance(value, str):
            return value in MARKERS
        if isinstance(value, int):
            return is_value_in_range(value, 0, 11)
        if isinstance(value, dict):
            valid_keys = all(
                key in ["marker", "fillstyle", "transform", "capstyle", "joinstyle"]
                for key in value
            )
            valid_marker = value["marker"] in MARKERS if "marker" in value else True
            valid_fillstyle = (
                value["fillstyle"] in MARKERS_FILLSTYLES
                if "fillstyle" in value
                else True
            )
            valid_transform = (
                isinstance(value["transform"], (str, Normalize))
                if "transform" in value
                else True
            )
            valid_capstyle = (
                value["capstyle"] in ["butt", "round", "projecting"]
                if "capstyle" in value
                else True
            )
            valid_joinstyle = (
                value["joinstyle"] in ["miter", "round", "bevel"]
                if "joinstyle" in value
                else True
            )
            return (
                valid_keys
                and valid_marker
                and valid_fillstyle
                and valid_transform
                and valid_capstyle
                and valid_joinstyle
            )
        return isinstance(value, (Path, MarkerStyle))
    elif value is None:
        return True
    return False


def is_marker_sequence(sequence: Sequence[Any]) -> bool:
    return is_sequence(sequence) and all(is_marker(val) for val in sequence)


def is_marker_sequences(sequences: Sequence[Sequence[Any]]) -> bool:
    return is_sequence(sequences) and all(
        is_marker_sequence(sequence) for sequence in sequences
    )


def is_value_in_range(value: Any, min_value: NumericType, max_value: NumericType):
    return (
        isinstance(value, NUMERIC_TYPES) and min_value <= value and value <= max_value
    )


def validate_value_in_range(
    value: Any, min_value: Union[float, None], max_value: Union[float, None], name: str
):
    max_value = max_value if max_value is not None else np.inf
    min_value = min_value if min_value is not None else -np.inf
    if not isinstance(value, (float, int)):
        raise ArgumentTypeError(f"'{name}'={value} must be a number.")
    if not min_value <= value <= max_value:
        raise ArgumentValueError(
            f"'{name}'={value} must be between {min_value} and {max_value}."
        )


def validate_sequence_values_in_range(
    sequence: Sequence,
    min_value: Union[float, None],
    max_value: Union[float, None],
    name: str,
):
    if min_value is None and max_value is None:
        return
    if not is_numeric_sequence(sequence):
        raise ArgumentTypeError(f"Invalid numeric sequence: {sequence}")
    for val in sequence:
        validate_value_in_range(val, min_value, max_value, name)


def validate_sequences_values_in_range(
    sequences: Sequence[Sequence],
    min_value: Union[float, None],
    max_value: Union[float, None],
    name: str,
):
    if min_value is None and max_value is None:
        return
    for sequence in sequences:
        validate_sequence_values_in_range(sequence, min_value, max_value, name)


def is_numeric(value: Any) -> bool:
    return isinstance(value, NUMERIC_TYPES)


def validate_numeric(value: Any, name: str) -> None:
    if not is_numeric(value):
        raise ArgumentTypeError(f"'{name}'={value} must be a number.")


def is_numeric_sequence(sequence: Any) -> bool:
    return is_sequence(sequence) and all(
        isinstance(item, NUMERIC_TYPES) or item is None for item in sequence
    )


def validate_numeric_sequence(sequence: Sequence, name: str) -> None:
    if not is_numeric_sequence(sequence):
        raise ArgumentTypeError(f"Invalid numeric sequence: {sequence}")


def is_numeric_sequences(sequences: Any) -> bool:
    return is_sequence(sequences) and all(
        is_numeric_sequence(item) for item in sequences
    )


def validate_numeric_sequences(sequences: Sequence[Sequence], name: str) -> None:
    validate_sequence_type(sequences, SEQUENCE_TYPES, name)
    for sequence in sequences:
        validate_numeric_sequence(sequence, name)


def is_consistent_len(sequences: NumericSequences, name: str) -> bool:
    try:
        validate_consistent_len(sequences, name)
        return True
    except ArgumentStructureError:
        return False


def validate_consistent_len(sequences: NumericSequences, name: str) -> None:
    max_len = max(len(sequence) for sequence in sequences)
    for i, sequence in enumerate(sequences):
        if len(sequence) not in [max_len, 1]:
            raise ArgumentStructureError(
                f"All sequences in '{name}' must have the same length or length 1. "
                f"Sequence at index {i} has length {len(sequence)}, but max length of sequences in '{name}' is {max_len}."
            )


def is_color_tuple(value: Any) -> bool:
    return (
        isinstance(value, SEQUENCE_TYPES)
        and len(value) in [3, 4]
        and all(isinstance(val, NUMERIC_TYPES) for val in value)
    )


def is_color_hex(value: Any) -> bool:
    return isinstance(value, str) and re.match(
        r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", value
    )


def is_color_str(value: Any) -> bool:
    return isinstance(value, str) and value in COLORS


def is_color(value: Any) -> bool:
    return (
        is_color_tuple(value)
        or is_color_hex(value)
        or is_color_str(value)
        or isinstance(value, NUMERIC_TYPES)
        or value is None
    )


def validate_color(value: Any) -> None:
    if not is_color(value):
        raise ArgumentTypeError(f"Invalid color: {value}")


def is_color_sequence(value: Any) -> bool:
    return (
        not isinstance(value, tuple)
        and is_sequence(value)
        and all(is_color(item) for item in value)
    )


def validate_color_sequence(value: Any) -> None:
    if not is_color_sequence(value):
        raise ArgumentTypeError(f"Invalid color sequence: {value}")


def is_color_sequences(sequences: Any) -> bool:
    return is_sequence(sequences) and all(is_color_sequence(item) for item in sequences)


def validate_color_sequences(sequences: Any) -> None:
    if not is_color_sequences(sequences):
        raise ArgumentTypeError(f"Invalid color sequences: {sequences}")


def is_str_sequence(sequence: Any) -> bool:
    return is_sequence(sequence) and all(isinstance(item, str) for item in sequence)


def validate_type(
    value: Any, expected_type: Union[Type, Tuple[Type, ...]], param_name: str
) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise ArgumentTypeError(
            f"'{param_name}' must be of type {expected_type}, got {type(value)}"
        )


def validate_sequence_type(
    sequence: Sequence, item_type: Union[Type, Tuple[Type, ...]], param_name: str
) -> None:
    if not all(isinstance(item, item_type) or item is None for item in sequence):
        raise ArgumentTypeError(
            f"All elements in '{param_name}' must be of type {item_type}."
        )


def validate_non_negative(value: Union[float, int, integer], param_name: str) -> None:
    if value < 0:
        raise ArgumentValueError(f"'{param_name}'={value} must be non-negative")


def validate_sequence_non_negative(sequence: Sequence, param_name: str) -> None:
    if any(item < 0 for item in sequence):
        raise ArgumentValueError(f"All elements in '{param_name}' must be non-negative")


def is_sequence(value: Any) -> bool:
    return isinstance(value, SEQUENCE_TYPES)


def is_sequences(sequences: Any) -> bool:
    return is_sequence(sequences) and all(is_sequence(item) for item in sequences)


def validate_sequence_length(
    sequence: Sequence, expected_length: Union[int, Tuple[int, ...]], param_name: str
) -> None:
    if isinstance(expected_length, int):
        expected_lengths = (expected_length,)
    else:
        expected_lengths = expected_length
    if len(sequence) not in expected_lengths:
        if len(expected_lengths) == 1:
            msg = f"'{param_name}' must be of length {expected_lengths[0]}, got {len(sequence)}"
        else:
            msg = f"'{param_name}' must be of one of lengths {expected_lengths}, got {len(sequence)}"
        raise ArgumentStructureError(msg)


def validate_subsequences_length(
    sequences: Sequence, expected_length: Union[int, Tuple[int, ...]], param_name: str
):
    for i, sequence in enumerate(sequences):
        validate_sequence_length(sequence, expected_length, f"{param_name}[{i}]")


def validate_same_length(
    sequence1: Sequence, sequence2: Sequence, param_name1: str, param_name2: str
) -> None:
    if len(sequence1) != len(sequence2):
        raise ArgumentStructureError(
            f"len({param_name1})={len(sequence1)} and len({param_name2})={len(sequence2)} must be the same."
        )


def validate_length(sequence: Sequence, expected_length: int, param_name: str) -> None:
    if len(sequence) != expected_length:
        raise ArgumentStructureError(
            f"len({param_name})={len(sequence)} and expected_length={expected_length} must be the same."
        )


def validate_value_in_options(
    value: Any, valid_options: Sequence[Any], param_name: str
) -> None:
    if value not in valid_options:
        raise ArgumentValueError(
            f"'{param_name}' must be one of {valid_options}, got {value}"
        )


def is_str(value: Any) -> bool:
    return isinstance(value, str)


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def validate_bool(value: Any, param_name: str) -> None:
    if not is_bool(value):
        raise ArgumentTypeError(f"'{param_name}' must be a boolean, got {type(value)}")


def is_bool_sequence(sequence: Any) -> bool:
    return is_sequence(sequence) and all(isinstance(item, bool) for item in sequence)


def validate_bool_sequence(sequence: Sequence, param_name: str) -> None:
    if not is_bool_sequence(sequence):
        raise ArgumentTypeError(f"Invalid bool sequence: {sequence}")


def is_dict(value: Any) -> bool:
    return isinstance(value, dict)
