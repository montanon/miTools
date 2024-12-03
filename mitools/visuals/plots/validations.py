import re
from typing import Any, Sequence, Tuple, Type, TypeVar, Union

from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitools.visuals.plots.matplotlib_typing import NumericSequences, _colors

T = TypeVar("T")

NUMERIC_TYPES = (float, int, integer)
SEQUENCE_TYPES = (list, tuple, ndarray, Series)


def is_numeric(value: Any, name: str) -> bool:
    try:
        validate_numeric(value, name)
        return True
    except ArgumentTypeError:
        return False


def validate_numeric(value: Any, name: str) -> None:
    validate_type(value, NUMERIC_TYPES, name)


def is_numeric_sequence(sequence: Any, name: str) -> bool:
    try:
        validate_numeric_sequence(sequence, name)
        return True
    except ArgumentTypeError:
        return False


def validate_numeric_sequence(sequence: Sequence, name: str) -> None:
    validate_sequence_type(sequence, NUMERIC_TYPES, name)


def is_numeric_sequences(sequences: Any, name: str) -> bool:
    try:
        validate_numeric_sequences(sequences, name)
        return True
    except ArgumentTypeError:
        return False


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
    first_len = len(sequences[0])
    for i, sequence in enumerate(sequences[1:], 1):
        if len(sequence) != first_len:
            raise ArgumentStructureError(
                f"All sequences in '{name}' must have the same length. "
                f"Sequence at index 0 has length {first_len}, but sequence at "
                f"index {i} has length {len(sequence)}."
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
    return isinstance(value, str) and value in _colors


def is_color(value: Any) -> bool:
    return is_color_tuple(value) or is_color_hex(value) or is_color_str(value)


def validate_color(value: Any) -> None:
    if not is_color(value):
        raise ArgumentTypeError(f"Invalid color: {value}")


def is_color_sequence(value: Any) -> bool:
    return is_sequence(value) and all(is_color(item) for item in value)


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
    if not isinstance(value, expected_type):
        raise ArgumentTypeError(
            f"'{param_name}' must be of type {expected_type}, got {type(value)}"
        )


def validate_sequence_type(
    sequence: Sequence, item_type: Union[Type, Tuple[Type, ...]], param_name: str
) -> None:
    if not all(isinstance(item, item_type) for item in sequence):
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
