from typing import Any, Sequence, Tuple, Type, TypeVar, Union

from numpy import integer, ndarray
from pandas import Series

from mitools.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)

T = TypeVar("T")

NUMERIC_TYPES = (float, int, integer)
SEQUENCE_TYPES = (list, tuple, ndarray, Series)


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


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, ndarray, Series))


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
