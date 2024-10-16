from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import chardet
import numpy as np
import pandas as pd
from numba import jit
from pandas import DataFrame

from mitools.exceptions import ArgumentValueError


def all_can_be_ints(items: Sequence) -> bool:
    try:
        return all(int(item) is not None for item in items)
    except (ValueError, TypeError):
        return False


def get_file_encoding(file: PathLike, fallback: str = "utf-8") -> str:
    try:
        with open(file, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
        encoding = result.get("encoding")
        confidence = result.get("confidence", 0.0)
        if not encoding or confidence < 0.8:
            return fallback
        if encoding.lower() == "ascii":
            return "utf-8"
        return encoding
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file}' was not found.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file '{file}': {e}")


def exports_data_to_matrix(
    dataframe: DataFrame,
    origin_col: str,
    products_cols: List[str],
    value_col: str,
    products_codes: DataFrame,
) -> DataFrame:
    if dataframe.empty:
        raise ArgumentValueError("Dataframe must not be empty")
    required_columns = {origin_col, value_col}.union(products_cols)
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Dataframe is missing required columns: {missing_columns}"
        )
    exports = dataframe[[origin_col, *products_cols, value_col]].reset_index(drop=True)
    origins = exports[origin_col].unique()
    exports = exports.set_index([origin_col, *products_cols])
    initial_total_value = exports[value_col].sum()
    new_index = pd.MultiIndex.from_product(
        [origins, *[products_codes[col].unique() for col in products_cols]],
        names=[origin_col, *products_cols],
    )
    exports = exports.reindex(new_index.drop_duplicates(), fill_value=0)
    reindexed_total_value = exports[value_col].sum()
    if not pd.isna(initial_total_value) and not pd.isna(reindexed_total_value):
        if not np.isclose(initial_total_value, reindexed_total_value):
            raise ArgumentValueError(
                "Total export values are inconsistent before and after reindexing"
            )
    exports_matrix = exports.unstack(level=products_cols)
    exports_matrix.columns = exports_matrix.columns.droplevel(0)
    all_origins = set(dataframe[origin_col].unique())
    matrix_origins = set(exports_matrix.index)
    if all_origins != matrix_origins:
        raise ArgumentValueError("Mismatch in origins between input and output")
    all_products = set(
        map(tuple, products_codes[products_cols].itertuples(index=False))
    )
    matrix_products = set(map(tuple, exports_matrix.columns))
    if all_products != matrix_products:
        raise ArgumentValueError("Mismatch in product codes between input and output")
    return exports_matrix


def calculate_exports_matrix_rca(exports_matrix: DataFrame) -> DataFrame:
    if not np.issubdtype(exports_matrix.values.dtype, np.number):
        raise ArgumentValueError("The exports matrix must contain only numeric values")
    xcp = exports_matrix.copy()
    x = exports_matrix.sum().sum()
    xc = exports_matrix.sum(axis=1)
    xp = exports_matrix.sum(axis=0)

    rca_matrix = xcp * x
    rca_matrix = rca_matrix.div(xp, axis=1)
    rca_matrix = rca_matrix.div(xc, axis=0)

    rca_matrix = rca_matrix.fillna(0.0)

    return rca_matrix


def mask_matrix(matrix: DataFrame, threshold: Union[float, int]) -> DataFrame:
    if not np.issubdtype(matrix.values.dtype, np.number):
        raise ArgumentValueError("The exports matrix must contain only numeric values")
    if not isinstance(threshold, (float, int)):
        raise ArgumentValueError("Threshold must be a float or an integer")
    masked_matrix = matrix.copy()
    masked_matrix[masked_matrix < threshold] = 0.0
    masked_matrix[masked_matrix >= threshold] = 1.0
    return masked_matrix


def calculate_proximity_matrix(dataframe: DataFrame, symmetric: Optional[bool] = True):
    if dataframe.isna().any().any():
        raise ArgumentValueError("The dataframe must not contain non-numeric values!")
    ubiquity = dataframe.sum(axis=0)
    proximity_matrix = dataframe.T @ dataframe
    proximity_matrix = proximity_matrix / ubiquity.values
    np.fill_diagonal(proximity_matrix.values, 0)
    proximity_matrix = proximity_matrix.fillna(0.0)
    if symmetric:
        proximity_matrix = np.minimum(proximity_matrix, proximity_matrix.T)
    return proximity_matrix


def calculate_relatedness_matrix(
    proximity_matrix: DataFrame,
    rca_matrix: DataFrame,
    relatedness_col: Optional[str] = "relatedness",
) -> DataFrame:
    if proximity_matrix.isna().any().any():
        raise ArgumentValueError(
            "The 'proximity_matrix' must not contain non-numeric values!"
        )
    if rca_matrix.isna().any().any():
        raise ArgumentValueError(
            "The 'rca_matrix' must not contain non-numeric values!"
        )
    row_sums = proximity_matrix.sum(axis=1)
    try:
        wcp = rca_matrix.dot(proximity_matrix)
    except ValueError as e:
        raise ArgumentValueError(
            f"Mismatched indexes in 'rca_matrix' and 'proximity_matrix': {e}"
        )
    wcp.columns = row_sums.index
    wcp = wcp / row_sums
    wcp = wcp.unstack().to_frame()
    wcp.columns = [relatedness_col]
    return wcp


def create_time_id(time_values: Union[str, int, Sequence]) -> str:
    if isinstance(time_values, (str, int)):
        return str(time_values)
    else:
        if not all_can_be_ints(time_values):
            raise ValueError("Some time values provided can't be converted to int")
        if len(time_values) == 1:
            return str(time_values[0])
        time_values = sorted(int(v) for v in time_values)
        return f"{str(time_values[0])}{str(time_values[-1])[-2:]}"


def create_data_id(id: str, time: Union[str, int, Sequence]) -> str:
    time = create_time_id(time)
    return f"{id}_{time}"


def create_data_name(data_id, tag):
    return f"{data_id}_{tag}"


@jit(nopython=True)
def fast_calculate_economic_complexity(
    rca_matrix: np.ndarray,
    diversity: np.ndarray,
    ubiquity: np.ndarray,
    standardize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    print("Fast Calculation of ECI/PCI!")
    M1 = np.divide(rca_matrix.T, diversity).T
    M1 = np.nan_to_num(M1)
    M2 = np.divide(rca_matrix, ubiquity)
    M2 = np.nan_to_num(M2)
    M2_t = M2.T.copy()

    # Mcc = M1.dot(M2_t)
    Mpp = M2_t.dot(M1)

    eigen_values, eigen_vectors = np.linalg.eig(Mpp)
    eigen_vectors = np.real(eigen_vectors)

    eigen_vector_index = eigen_values.argsort()[-2]
    kp = eigen_vectors[:, eigen_vector_index]
    kc = M1.dot(kp)

    signature = np.sign(np.corrcoef(diversity, kc)[0, 1])
    eci_t = signature * kc
    pci_t = signature * kp

    if standardize:
        pci_t = (pci_t - pci_t.mean()) / pci_t.std()
        eci_t = (eci_t - eci_t.mean()) / eci_t.std()

    return eci_t, pci_t


def calculate_economic_complexity(
    rca_matrix: DataFrame, year: int, standardize=True
) -> Tuple[DataFrame, DataFrame]:
    diversity = rca_matrix.sum(axis=1)
    ubiquity = rca_matrix.sum(axis=0)

    # try:
    #     eci_t, pci_t = fast_calculate_economic_complexity(
    #         rca_matrix.to_numpy(),
    #         diversity.to_numpy(),
    #         ubiquity.to_numpy(),
    #         standardize,
    #     )
    # except Exception as e:

    M1 = np.divide(rca_matrix.T, diversity).T
    M1 = np.nan_to_num(M1)
    M2 = np.divide(rca_matrix, ubiquity)
    M2 = np.nan_to_num(M2)
    M2_t = M2.T.copy()

    # Mcc = M1.dot(M2_t)
    Mpp = M2_t.dot(M1)

    eigen_values, eigen_vectors = np.linalg.eig(Mpp)
    eigen_vectors = np.real(eigen_vectors)

    eigen_vector_index = eigen_values.argsort()[-2]
    kp = eigen_vectors[:, eigen_vector_index]
    kc = M1.dot(kp)

    signature = np.sign(np.corrcoef(diversity, kc)[0, 1])
    eci_t = signature * kc
    pci_t = signature * kp

    if standardize:
        pci_t = (pci_t - pci_t.mean()) / pci_t.std()
        eci_t = (eci_t - eci_t.mean()) / eci_t.std()

    eci_df = pd.DataFrame(eci_t, columns=["ECI"], index=rca_matrix.index).sort_values(
        by="ECI", ascending=False
    )
    pci_df = pd.DataFrame(pci_t, columns=["PCI"], index=rca_matrix.columns).sort_values(
        by="PCI", ascending=False
    )

    eci_df["Year"] = year
    eci_df = eci_df.set_index("Year", append=True)
    pci_df["Year"] = year
    pci_df = pci_df.set_index("Year", append=True)

    return eci_df, pci_df


def store_dataframe_sequence(
    dataframes: Dict[Union[str, int], DataFrame], name: str, data_dir: Path
) -> None:
    sequence_dir = data_dir / name
    sequence_dir.mkdir(exist_ok=True, parents=True)
    for seq_val, dataframe in dataframes.items():
        seq_val_name = f"{name}_{seq_val}".replace(" ", "")
        dataframe.to_parquet(sequence_dir / f"{seq_val_name}.parquet")


def load_dataframe_sequence(
    data_dir: Path, name: str, sequence_values: Optional[List] = None
) -> Dict[Union[str, int], DataFrame]:
    sequence_dir = data_dir / name
    sequence_files = sequence_dir.glob("*.parquet")
    # TODO: So far it only handles ints as sequence values
    return {
        int(file.stem.split("_")[-1]): pd.read_parquet(file)
        for file in sequence_files
        if sequence_values is None or int(file.stem.split("_")[-1]) in sequence_values
    }


def check_if_sequence(
    data_dir: Path, name: str, sequence_values: Optional[List] = None
) -> bool:
    sequence_dir = data_dir / name
    if sequence_values:
        sequence_files = sequence_dir.glob("*.parquet")
        sequence_files = [int(file.stem.split("_")[-1]) for file in sequence_files]
        return set(sequence_values) == set(sequence_files)
    return sequence_dir.exists()
