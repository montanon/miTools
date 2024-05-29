from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import chardet
import numpy as np
import pandas as pd
from numba import jit
from pandas import DataFrame


def create_time_id(time_values: Union[str, int, Sequence]) -> str:
    if isinstance(time_values, (str, int)):
        return str(time_values)
    else:
        if not all_items_can_be_ints(time_values):
            raise ValueError("Some time values provided can't be converted to int")
        if len(time_values) == 1:
            return str(time_values[0])
        time_values = sorted(int(v) for v in time_values)
        return f"{str(time_values[0])}{str(time_values[-1])[-2:]}"


def all_items_can_be_ints(items: Sequence) -> bool:
    try:
        return all(isinstance(int(item), int) for item in items)
    except ValueError:
        return False


def create_data_id(id: str, time: Union[str, int, Sequence]) -> str:
    time = create_time_id(time)
    return f"{id}_{time}"


def create_data_name(data_id, tag):
    return f"{data_id}_{tag}"


def get_encoding(file):
    with open(file, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


def exports_data_to_matrix(
    dataframe: DataFrame,
    origin_col: str,
    products_cols: List[str],
    value_col: str,
    products_codes: DataFrame,
) -> DataFrame:
    required_columns = {origin_col, value_col}.union(products_cols)
    assert required_columns.issubset(
        dataframe.columns
    ), "Dataframe must contain all required columns"

    exports = dataframe[[origin_col, *products_cols, value_col]].reset_index(drop=True)
    origins = exports[origin_col].unique()

    exports = exports.set_index([origin_col, *products_cols])
    initial_total_value = exports[value_col].sum()

    new_index = pd.concat(
        [
            products_codes[products_cols]
            .assign(**{origin_col: origin})
            .set_index([origin_col, *products_cols])
            for origin in origins
        ]
    ).index

    exports = exports.reindex(new_index.drop_duplicates(), fill_value=0)

    reindexed_total_value = exports[value_col].sum()
    assert (
        initial_total_value == reindexed_total_value
    ), "Total export values must be consistent before and after reindexing"

    index_levels = exports.index.nlevels
    exports_matrix = exports.unstack(level=[i for i in range(1, index_levels)])
    exports_matrix.columns = exports_matrix.columns.droplevel(0)

    all_origins = dataframe[origin_col].unique()
    assert set(all_origins) == set(
        exports_matrix.index.unique()
    ), "All origins must be equal between products_codes DataFrame"

    all_products = set(
        [tuple(row) for _, row in products_codes[products_cols].iterrows()]
    )
    matrix_products = set([c for c in exports_matrix.columns])
    assert (
        all_products == matrix_products
    ), "All product codes must be represented in the exports matrix"

    matrix_origins = set(exports_matrix.index)
    assert (
        set(origins) == matrix_origins
    ), "All origins must be represented in the exports matrix"

    return exports_matrix


def calculate_exports_matrix_rca(exports_matrix: DataFrame) -> DataFrame:
    xcp = exports_matrix.copy()
    x = exports_matrix.sum().sum()
    xc = exports_matrix.sum(axis=1)
    xp = exports_matrix.sum(axis=0)

    rca_matrix = xcp * x
    rca_matrix = rca_matrix.div(xp)
    rca_matrix = rca_matrix.T.div(xc).T

    rca_matrix = rca_matrix.fillna(0.0)

    return rca_matrix


def mask_matrix(matrix: DataFrame, threshold: float) -> DataFrame:
    masked_matrix = matrix.copy()
    masked_matrix[masked_matrix < threshold] = 0.0
    masked_matrix[masked_matrix >= threshold] = 1.0
    return masked_matrix


def calculate_proximity_matrix(dataframe: DataFrame, symmetric: Optional[bool] = True):
    ubiquity = dataframe.sum(axis=0)

    proximity_matrix = dataframe.T @ dataframe
    proximity_matrix = proximity_matrix / ubiquity.values

    np.fill_diagonal(proximity_matrix.values, 0)
    proximity_matrix = proximity_matrix.fillna(0.0)

    if symmetric:
        proximity_matrix = np.minimum(proximity_matrix, proximity_matrix.T)

    return proximity_matrix


def calculate_relatedness_matrix(
    proximity_matrix: DataFrame, rca_matrix: DataFrame, year: int
) -> DataFrame:
    row_sums = proximity_matrix.sum(axis=1)
    wcp = rca_matrix.dot(proximity_matrix)
    wcp.columns = row_sums.index
    wcp = wcp / row_sums
    wcp = wcp.unstack().to_frame()
    wcp["Year"] = year
    wcp = wcp.set_index("Year", append=True)
    wcp.columns = ["Relatedness"]
    return wcp


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
