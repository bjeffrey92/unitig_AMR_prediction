from functools import lru_cache

import numpy as np
from torch import Tensor


@lru_cache(maxsize=1)
def convert_adj_matrix(adj: Tensor) -> np.ndarray:
    return np.array(adj.indices()) + 1


def check_data_format(data: np.ndarray) -> np.ndarray:
    if data.dtype == "float64":
        return data
    else:
        return data.astype("float64")


@lru_cache(maxsize=1)
def adj_matrix_to_dict(adj: Tensor):
    indices = convert_adj_matrix(adj)
    return {i: indices[1, indices[0] == i] for i in np.unique(indices[0])}
