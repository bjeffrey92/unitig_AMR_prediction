from functools import lru_cache
from typing import Dict, List

import numpy as np
from torch import Tensor, unique


@lru_cache(maxsize=1)
def convert_adj_matrix(adj: Tensor) -> np.ndarray:
    return np.array(adj.indices()).astype(int)


def check_data_format(data: np.ndarray) -> np.ndarray:
    if data.dtype == "float64":
        return data
    else:
        return data.astype("float64")
