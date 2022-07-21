from functools import lru_cache, partial
from multiprocessing import cpu_count, Pool
from typing import Dict

import numpy as np
from torch import Tensor

from decision_tree_models.parallel_get_indices import get_indices_parallel


@lru_cache(maxsize=1)
def convert_adj_matrix(adj: Tensor) -> np.ndarray:
    return np.array(adj.indices()) + 1


def check_data_format(data: np.ndarray) -> np.ndarray:
    if data.dtype == "float64":
        return data
    else:
        return data.astype("float64")


@lru_cache(maxsize=1)
def adj_matrix_to_dict(adj: Tensor, njobs: int = 1) -> Dict:
    indices = convert_adj_matrix(adj)
    if njobs == 1:
        return {i: indices[1, indices[0] == i] for i in np.unique(indices[0])}

    if njobs == -1:
        njobs = cpu_count()

    unique_0_indices = np.unique(indices[0])
    array_chunks = np.array_split(unique_0_indices, njobs)

    pool = Pool(njobs)
    results = pool.map(partial(get_indices_parallel, indices=indices), array_chunks)

    return {k: v for d in results for k, v in d.items()}
