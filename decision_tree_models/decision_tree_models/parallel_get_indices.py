def get_indices_parallel(indices_0, indices):
    return {i: indices[1, indices[0] == i] for i in indices_0}
