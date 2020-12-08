import pandas as pd
import numpy as np
import torch
from torch_sparse import SparseTensor
from scipy.sparse import identity, csr_matrix, coo_matrix

from GNN_model.utils import load_training_data, load_testing_data, \
                            load_adjacency_matrix


def get_k_neighbours(adj_tensor:torch.tensor, k: int):
    if type(k) != int:
        raise TypeError('K must be an integer')
    if k == 1:
        raise Warning('K == 1, therefore adjacency matrix will be returned')
    elif k < 1:
        raise ValueError('K must be at least 1')
    
    #convert back to scipy sparse matrix to make easier to manipulate
    adj_tensor = adj_tensor.coalesce()
    values = adj_tensor.values().tolist()
    row, col = adj_tensor.indices()
    N = max(row.tolist()) + 1
    adj = csr_matrix((values, (row.tolist(), col.tolist())), shape = (N, N))
    
    #stored adjacency matrix is sum of original and identity matrix
    I = identity(adj.shape[0])
    adj = adj - I #subtract I to calculate neighbourhoods
    
    def k_neighbours(adj, k):
        adj_k = adj ** k #gets neighbours which are k edges away from each node
        adj_k[adj_k > 1] = 1 #set values to one
        return adj_k
    
    k_neighbours_dict = {k_:None for k_ in list(range(k + 1))}
    k_neighbours_dict[1] = adj.tocoo()
    k_neighbours_dict[0] = I.tocoo() #convert for consistency    
    #interate across all values between 2 and k and get those neighbours
    for k_ in list(range(2, k + 1)):
        k_neighbours_matrix = k_neighbours(adj, k_)

        #need to subtract the reverse loops created in calculating the k neighbours
        k_neighbours_matrix = k_neighbours_matrix - adj
        if k_ > 2:
            for intermediate_k in range(2, k_):
                k_neighbours_matrix = k_neighbours_matrix - \
                                        k_neighbours_dict[intermediate_k]
        k_neighbours_matrix[k_neighbours_matrix < 0] = 0 #subtraction will induce some negative values
        
        #now need to rebuild matrix to remove 0s
        k_neighbours_matrix = k_neighbours_matrix.tocoo()
        row = k_neighbours_matrix.row[k_neighbours_matrix.data == 1]
        col = k_neighbours_matrix.col[k_neighbours_matrix.data == 1]
        data = np.ones(len(row))
        N = max(row) + 1
        k_neighbours_dict[k_] = coo_matrix((data, (row, col)), shape=(N, N))

    #improve searchability
    def convert_to_dataframes(matrix):
        return pd.DataFrame({'row': matrix.row, 'col':matrix.col})
    k_neighbours_dict = {k_:convert_to_dataframes(v) 
                            for k_,v in k_neighbours_dict.items()}

    return k_neighbours_dict


def build_neighbourhoods(features, k_neighbours_dict):
    def extract_neighbours(df):
        return df.col.tolist()

    for k, df in k_neighbours_dict.items():
        df.groupby('row').apply(extract_neighbours)


if __name__ == '__main__':
    data_dir = ''    
    adj_tensor = load_adjacency_matrix(data_dir, degree_normalised = False)
    training_data = load_training_data(data_dir)
    testing_data = load_testing_data(data_dir)

    k = 4
    k_neighbours_dict = get_k_neighbours(adj_tensor, k)