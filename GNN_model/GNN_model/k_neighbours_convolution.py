import pandas as pd
import numpy as np
import torch
import warnings
import sys
import pickle
from torch_sparse import SparseTensor
from scipy.sparse import identity, csr_matrix, coo_matrix

from GNN_model.utils import load_training_data, load_testing_data, \
                            load_adjacency_matrix


def get_k_neighbours(adj_tensor:torch.tensor, k: int):
    if type(k) != int:
        raise TypeError('K must be an integer')
    if k == 1:
        warnings.warn(
            'K == 1, therefore just adjacency matrix and self loops will be returned',
                    RuntimeWarning)
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
    k_neighbours_dict[0] = I.tocoo() #convert for consistency    
    k_neighbours_dict[1] = adj.tocoo() * 2 #1st neighbour is same node
    #interate across all values between 2 and k and get those neighbours
    for k_ in list(range(2, k + 1)):
        k_neighbours_matrix = k_neighbours(adj, k_)

        #need to subtract the reverse loops created in calculating the k neighbours
        k_neighbours_matrix = k_neighbours_matrix - I
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
        data = np.ones(len(row)) * (k_ + 1) #1st neighbour is same node
        N = max(row) + 1
        k_neighbours_dict[k_] = coo_matrix((data, (row, col)), shape=(N, N))

    #improve searchability
    def convert_to_dataframes(matrix):
        return pd.DataFrame({'row': matrix.row, 
                            'col':matrix.col, 
                            'k': matrix.data})
    k_neighbours = pd.concat([convert_to_dataframes(v) 
                            for v in k_neighbours_dict.values()]) #return as a single dataframe
    k_neighbours.reset_index(inplace = True, drop = True)

    return k_neighbours


def get_max_k_neighbours(k_neighbours):
    '''
    Returns dictionary with the max number of possible neighbours at each value k
    '''
    k_neighbour_counts = \
        k_neighbours.groupby(['row', 'k']).apply(len).reset_index()
    def max_n(df):
        return max(df[0])
    return k_neighbour_counts.groupby(['k']).apply(max_n).to_dict()


def build_neighbourhoods(k_neighbours, k_neighbour_counts):
    '''
    Returns indices for neighbourhoods, which is the k step neighbours of each 
    node up to the maximum number neighbours for each value k of any node
    '''
    #get columns linked to each row by k steps as a list
    def get_cols(df):
        return df.col.tolist()
    cols_per_k_and_row = k_neighbours.groupby(['row', 'k']).apply(get_cols)
    cols_per_k_and_row = cols_per_k_and_row.reset_index()

    n_nodes = max(k_neighbours.row) + 1

    #get indices of the nodes for each step k
    def k_wise_neighbourhoods(df):
        k = df.k.iloc[0]
        k_max = k_neighbour_counts[k]     
        
        def nth_element(x, n):
            try:
                return x[n]
            except IndexError: pass
        out = []
        for n in range(k_max):
            indices = df[0].apply(lambda x: nth_element(x, n)).tolist()
            out.append([i for i in indices if not pd.isna(i)])

        return out

    #dictionary of the indices of the neighbours for each step in the neighbourhood
    neighbourhoods_dict = {}
    for k, df in cols_per_k_and_row.groupby('k'):
        neighbourhoods_dict[k] = k_wise_neighbourhoods(df)
        
    return neighbourhoods_dict


def feature_neighbourhoods(features, neighbourhoods_dict, k_neighbours,
                            k_neighbour_counts):
    features_df = pd.DataFrame(features)

    tensor_size = (len(features_df.columns), 
                    sum(k_neighbour_counts.values()))
    max_values = len(k_neighbours)
    def get_sparse_neighbourhood_tensor(f):
        indices_0 = [None] * max_values
        indices_1 = [None] * max_values
        N = 0
        d = 0
        for k in sorted(neighbourhoods_dict.keys()):
            k_indices = neighbourhoods_dict[k]
            for n in k_indices:
                f_2 = f.iloc[n]
                ones = list(f_2.loc[f_2 == 1].index)
                indices_0[N:len(ones)] = ones
                indices_1[N:len(ones)] = [d] * len(ones)
                N += len(ones)
                d += 1
        indices = [indices_0[:N], indices_1[:N]]
        return torch.sparse_coo_tensor(indices, np.ones(N, dtype = np.float32), 
                                        size = tensor_size)

    sparse_feature_tensors = features_df.apply(get_sparse_neighbourhood_tensor, 
                                                axis = 1)

    return sparse_feature_tensors.tolist()


if __name__ == '__main__':
    # Ab = sys.argv[1]
    train_test = 0

    Abs = ['log2_azm_mic',
        'log2_cip_mic',
        'log2_cro_mic',
        'log2_cfx_mic']
    for Ab in Abs:
        data_dir = f'data/model_inputs/freq_5_95/{Ab}/'    
        adj_tensor = load_adjacency_matrix(data_dir, degree_normalised = False)

        k = 3
        k_neighbours = get_k_neighbours(adj_tensor, k) #values is #steps - 1
        k_neighbour_counts = get_max_k_neighbours(k_neighbours)
        neighbourhoods_dict = build_neighbourhoods(k_neighbours, k_neighbour_counts)

        if int(train_test) == 1:
            training_data = load_training_data(data_dir)[0].tolist()
            training_features = feature_neighbourhoods(training_data, 
                                                    neighbourhoods_dict, 
                                                    k_neighbours,
                                                    k_neighbour_counts)
            with open(f'{data_dir}{k}_convolved_training_features.pt', 'wb') as a:
                pickle.dump(training_features, a)                    

        else:
            testing_data = load_testing_data(data_dir)[0].tolist()
            testing_features = feature_neighbourhoods(testing_data, 
                                                    neighbourhoods_dict, 
                                                    k_neighbours,
                                                    k_neighbour_counts)
            with open(f'{data_dir}{k}_convolved_testing_features.pt', 'wb') as a:
                pickle.dump(testing_features, a)

        print(Ab, '\tdone')