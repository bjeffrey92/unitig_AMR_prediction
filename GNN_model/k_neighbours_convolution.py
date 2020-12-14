import pandas as pd
import numpy as np
import torch
import warnings
import sys
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


def build_neighbourhoods(features, k_neighbours, k_neighbour_counts):
    features_T = features.transpose(0,1)
    features_df = pd.DataFrame(features_T.tolist()) 

    #padding with negative one so it can be discerned from 0 in the data
    padding = np.zeros(len(features))
    padding_expanaded = np.expand_dims(padding, 0)
    def get_neighbourhood_tensor(df):
        neighbourhood = [features_df.iloc[
                            df.loc[df.k == k].col.to_list()].to_numpy() 
                                for k in df.k.unique()] #all k step neighbours of given node in f
        
        #pad neighbourhood with negative one where there are missing values
        if len(neighbourhood) < k + 1:
             neighbourhood += [padding_expanaded] * (k + 1 - len(neighbourhood))
        for i in range(len(neighbourhood)):
            padding_needed = k_neighbour_counts[i + 1] - len(neighbourhood[i])
            if padding_needed > 0:
                neighbourhood[i] = np.vstack((neighbourhood[i], 
                                        np.tile(padding, (padding_needed,1))))
        return torch.cat([torch.FloatTensor(i) for i in neighbourhood])

    neighbourhood_tensors_dict = {}
    nodes = max(k_neighbours.row)
    for row, df in k_neighbours.groupby(['row']):
        neighbourhood_tensors_dict[row] = get_neighbourhood_tensor(df) #tensor of all neighbours for one node in the graph for each sample
        print(f'\r{row}/{nodes} nodes processed', end = '', flush = True)

    #ensure neighbourhood tensors are ordered by node 
    neighbourhood_tensors = [neighbourhood_tensors_dict[i].transpose(0,1) 
                            for i in sorted(neighbourhood_tensors_dict.keys())]
    neighbourhood_tensors = torch.stack(neighbourhood_tensors).transpose(0)
    
    return [n.to_sparse() for n in neighbourhood_tensors]


if __name__ == '__main__':
    # Ab = sys.argv[1]
    # train_test = sys.argv[2]
    Ab = 'log2_azm_mic'

    data_dir = f'data/model_inputs/freq_5_95/{Ab}/'    
    adj_tensor = load_adjacency_matrix(data_dir, degree_normalised = False)

    k = 3
    k_neighbours = get_k_neighbours(adj_tensor, k) #values is #steps - 1
    k_neighbour_counts = get_max_k_neighbours(k_neighbours)

    # if int(train_test) == 1:
    training_data = load_training_data(data_dir)[0]
    training_features = build_neighbourhoods(training_data, k_neighbours, 
                                            k_neighbour_counts)
    torch.save(training_features, 
            f'{data_dir}{k}_convolved_training_features.pt')
    # else:
    testing_data = load_testing_data(data_dir)[0]
    testing_features = build_neighbourhoods(testing_data, k_neighbours,
                                            k_neighbour_counts)
    torch.save(testing_features, 
            f'{data_dir}{k}_convolved_testing_features.pt')

