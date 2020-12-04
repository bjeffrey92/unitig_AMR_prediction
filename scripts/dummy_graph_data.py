import pickle
import networkx as nx
import numpy as np
import torch
import random
from scipy import sparse
from torch_sparse import SparseTensor

from scripts import gcn_vis


def convert_to_tensor(matrix, torch_sparse_coo = True):
    
    shape = matrix.shape
    
    row = torch.LongTensor(matrix.row)
    col = torch.LongTensor(matrix.col)
    value = torch.Tensor(matrix.data)
    
    sparse_tensor = SparseTensor(row = row, col = col, 
                        value = value, sparse_sizes = shape)

    if torch_sparse_coo:
        return sparse_tensor.to_torch_sparse_coo_tensor()
    else:
        return sparse_tensor


def create_data(G, out_dir):
    n_nodes = G.number_of_nodes() 

    #extract adjacency matrix and convert to sparse tensor 
    adj = nx.adj_matrix(G)
    adj = adj.toarray()
    adj_sparse = sparse.csr_matrix(adj)
    adj_tensor = convert_to_tensor(adj_sparse.tocoo())

    gene_1_nodes = [17637, 7034, 236673, 2915]
    gene_2_nodes = [367704, 70817, 21675, 291384]

    labels = []
    features = []
    #make resistant samples with two resistance genes
    for i in range(250):
        for node in G:
            if node in gene_1_nodes + gene_2_nodes:
                G.nodes[node]['attr'] = 1.0 #label nodes of two 'genes'
            else:
                G.nodes[node]['attr'] = float(np.random.binomial(1, 2/n_nodes)) #noise
        labels.append(1)
        features.append(gcn_vis.get_attribute_vector(G))

    #make non resistant samples with random unitigs
    for i in range(250):
        for node in G:
            G.nodes[node]['attr'] = float(np.random.binomial(1, 10/n_nodes)) 
        labels.append(0)
        features.append(gcn_vis.get_attribute_vector(G))

    indices = list(range(500))
    random.shuffle(indices)

    training_features = torch.tensor([features[i] for i in indices[:300]])
    testing_features = torch.tensor([features[i] for i in indices[300:]])
    training_labels = torch.tensor([labels[i] for i in indices[:300]])
    testing_labels = torch.tensor([labels[i] for i in indices[300:]])

    training_features = training_features.to(torch.float32)
    testing_features = testing_features.to(torch.float32)
    training_labels = training_labels.to(torch.float32)
    testing_labels = testing_labels.to(torch.float32)

    torch.save(adj_tensor, 'dummy_data/unitig_adjacency_tensor.pt')
    torch.save(training_features.to_sparse(), 'dummy_data/training_features.pt')
    torch.save(testing_features.to_sparse(), 'dummy_data/testing_features.pt')
    torch.save(training_labels, 'dummy_data/training_labels.pt')
    torch.save(testing_labels, 'dummy_data/testing_labels.pt')


if __name__ == '__main__':
    with open('smaller_graph.pkl', 'rb') as a:
        G = pickle.load(a)

    create_data(G)
