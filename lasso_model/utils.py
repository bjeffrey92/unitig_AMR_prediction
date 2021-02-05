import torch
import os
from functools import lru_cache
import pandas as pd

def load_training_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'training_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'training_labels.pt'))
    return features.to_dense(), labels

def load_testing_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'testing_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'testing_labels.pt'))
    return features.to_dense(), labels

def load_metadata(data_dir):
    training_metadata = pd.read_csv(
                            os.path.join(data_dir, 'training_metadata.csv'))
    testing_metadata = pd.read_csv(
                            os.path.join(data_dir, 'testing_metadata.csv'))
    return training_metadata, testing_metadata

def accuracy(predictions: torch.tensor, labels: torch.tensor):
    diff = predictions - labels
    correct = diff[[i < 1 for i in diff]]
    return len(correct)/len(predictions) * 100

def load_adjacency_matrix(data_dir, degree_normalised = False):
    if degree_normalised:
        adj = torch.load(os.path.join(data_dir, 
                                'degree_normalised_unitig_adjacency_tensor.pt'))
    else:
        adj = torch.load(os.path.join(data_dir, 'unitig_adjacency_tensor.pt'))
    return adj

@lru_cache()
def convolve(features, adj):
    x = torch.sparse.mm(adj, features.transpose(0,1))
    return x.transpose(0,1)