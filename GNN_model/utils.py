import torch
import os
import pandas as pd

def load_training_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'training_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'training_labels.pt'))
    return features, labels

def load_testing_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'testing_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'testing_labels.pt'))
    return features, labels

def load_adjacency_matrix(data_dir):
    adj = torch.load(os.path.join(data_dir, 'unitig_adjacency_tensor.pt'))
    return adj

def save_outputs():
    pass

def accuracy():
    pass