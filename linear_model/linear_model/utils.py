import torch
import os
from functools import lru_cache
import pandas as pd
import numpy as np

@lru_cache(maxsize = 1)
def load_training_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'training_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'training_labels.pt'))
    return features.to_dense(), labels

@lru_cache(maxsize = 1)
def load_testing_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'testing_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'testing_labels.pt'))
    return features.to_dense(), labels

@lru_cache(maxsize = 1)
def load_metadata(data_dir):
    training_metadata = pd.read_csv(
                            os.path.join(data_dir, 'training_metadata.csv'))
    testing_metadata = pd.read_csv(
                            os.path.join(data_dir, 'testing_metadata.csv'))
    return training_metadata, testing_metadata

def accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    '''
    Prediction accuracy defined as percentage of predictions within 1 twofold 
    dilution of true value
    '''
    diff = predictions - labels
    correct = diff[[i < 1 for i in diff]]
    return len(correct)/len(predictions) * 100

def mean_acc_per_bin(predictions: torch.Tensor, labels: torch.Tensor, 
                    bin_size = 'optimise')->float:
    '''
    Splits labels into bins of size = bin_size, and calculates the prediction 
    accuracy in each bin. 
    Returns the mean accuracy across all bins
    '''
    assert len(predictions) == len(labels)
    
    if type(bin_size) != int and bin_size != 'optimise':
        raise ValueError(
            'bin_size must = optimise, or be an integer >= 1 ')
    if type(bin_size) == int and bin_size < 1:
        raise NotImplementedError(
            'If an integer, bin_size must be >= 1')
    
    #apply Freedman-Diaconis rule to get optimal bin size
    #https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    if bin_size == 'optimise': 
        IQR = np.subtract(*np.percentile(labels, [75, 25]))
        bin_size = 2 * IQR/(len(labels) ** (1/3)) 
        bin_size = int(np.ceil(bin_size)) #round up cause if less than 1 will not work with accuracy function

    min_value = int(np.floor(min(labels)))
    max_value = int(np.floor(max(labels)))
    bins = list(range(min_value, max_value + bin_size, bin_size))
    binned_labels = np.digitize(labels, bins)

    df = pd.DataFrame({
        'labels': labels,
        'predictions': predictions,
        'binned_labels': binned_labels
    }) #to allow quick searches across bins

    #percentage accuracy per bin
    def _get_accuracy(d):
        acc = accuracy(torch.tensor(d.labels.to_list()), 
                      torch.tensor(d.predictions.to_list()))   
        return acc
    bin_accuracies = df.groupby(df.binned_labels).apply(_get_accuracy)

    return bin_accuracies.mean()

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