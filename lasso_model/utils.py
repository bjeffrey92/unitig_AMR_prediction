import torch
import os
import pandas as pd

def load_training_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'training_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'training_labels.pt'))
    return features.to_dense(), labels

def load_testing_data(data_dir):
    features = torch.load(os.path.join(data_dir, 'testing_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'testing_labels.pt'))
    return features.to_dense(), labels

def accuracy(predictions: torch.tensor, labels: torch.tensor):
    diff = predictions - labels
    correct = diff[[i < 1 for i in diff]]
    return len(correct)/len(predictions) * 100