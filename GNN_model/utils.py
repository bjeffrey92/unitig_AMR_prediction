import torch
import os
import random
import numpy as np

def load_training_data(data_dir, to_dense = True):
    features = torch.load(os.path.join(data_dir, 'training_features.pt'))
    if to_dense:
        features = features.to_dense()
    labels = torch.load(os.path.join(data_dir, 'training_labels.pt'))
    return features, labels

def load_testing_data(data_dir, to_dense = True):
    features = torch.load(os.path.join(data_dir, 'testing_features.pt'))
    if to_dense:
        features = features.to_dense()
    labels = torch.load(os.path.join(data_dir, 'testing_labels.pt'))
    return features, labels

def load_adjacency_matrix(data_dir):
    adj = torch.load(os.path.join(data_dir, 'unitig_adjacency_tensor.pt'))
    return adj

def load_countries(data_dir):
    training_countries = torch.load(os.path.join(data_dir, 
                                                'training_countries.pt'))
    testing_countries = torch.load(os.path.join(data_dir, 
                                                'testing_countries.pt'))
    def transform(countries):
        countries = countries.tolist()
        c_index =  countries.index(max(countries))
        return torch.LongTensor([c_index])
    
    training_countries = torch.cat(
                            list(map(transform, training_countries.unbind(0))))
    testing_countries = torch.cat(
                            list(map(transform, testing_countries.unbind(0))))
    return training_countries, testing_countries

def write_epoch_results(epoch, epoch_results, summary_file):
    if not os.path.isfile(summary_file):
        with open(summary_file, 'w') as a:
            a.write('epoch\ttraining_data_loss\ttraining_data_acc\ttesting_data_loss\ttesting_data_acc\n')

    with open(summary_file, 'a') as a:
        line = str(epoch) + '\t' + '\t'.join([str(i) for i in epoch_results])
        a.write(line + '\n')

def save_model(model, filename):
    torch.save(model, filename)

def accuracy(predictions: torch.tensor, labels: torch.tensor):
    diff = abs(predictions - labels)
    correct = diff[[i < 1 for i in diff]]
    return len(correct)/len(predictions) * 100

def country_accuracy(predictions: torch.tensor, labels: torch.tensor):
    assert len(predictions) == len(labels), \
        'Predictions and labels are of unequal lengths'
    predictions = predictions.tolist()
    labels = labels.tolist()
    correct_bool = [predictions[i].index(max(predictions[i]))  == labels[i] 
                        for i in range(len(predictions))]
    return len(np.array(labels)[correct_bool])/len(labels) * 100

#custom loss function 
def logcosh(true, pred):
    loss = torch.log(torch.cosh(pred - true))
    return torch.sum(loss)

class DataGenerator():
    def __init__(self, features, labels, auto_reset_generator = True):
        assert len(features) == len(labels), \
            'Features and labels are of different length'
        self.features = self._parse_features(features)
        self.labels = labels
        self.n_nodes = self.features[0].shape[0]
        self.n_samples = len(labels)
        self.n = 0
        self._index = list(range(self.n_samples))
        self.auto_reset_generator = auto_reset_generator

    def _parse_features(self, features):
        l = [None] * len(features)
        for i in range(len(features)):
            l[i] = features[i].unsqueeze(1)
        return l

    def _iterate(self):
        self.n += 1
        if self.n == self.n_samples and self.auto_reset_generator:
            sample = self.features[self.n - 1], self.labels[self.n - 1]
            self.reset_generator()
            yield sample
        else:
            yield self.features[self.n - 1], self.labels[self.n - 1]

    def next_sample(self):
        return next(self._iterate())

    def reset_generator(self):
        self.n = 0

    def shuffle_samples(self):
        random.shuffle(self._index)
        self.labels = torch.tensor(list(
                        {i:self.labels[i] for i in self._index}.values()))
        self.features = list(
            {i:self.features[i] for i in self._index}.values())
