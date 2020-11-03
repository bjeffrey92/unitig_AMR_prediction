import torch
import os

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

def write_epoch_results(epoch, epoch_results, summary_file):
    if not os.path.isfile(summary_file):
        with open(summary_file, 'w') as a:
            a.write('epoch\ttraining_data_loss\ttraining_data_acc\ttesting_data_loss\ttesting_data_loss\n')

    with open(summary_file, 'a') as a:
        line = str(epoch) + '\t' + '\t'.join([str(i) for i in epoch_results])
        a.write(line + '\n')

def save_model(model, filename):
    pass

def accuracy(predictions: torch.tensor, labels: torch.tensor):
    diff = predictions - labels
    correct = diff[[i < 1 for i in diff]]
    return len(correct)/len(predictions) * 100

class DataGenerator():
    def __init__(self, features, labels):
        assert len(features) == len(labels), \
            'Features and labels are of different length'
        self.features = self._parse_features(features)
        self.labels = torch.FloatTensor(labels)
        self.samples = len(labels)
        self.n = 0

    def _parse_features(self, features):
        l = [None] * len(features)
        for i in range(len(features)):
            l[i] = features[i].unsqueeze(1)
        return l

    def _iterate_features(self):
        self.n += 1
        yield self.features[self.n - 1]

    def next_features(self):
        return next(self._iterate_features())

    def reset_generator(self):
        self.n = 0