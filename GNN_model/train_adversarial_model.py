#!/usr/bin/env python3

import time
import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model.utils import load_training_data, load_testing_data, \
                        load_adjacency_matrix, load_countries, save_model, \
                        write_epoch_results, DataGenerator, \
                        accuracy, country_accuracy, MetricAccumulator
from GNN_model.adversarial_model import MICPredictor, Adversary

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_data(data_dir):
    adj = load_adjacency_matrix(data_dir)
    training_features, training_labels = load_training_data(data_dir)
    testing_features, testing_labels = load_testing_data(data_dir)
    training_countries, testing_countries = load_countries(data_dir)
    assert training_features.shape[1] == testing_features.shape[1], \
        'Dimensions of training and testing data not equal'
    
    training_data = DataGenerator(training_features, training_labels, 
                                training_countries)
    testing_data = DataGenerator(testing_features, testing_labels, 
                                testing_countries)

    return training_data, testing_data, adj


def main():
    data_dir =  'data/model_inputs/country_normalised/log2_azm_mic/'

    training_data, testing_data, adj = load_data(data_dir)

    predictor = MICPredictor(n_feat = training_data.n_nodes,
                            n_hid_1 = 50, 
                            n_hid_2 = 50,
                            n_hid_3 = 20,
                            out_dim = 1,
                            dropout = 0.5)    
    adversary = Adversary(n_feat = 20,
                        n_hid = 20, 
                        out_dim = max(training_data.countries.tolist()) + 1,
                        dropout = 0.3)
    assert predictor.n_hid_3 == adversary.n_feat, \
        'Nodes in penultimate layer of predictor must equal number of inputs to adversary'


if __name__ == '__main__':
    main()