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
                         country_accuracy, MetricAccumulator
from GNN_model.models import GCNCountry
from GNN_model.train import epoch_, train, test

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def load_data(data_dir):
    adj = load_adjacency_matrix(data_dir)
    
    training_features = load_training_data(data_dir)[0] #just return first element which is unitigs, second is MIC
    testing_features = load_testing_data(data_dir)[0]
    training_countries, testing_countries = load_countries(data_dir)

    training_data = DataGenerator(training_features, training_countries)
    testing_data = DataGenerator(testing_features, testing_countries)

    return training_data, testing_data, adj


def main():
    data_dir = 'data/model_inputs/country_normalised/log2_azm_mic/'
    training_data, testing_data, adj = load_data(data_dir)

    model = GCNCountry(n_feat = training_data.n_nodes, 
                        n_hid_1 = 50,
                        n_hid_2 = 50, 
                        out_dim = 13,
                        dropout = 0.2)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, 
                        weight_decay = 5e-4)
    loss_function = nn.CrossEntropyLoss()

    training_metrics = MetricAccumulator() 

    start_time = time.time()
    for epoch in range(500):
        epoch += 1
        model, epoch_results = train(training_data, model, 
                                optimizer, adj, epoch, loss_function, 
                                country_accuracy, testing_data)
        training_metrics.add(epoch_results)
        if epoch >= 20:
            training_metrics.log_gradients(epoch)
        write_epoch_results(epoch, epoch_results, 'country_classifier.tsv')
    logging.info(f'Model Fitting Complete. Time elapsed {time.time() - start_time}')


if __name__ == '__main__':
    main()