#!/usr/bin/env python3

import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model.utils import load_training_data, load_testing_data, \
                         load_adjacency_matrix, save_model, accuracy,\
                         write_epoch_results, DataGenerator
from GNN_model.models import GCN


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def epoch_(model, data, adj):
    data.reset_generator()
    
    outputs = [None] * data.samples
    for i in range(data.samples):
        features = data.next_features()
        outputs[i] = model(features, adj).unsqueeze(0)

    output_tensor = torch.cat(outputs, dim = 0)
    return output_tensor


def train(data, model, optimizer, adj, epoch, 
        loss_function, testing_data = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    
    output = epoch_(model, data, adj)
    loss_train = loss_function(output, data.labels)
    acc_train = accuracy(output, data.labels)
    
    if testing_data:
        loss_test, acc_test = test(testing_data, model, adj, loss_function)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'
    
    loss_train.backward()
    optimizer.step()


    logging.info(f'Epoch {epoch} complete\n' + \
                f'\tTime taken = {time.time() - t}\n' + \
                f'\tTraining Data Loss = {loss_train}\n' + \
                f'\tTraining Data Accuracy = {acc_train}\n'
                f'\tTesting Data Loss = {loss_test}\n' + \
                f'\tTesting Data Accuracy = {acc_test}\n'
                )

    return model, (float(loss_train), acc_train, float(loss_test), acc_test)


def test(data, model, adj, loss_function):
    data.reset_generator()
    model.train(False)
    
    output = epoch_(model, data, adj)
    loss = loss_function(output, data.labels)
    acc = accuracy(output, data.labels)

    return loss, acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 200,
                        help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.001,
                        help = 'Initial learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 5e-4,
                        help = 'Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type = float, default = 0.5,
                        help = 'Dropout rate (1 - keep probability).')
    parser.add_argument('--data_dir', type = str, 
                        help = 'Directory from which to load input data')
    parser.add_argument('--summary_file', type = str, 
                        help = '.tsv file to write epoch summaries to.')
    parser.add_argument('--logfile', type = str, default = '',
                        help = 'Path to log file. \
                        If left blank logging will be printed to stdout only.')

    return parser.parse_args()


def main(args):
    if args.logfile:
        logging.basicConfig(filename = args.logfile)

    adj = load_adjacency_matrix(args.data_dir)
    training_features, training_labels = load_training_data(args.data_dir)
    testing_features, testing_labels = load_testing_data(args.data_dir)
    assert training_features.shape[1] == testing_features.shape[1], \
        'Dimensions of training and testing data not equal'
    
    training_data = DataGenerator(training_features, training_labels)
    testing_data = DataGenerator(testing_features, testing_labels)

    model = GCN(n_feat = 1,
                n_hid_1 = 4,
                n_hid_2 = 8,
                out_dim = 1,
                dropout = args.dropout)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, 
                        weight_decay = args.weight_decay)
    loss_function = nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epoch):
        epoch += 1
        model, epoch_results = train(training_data, model, optimizer, 
                                    adj, epoch, loss_function, testing_data)
        write_epoch_results(epoch, epoch_results, args.summary_file)
    logging.info(f'Model Fitting Complete. Time elapsed {start_time - time.time()}')


if __name__ == '__main__':
    main(parse_args())
