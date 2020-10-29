#!/usr/bin/env python3

import argparse
import time
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim

from GNN_model.utils import load_training_data, load_testing_data, \
                         load_adjacency_matrix, save_outputs, accuracy
from GNN_model.models import GCN


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def train(epoch, features, labels):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(features, labels)
    acc_train = accuracy(features, labels)
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    logging.info('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))


def test():
    pass


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 200,
                        help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.001,
                        help = 'Initial learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 5e-4,
                        help = 'Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type = int, default = 16,
                        help = 'Number of hidden units.')
    parser.add_argument('--dropout', type = float, default = 0.5,
                        help = 'Dropout rate (1 - keep probability).')
    parser.add_argument('--data_dir', type = str, 
                        help = 'Directory from which to load input data')
    parser.add_argument('--logfile', type = str, default = '',
                        help = 'Path to log file. \
                        If left blank logging will be printed to stdout only.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.logfile:
        logging.basicConfig(filename = args.logfile)

    adj = load_adjacency_matrix(args.data_dir)
    training_features, training_labels = load_training_data(args.data_dir)
    testing_features, testing_labels = load_testing_data(args.data_dir)

    assert training_features.shape[1] == testing_features.shape[1], \
        'Dimensions of training and testing data not equal'

    model = GCN(n_feat = training_features.shape[1],
                n_hid = args.hidden,
                n_class = 2,
                dropout = args.dropout)
    optimizer = optim.Adam(model.parameters(), lr = args.lr, 
                        weight_decay = args.weight_decay)

    start_time = time.time()
    for epoch in range(args.epoch):
        train(epoch, model, optimizer)
    logging.info(f'Model Fitting Complete. Time elapsed {start_time - time.time()}')