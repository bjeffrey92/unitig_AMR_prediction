#!/usr/bin/env python3

import argparse
import time
import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model.utils import load_training_data, load_testing_data, \
                         load_adjacency_matrix, save_model, accuracy,\
                         write_epoch_results, DataGenerator
from GNN_model.models import GCN, GCNPerNode


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def epoch_(model, data, adj):
    data.reset_generator()
   
    outputs = [None] * data.n_samples
    for i in range(data.n_samples):
        features = data.next_sample()[0]
        outputs[i] = model(features, adj).unsqueeze(0)

    output_tensor = torch.cat(outputs, dim = 0)
    
    return output_tensor


def batch_train(data, model, optimizer, adj, epoch, 
        loss_function, testing_data = None, batch_size = 32):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    data.reset_generator()
    data.shuffle_samples()

    batches = math.floor(data.n_samples/batch_size)
    final_batch = False
    for batch in range(batches):
        if final_batch: break #last batch will be empty if batch_size is not a multiple of n samples

        #add remainder, to what would be second from last batch
        #better to have one slightly larger batch at the end than one very small one
        if batch == batches - 1:
            batch_size += data.n_samples % batch_size
            final_batch = True

        outputs = [None] * batch_size
        labels = [None] * batch_size
        for i in range(batch_size):
            features, label = data.next_sample()
            labels[i] = label.unsqueeze(0)
            outputs[i] = model(features, adj).unsqueeze(0)
                
        output_tensor = torch.cat(outputs, dim = 0)
        labels = torch.cat(labels, dim = 0)

        loss_train = loss_function(output_tensor, labels)
        loss_train.backward()
        optimizer.step()    

    output = epoch_(model, data, adj)
    loss_train = float(loss_function(output, data.labels))
    acc_train = accuracy(output, data.labels)
    
    if testing_data:
        loss_test, acc_test = test(testing_data, model, adj, loss_function)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'
    
    logging.info(f'Epoch {epoch} complete\n' + \
                f'\tTime taken = {time.time() - t}\n' + \
                f'\tTraining Data Loss = {loss_train}\n' + \
                f'\tTraining Data Accuracy = {acc_train}\n'
                f'\tTesting Data Loss = {loss_test}\n' + \
                f'\tTesting Data Accuracy = {acc_test}\n'
                )

    return model, (loss_train, acc_train, loss_test, acc_test)


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
    loss_train = float(loss_train) #to write it to file

    logging.info(f'Epoch {epoch} complete\n' + \
                f'\tTime taken = {time.time() - t}\n' + \
                f'\tTraining Data Loss = {loss_train}\n' + \
                f'\tTraining Data Accuracy = {acc_train}\n'
                f'\tTesting Data Loss = {loss_test}\n' + \
                f'\tTesting Data Accuracy = {acc_test}\n'
                )

    return model, (loss_train, acc_train, loss_test, acc_test)


def test(data, model, adj, loss_function):
    data.reset_generator()
    model.train(False)
    
    output = epoch_(model, data, adj)
    loss = float(loss_function(output, data.labels))
    acc = accuracy(output, data.labels)

    return loss, acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 200,
                        help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        help = 'Initial learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 5e-4,
                        help = 'Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type = float, default = 0.5,
                        help = 'Dropout rate (1 - keep probability).')
    parser.add_argument('--data_dir', type = str, 
                        help = 'Directory from which to load input data')
    parser.add_argument('--summary_file', type = str, 
                        help = '.tsv file to write epoch summaries to.')
    parser.add_argument('--alt_model', action = 'store_true', default = False,
                        help = 'Use per node model formulation')
    parser.add_argument('--batch_size', default = None, 
                        help = 'Size of batches to use for batch training, \
                            if None training will be done per epoch')
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

    if not args.alt_model:
        model = GCN(n_feat = training_data.n_features,
                    n_hid_1 = 4,
                    n_hid_2 = 8,
                    out_dim = 1,
                    dropout = args.dropout)
    else: 
        model = GCNPerNode(n_feat = training_data.n_features, n_hid_1 = 1000, 
                        n_hid_2 = 500, out_dim = 1, dropout = 0.5)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, 
                        weight_decay = args.weight_decay)
    loss_function = nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epoch):
        epoch += 1
        if args.batch_size:
            #parameters tuned per batch
            model, epoch_results = batch_train(training_data, model, 
                                            optimizer, adj, epoch, loss_function, 
                                            testing_data, 
                                            batch_size = args.batch_size)
        else:    
            #parameters tuned per epoch
            model, epoch_results = train(training_data, model, 
                                    optimizer, adj, epoch, loss_function, 
                                    testing_data)
        write_epoch_results(epoch, epoch_results, args.summary_file)
    logging.info(f'Model Fitting Complete. Time elapsed {start_time - time.time()}')


if __name__ == '__main__':
    main(parse_args())
