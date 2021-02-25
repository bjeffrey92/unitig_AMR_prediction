#!/usr/bin/env python3

import sys
import pickle
import os
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from math import log10
from bayes_opt import BayesianOptimization

from GNN_model import train, utils, models


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def train_evaluate(Ab, left_out_clade, dropout, l2_alpha, lr, n_hid_1, n_hid_2):

    data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)

    inputs = train.load_data(data_dir, distances = False, adj = True,
                            left_out_clade = left_out_clade)
    training_data, testing_data, validation_data, adj = inputs
    n_feat = training_data.n_nodes

    #these hyperparams are selected from log uniform distribution
    l2_alpha = 10 ** l2_alpha
    lr = 10 ** lr 

    torch.manual_seed(0)
    model = models.GCNMaxPooling(n_feat, conv_1 = int(n_hid_1), 
                        n_hid_1 = int(n_hid_2), out_dim = 1, dropout = dropout) 

    optimizer = optim.Adam(model.parameters(), lr = lr, 
                    weight_decay = l2_alpha) #weight decay is l2 loss
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator() 
    for epoch in range(40):
        epoch += 1
        
        model, epoch_results = train.train(training_data, model, 
                                    optimizer, epoch, loss_function, 
                                    utils.mean_acc_per_bin, adj, 
                                    testing_data = testing_data, 
                                    validation_data = validation_data)
        training_metrics.add(epoch_results)

        #if testing data accuracy has plateaued
        if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                if i < 0.1]) >= 10 and epoch > 30:
            break

    #returns average testing data loss in the last five epochs
    return - sum(training_metrics.testing_data_loss[-5:])/5 #negative so can use maximization function


def fit_best_model(Ab, left_out_clade, dropout, l2_alpha, lr, n_hid_1, n_hid_2):

    data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)

    inputs = train.load_data(data_dir, distances = False, adj = True,
                            left_out_clade = left_out_clade)
    training_data, testing_data, validation_data, adj = inputs
    n_feat = training_data.n_nodes

    #these hyperparams are selected from log uniform distribution
    l2_alpha = 10 ** l2_alpha
    lr = 10 ** lr 

    torch.manual_seed(0)
    model = models.GCNMaxPooling(n_feat, conv_1 = int(n_hid_1), 
                            n_hid_1 = int(n_hid_2), out_dim = 1, 
                            dropout = dropout) 

    optimizer = optim.Adam(model.parameters(), lr = lr, 
                    weight_decay = l2_alpha) #weight decay is l2 loss
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator() 
    for epoch in range(200):
        epoch += 1
        
        model, epoch_results = train.train(training_data, model, 
                                    optimizer, epoch, loss_function, 
                                    utils.mean_acc_per_bin, adj, 
                                    testing_data = testing_data, 
                                    validation_data = validation_data)
        training_metrics.add(epoch_results)

        if epoch >= 20:
            training_metrics.log_gradients(epoch)

        if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                if i < 0.1]) >= 10 and epoch > 50:
            break

    train_acc = train.test(training_data, model, 
                    loss_function, utils.mean_acc_per_bin, adj)[1]
    test_acc = train.test(testing_data, model, 
                    loss_function, utils.mean_acc_per_bin, adj)[1]
    validation_acc = train.test(validation_data, model,
                    loss_function, utils.mean_acc_per_bin, adj)[1]

    return {'training_accuracy': train_acc, 
            'testing_accuracy': test_acc, 
            'validation_accuracy': validation_acc}


def save_CV_results(accuracies, out_dir, fname):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, fname)
    with open(out_file, 'wb') as a:
        pickle.dump(accuracies, a)


if __name__ == '__main__':

    Ab = sys.argv[1]

    pbounds = {
            'dropout': (0.2, 0.7),
            'l2_alpha': (log10(1e-4), log10(1e-3)),
            'lr': (log10(1e-5), log10(5e-4)),
            'n_hid_1': (25, 100),
            'n_hid_2': (10, 50)
    }

    clade_wise_results = {}
    for left_out_clade in [1,2,3]:
        logging.info(f'Ab = {Ab}, left out clade = {left_out_clade}')
        
        partial_fitting_function = partial(train_evaluate, 
                                    Ab = Ab, left_out_clade = left_out_clade)    

        optimizer = BayesianOptimization(
            f = partial_fitting_function,
            pbounds = pbounds,
            random_state = 1,
        )

        optimizer.maximize(n_iter = 15)
        logging.info(
            f'Completed hyperparam optimisation: {Ab}, {left_out_clade}')

        best_hyperparams = optimizer.max['params']
        dropout = best_hyperparams['dropout']
        l2_alpha = 10 ** best_hyperparams['l2_alpha']
        lr = 10 ** best_hyperparams['lr']
        n_hid_1 = best_hyperparams['n_hid_1']
        n_hid_2 = best_hyperparams['n_hid_2']

        accuracies = fit_best_model(Ab, left_out_clade, dropout, l2_alpha, 
                                    lr, n_hid_1, n_hid_2)
        logging.info(
            f'Fitted model with best hyperparams: {Ab}, {left_out_clade}')

        clade_wise_results[left_out_clade] = {
                (dropout, l2_alpha, lr, int(n_hid_1), int(n_hid_2)): accuracies}
        
    out_dir = 'GNN_model/Results/mean_acc_per_bin/Clade_wise_CV'
    fname = f'{Ab}_GCNPerNode.pkl'
    save_CV_results(clade_wise_results, out_dir, fname)