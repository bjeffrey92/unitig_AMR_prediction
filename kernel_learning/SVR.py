#!/usr/bin/env python3

import sys
import os 
import pickle 
import numpy as np
import itertools
from math import log10
import torch
from pandas import Series
from bayes_opt import BayesianOptimization
from functools import lru_cache, partial
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from lasso_model.utils import load_training_data, load_testing_data
from GNN_model.utils import mean_acc_per_bin, load_metadata


@lru_cache(maxsize = 1)
def load_data(Ab: str, left_out_cluster: int)-> tuple: 
    root_dir = 'data/model_inputs/freq_5_95/'
    data_dir = os.path.join(root_dir, Ab, 'gwas_filtered')

    left_out_cluster = int(left_out_cluster)

    train_labels = load_training_data(data_dir)[1].tolist()
    test_labels = load_testing_data(data_dir)[1].tolist()
    training_metadata, testing_metadata = load_metadata(data_dir)

    training_indices = training_metadata.loc[
                        training_metadata.clusters != left_out_cluster].index
    testing_indices = testing_metadata.loc[
                        testing_metadata.clusters != left_out_cluster].index
    validation_indices_1 = training_metadata.loc[
                        training_metadata.clusters == left_out_cluster].index #extract data from training set
    validation_indices_2 = testing_metadata.loc[
                        testing_metadata.clusters == left_out_cluster].index #extract data from testing set
    
    training_labels = Series(train_labels)[training_indices].tolist()
    testing_labels = Series(test_labels)[testing_indices].tolist()
    validation_labels = Series(train_labels)[validation_indices_1].tolist() + \
                    Series(test_labels)[validation_indices_2].tolist() 

    kernel_dir = f'kernel_learning/{Ab}/gwas_filtered/cluster_CV'
    k_train_file = os.path.join(kernel_dir, 
                        f'clade_{left_out_cluster}_left_out_k_train.pkl')
    k_test_file = os.path.join(kernel_dir, 
                        f'clade_{left_out_cluster}_left_out_k_test.pkl')
    k_validate_file = os.path.join(kernel_dir, 
                        f'clade_{left_out_cluster}_left_out_k_validate.pkl')
    with open(k_train_file, 'rb') as a:
        k_train = pickle.load(a)
    with open(k_test_file, 'rb') as a:
        k_test = pickle.load(a)
    with open(k_validate_file, 'rb') as a:
        k_validate = pickle.load(a)

    return (k_train, training_labels), \
            (k_test, testing_labels), \
            (k_validate, validation_labels)


def train_evaluate(Ab, left_out_cluster, C, epsilon, verbose = False):

    train_data, test_data, _ = load_data(Ab, left_out_cluster)
    k_train, training_labels = train_data
    k_test, testing_labels = test_data

    #hyperparams are selected from log uniform distribution
    C = 10 ** C
    epsilon = 10 ** epsilon

    model = SVR(C = C, epsilon = epsilon, kernel = 'precomputed')
    model.fit(k_train, training_labels)

    train_acc = mean_acc_per_bin(torch.as_tensor(model.predict(k_train)), 
                        torch.as_tensor(training_labels))
    test_acc = mean_acc_per_bin(torch.as_tensor(model.predict(k_test)), 
                        torch.as_tensor(testing_labels))
    
    if verbose:
        print(f'{C, epsilon}\n', 
            f'Training Data Accuracy = {train_acc}\n',
            f'Testing Data Accuracy = {test_acc}\n')

    #negative error to use maximisation function
    return -mean_squared_error(testing_labels, model.predict(k_test))


def fit_best_model(Ab, left_out_cluster, C, epsilon):
    
    train_data, test_data, validation_data = load_data(Ab, left_out_cluster)
    k_train, training_labels = train_data
    k_test, testing_labels = test_data
    k_validate, validation_labels = validation_data

    model = SVR(C = C, epsilon = epsilon, kernel = 'precomputed')
    model.fit(k_train, training_labels)

    #convert to torch.as_tensor for compatibility 
    train_pred = torch.as_tensor(model.predict(k_train))
    test_pred = torch.as_tensor(model.predict(k_test))
    validation_pred = torch.as_tensor(model.predict(k_validate))

    training_labels = torch.as_tensor(training_labels)
    testing_labels = torch.as_tensor(testing_labels)
    validation_labels = torch.as_tensor(validation_labels)

    train_acc = mean_acc_per_bin(train_pred, training_labels)
    test_acc = mean_acc_per_bin(test_pred, testing_labels)
    validation_acc = mean_acc_per_bin(validation_pred, validation_labels)

    accuracies =  {'training_accuracy': train_acc, 
                    'testing_accuracy': test_acc, 
                    'validation_accuracy': validation_acc}
    model_outputs = {'training_predictions': train_pred,
                    'training_labels': training_labels,
                    'testing_predictions': test_pred,
                    'testing_labels': testing_labels,
                    'validation_predictions': validation_pred,
                    'validation_labels': validation_labels}

    return accuracies, model_outputs


def save_CV_results(accuracies, out_dir, fname):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, fname)
    with open(out_file, 'wb') as a:
        pickle.dump(accuracies, a)


if __name__ == '__main__':
    pbounds = {
            'C': (log10(1e-8), log10(1e-3)),
            'epsilon': (log10(1e-10), log10(1e-3)),
    } #log uniform distribution

    Ab = sys.argv[1]
    clade_wise_results = {}
    for left_out_cluster in list(range(1, 13)):
        print(f'Ab = {Ab}, left out cluster = {left_out_cluster}')
        
        partial_fitting_function = partial(train_evaluate, 
                                    Ab = Ab, left_out_cluster = left_out_cluster)

        optimizer = BayesianOptimization(
            f = partial_fitting_function,
            pbounds = pbounds,
            random_state = 1
        )

        optimizer.maximize()
        print(f'Completed hyperparam optimisation: {Ab}, {left_out_cluster}')
        
        best_hyperparams = {}
        best_hyperparams['C'] = 10 ** optimizer.max['params']['C']
        best_hyperparams['epsilon'] = 10 ** optimizer.max['params']['epsilon']

        accuracies, model_outputs = fit_best_model(Ab, 
                                        left_out_cluster, 
                                        C = best_hyperparams['C'],
                                        epsilon = best_hyperparams['epsilon'])
        print(f'Fitted model with best hyperparams: {Ab}, {left_out_cluster}')

        clade_wise_results[left_out_cluster] = {
                                            'params': best_hyperparams,
                                            'accuracies': accuracies,
                                            'model_outputs': model_outputs}

        out_dir = f'kernel_learning/{Ab}/cluster_wise_cross_validation/SVR_results'
        fname = 'SVR_CV_results_mean_acc_per_bin.pkl'
        save_CV_results(clade_wise_results, out_dir, fname)
