#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from numpy import linspace

from lasso_model.utils import load_training_data, load_testing_data, accuracy

import logging
import warnings

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def CV(training_features, training_labels, 
        testing_features, testing_labels, alphas: list) -> dict:
    
    accuracy_dict = {i:None for i in alphas}

    for a in alphas:
        logging.info(f'Fitting model for alpha = {a}')

        max_iter = 1000
        fitted = False
        while not fitted:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
    
                reg = Lasso(alpha = a, random_state = 0, max_iter = max_iter)
                reg.fit(training_features, training_labels)
                
                if w and issubclass(w[0].category, ConvergenceWarning):
                    logging.warning(f'Failed to converge with max_iter = {max_iter}, adding 1000 more')
                    max_iter += 1000
                elif len(w) > 1:
                    for warning in w:
                        logging.error(warning.category)
                    raise Exception
                else:
                    fitted = True

        logging.info(f'alpha = {a} model fitted, generating predictions')

        training_predictions = reg.predict(training_features)
        testing_predictions = reg.predict(testing_features)

        training_accuracy = accuracy(torch.tensor(training_predictions), 
                                    training_labels)
        testing_accuracy = accuracy(torch.tensor(testing_predictions), 
                                    testing_labels)

        accuracy_dict[a] = [training_accuracy, testing_accuracy]

    return accuracy_dict


def plot_results(accuracy_dict, fname):

    training_acc = [i[0] for i in accuracy_dict.values()]
    testing_acc = [i[1] for i in accuracy_dict.values()]

    plt.scatter(list(accuracy_dict.keys()), training_acc)
    plt.scatter(list(accuracy_dict.keys()), testing_acc)

    plt.savefig(fname)

if __name__ == '__main__':
    data_dir = 'model_inputs/dev_set'

    training_features, training_labels = load_training_data(data_dir)
    testing_features, testing_labels = load_testing_data(data_dir)

    alphas = linspace(0.01, 0.1, 10)

    accuracy_dict = CV(training_features, training_labels, 
                testing_features, testing_labels, alphas)

    plot_results(accuracy_dict, fname)