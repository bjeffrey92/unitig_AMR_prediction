#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from lasso_model.utils import load_training_data, load_testing_data
from GNN_model.utils import R_or_S, breakpoints

import pickle
import os 
import logging
import warnings
import numpy as np

logging.basicConfig()
logging.root.setLevel(logging.INFO)

def CV(training_features, training_labels, 
        testing_features, testing_labels, cs: list) -> dict:
    
    accuracy_dict = {i:None for i in cs}

    for c in cs:
        logging.info(f'Fitting model for alpha = {c}')

        max_iter = 1000
        fitted = False
        while not fitted:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
    
                reg = LogisticRegression(C = c, random_state = 0, 
                                    max_iter = max_iter)
                reg.fit(training_features, training_labels)
                
                if len(w) > 1:
                    for warning in w:
                        logging.error(warning.category, warning)
                    raise Exception
                elif w and issubclass(w[0].category, ConvergenceWarning):
                    logging.warning(
            f'Failed to converge with max_iter = {max_iter}, adding 1000 more')
                    max_iter += 1000
                else:
                    fitted = True

        logging.info(f'alpha = {c} model fitted, generating predictions')

        training_predictions = reg.predict(training_features)
        testing_predictions = reg.predict(testing_features)

        training_accuracy = accuracy_score(training_predictions, 
                                            training_labels) * 100
        testing_accuracy = accuracy_score(testing_predictions, 
                                            testing_labels) * 100

        accuracy_dict[c] = [training_accuracy, testing_accuracy]

    return accuracy_dict


if __name__ == '__main__':
    root_dir = 'data/model_inputs/freq_5_95/'
    Ab = 'log2_azm_mic'
    data_dir = os.path.join(root_dir, Ab)

    training_features, training_labels = load_training_data(data_dir)
    testing_features, testing_labels = load_testing_data(data_dir)

    training_labels = R_or_S(training_labels.tolist(), 
                            breakpoints[Ab.split('_')[1]])
    testing_labels = R_or_S(testing_labels.tolist(), 
                            breakpoints[Ab.split('_')[1]])

    cs = np.linspace(1, 20, 5)
    accuracy_dict = CV(training_features, training_labels, 
                    testing_features, testing_labels, cs)
