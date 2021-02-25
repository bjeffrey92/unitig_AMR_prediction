import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from numpy import linspace, sort, array_equal

from lasso_model.utils import load_training_data, load_testing_data, \
                            load_adjacency_matrix, mean_acc_per_bin, convolve, \
                            load_metadata

import pickle
import os 
import logging
import warnings

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def fit_model_by_grid_search(training_features, training_labels, 
                            testing_features, testing_labels, 
                            validation_features, validation_labels, 
                            alphas: list, adj = None) -> dict:
    '''
    Training and testing data is random split of part of the tree, 
    validation data is from a separate clade
    '''

    accuracy_dict = {i:None for i in alphas}

    if adj is not None:
        training_features = convolve(training_features, adj)
        testing_features = convolve(testing_features, adj)

    for a in alphas:
        logging.info(f'Fitting model for alpha = {a}')

        max_iter = 1000
        fitted = False
        while not fitted:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
    
                reg = Lasso(alpha = a, random_state = 0, max_iter = max_iter)
                reg.fit(training_features, training_labels)
                
                if len(w) > 1:
                    for warning in w:
                        logging.error(warning.category)
                    raise Exception
                elif w and issubclass(w[0].category, ConvergenceWarning):
                    logging.warning(f'Failed to converge with max_iter = {max_iter}, adding 1000 more')
                    max_iter += 1000
                else:
                    fitted = True

        logging.info(f'alpha = {a} model fitted, generating predictions')

        training_predictions = reg.predict(training_features)
        testing_predictions = reg.predict(testing_features)
        validation_predictions = reg.predict(validation_features)

        training_accuracy = mean_acc_per_bin(
                                    torch.tensor(training_predictions), 
                                    training_labels)
        testing_accuracy = mean_acc_per_bin(
                                    torch.tensor(testing_predictions), 
                                    testing_labels)
        validation_accuracy = mean_acc_per_bin(
                                        torch.tensor(validation_predictions),
                                        validation_labels)

        accuracy_dict[a] = {'training_accuracy': training_accuracy, 
                            'testing_accuracy': testing_accuracy, 
                            'validation_accuracy': validation_accuracy}

    return accuracy_dict


def leave_one_out_CV(training_data, testing_data, 
                    training_metadata, testing_metadata):

    clades = training_metadata.clusters.unique()
    assert array_equal(sort(clades), sort(testing_metadata.clusters.unique())), \
        'Different clades found in training and testing metadata'

    alphas = linspace(0.01, 0.1, 5)
    
    results_dict = {}
    for left_out_clade in clades:
        logging.info(
            f'Formatting data for model with clade {left_out_clade} left out')
        training_indices = training_metadata.loc[
                            training_metadata.clusters != left_out_clade].index
        testing_indices = testing_metadata.loc[
                            testing_metadata.clusters != left_out_clade].index
        validation_indices_1 = training_metadata.loc[
                            training_metadata.clusters == left_out_clade].index #extract data from training set
        validation_indices_2 = testing_metadata.loc[
                            testing_metadata.clusters == left_out_clade].index #extract data from testing set

        training_features = torch.index_select(training_data[0], 0, 
                                            torch.tensor(training_indices))
        training_labels = torch.index_select(training_data[1], 0, 
                                            torch.tensor(training_indices))
        testing_features = torch.index_select(testing_data[0], 0,
                                            torch.tensor(testing_indices))
        testing_labels = torch.index_select(testing_data[1], 0, 
                                            torch.tensor(testing_indices))
        validation_features = torch.cat([
                                torch.index_select(training_data[0], 0, 
                                            torch.tensor(validation_indices_1)),
                                torch.index_select(testing_data[0], 0, 
                                            torch.tensor(validation_indices_2))
                                ])
        validation_labels = torch.cat([
                                torch.index_select(training_data[1], 0, 
                                            torch.tensor(validation_indices_1)),
                                torch.index_select(testing_data[1], 0,
                                            torch.tensor(validation_indices_2))
                                ])

        accuracy_dict = fit_model_by_grid_search(training_features, training_labels, 
                                testing_features, testing_labels, 
                                validation_features, validation_labels,
                                alphas)
        results_dict[left_out_clade] = accuracy_dict

    return results_dict


def save_output(accuracy_dict, fname):
    with open(fname, 'wb') as a:
        pickle.dump(accuracy_dict, a)


def plot_results(accuracy_dict, fname):

    training_acc = [i['training_accuracy'] for i in accuracy_dict.values()]
    testing_acc = [i['testing_accuracy'] for i in accuracy_dict.values()]
    validation_acc = [i['validation_accuracy'] for i in accuracy_dict.values()]

    plt.clf()

    plt.scatter(list(accuracy_dict.keys()), training_acc, 
                label = 'Training Data Accuracy')
    plt.scatter(list(accuracy_dict.keys()), testing_acc, 
                label = 'Testing Data Accuracy')
    plt.scatter(list(accuracy_dict.keys()), validation_acc, 
                label = 'Left Out Clade Accuracy')
    plt.legend(loc = 'lower left')

    plt.savefig(fname)


if __name__ == '__main__':
    root_dir = 'data/model_inputs/freq_5_95/'

    outcomes = os.listdir(root_dir)
    for outcome in outcomes:
        data_dir = os.path.join(root_dir, outcome)
        results_dir = \
            'lasso_model/results/linear_model_results/mean_accuracy_per_bin'

        training_data = load_training_data(data_dir)
        testing_data = load_testing_data(data_dir)
        training_metadata, testing_metadata = load_metadata(data_dir)        

        results_dict = leave_one_out_CV(training_data, 
                                        testing_data,
                                        training_metadata, 
                                        testing_metadata)

        fname = os.path.join(results_dir, outcome + '_CV_lasso_predictions.pkl')
        save_output(results_dict, fname)

        for left_out_clade in results_dict.keys():
            fname = os.path.join(results_dir,
                outcome + \
                f'_validation_cluster_{left_out_clade}_lasso_predictions.png')
            plot_results(results_dict[left_out_clade], fname)