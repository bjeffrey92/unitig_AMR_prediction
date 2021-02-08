import os 
import pickle 
import numpy as np
import itertools
import hyperopt
import matplotlib.pyplot as plt
from functools import lru_cache
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from torch import tensor

from lasso_model.utils import load_training_data, load_testing_data
from GNN_model.utils import R_or_S, accuracy


@lru_cache(maxsize = None)
def load_data(Ab, iterations = 5):
    root_dir = 'data/model_inputs/freq_5_95/'
    data_dir = os.path.join(root_dir, Ab, 'gwas_filtered')

    training_labels = load_training_data(data_dir)[1].tolist()
    testing_labels = load_testing_data(data_dir)[1].tolist()

    with open(f'kernel_learning/{Ab}/{iterations}_iter_k_train.pkl', 'rb') as a:
        k_train = pickle.load(a)
    with open(f'kernel_learning/{Ab}/{iterations}_iter_k_test.pkl', 'rb') as a:
        k_test = pickle.load(a)

    return k_train, training_labels, k_test, testing_labels


# def fit_and_evaluate(C, epsilon, 
#                     k_train = k_train, k_test = k_test,
#                     training_labels = training_labels, 
#                     testing_labels = testing_labels):
    
#     model = SVR(C = C, epsilon = epsilon, kernel = 'precomputed')
#     model.fit(k_train, training_labels)

#     train_acc = accuracy(tensor(model.predict(k_train)), 
#                         tensor(training_labels))
#     test_acc = accuracy(tensor(model.predict(k_test)), 
#                         tensor(testing_labels))

#     return train_acc, test_acc


def train_evaluate(params, Ab):

    print(params)
    C = params['C']
    epsilon = params['epsilon']

    k_train, training_labels, k_test, testing_labels = load_data(Ab)

    model = SVR(C = C, epsilon = epsilon, kernel = 'precomputed')
    model.fit(k_train, training_labels)

    train_acc = accuracy(tensor(model.predict(k_train)), 
                        tensor(training_labels))
    test_acc = accuracy(tensor(model.predict(k_test)), 
                        tensor(testing_labels))
    print(f'{C, epsilon}\n', 
        f'Training Data Accuracy = {train_acc}\n',
        f'Testing Data Accuracy = {test_acc}\n')

    return mean_squared_error(testing_labels, model.predict(k_test))


def optimise_hps(Ab):

    def objective(params, Ab = Ab):
        return train_evaluate(params, Ab)

    space = {'C': hyperopt.hp.uniform('C', 1e-08, 1e-05),
            'epsilon': hyperopt.hp.uniform('epsilon', 1e-10, 1e-3)
            }
    
    trials = hyperopt.Trials()
    _ = hyperopt.fmin(objective, 
                    space, 
                    trials = trials, 
                    algo = hyperopt.tpe.suggest,
                    max_evals = 1000)

    return trials


def plot_chains(trials, fig_name):
    '''
    Plots markov chain of each hyperparam and the loss
    '''
    x = list(range(len(trials.losses()))) #indices

    #plots for the number of parameters plus the loss
    fig, axs = plt.subplots(len(trials.vals) + 1, sharex = True) 

    axs[0].plot(x, trials.losses())
    axs[0].set_ylabel('loss')
    for i, key in enumerate(trials.vals):
        y = trials.vals[key]
        axs[i + 1].plot(x, y)
        axs[i + 1].set_ylabel(key)

    fig.savefig(fig_name)


if __name__ == '__main__':
    Ab = 'log2_azm_mic' 
    trials = optimise_hps(Ab)