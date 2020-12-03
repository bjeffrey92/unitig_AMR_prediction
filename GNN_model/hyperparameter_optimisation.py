import time
import pickle
import hyperopt
import os
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from functools import lru_cache
from GNN_model import train, utils, models

logging.basicConfig()
logging.root.setLevel(logging.INFO)


@lru_cache(maxsize = 1)
def load_data(data_dir, normed_adj_matrix = True, global_node = False):
    adj = utils.load_adjacency_matrix(data_dir, normed_adj_matrix)
    
    #adds node which is connected to all others
    #required for gcn without max pooling layers
    if global_node:
        adj = utils.add_global_node(adj)
    
    training_features, training_labels = utils.load_training_data(data_dir)
    testing_features, testing_labels = utils.load_testing_data(data_dir)
    assert training_features.shape[1] == testing_features.shape[1], \
        'Dimensions of training and testing data not equal'
    
    training_data = utils.DataGenerator(training_features, training_labels, 
                                global_node = global_node)
    testing_data = utils.DataGenerator(testing_features, testing_labels, 
                                global_node = global_node)

    return training_data, testing_data, adj


def train_evaluate(Ab, params):

    n_hid_1 = round(params['n_hid_1'])
    n_hid_2 = round(params['n_hid_2'])
    n_hid_3 = round(params['n_hid_3'])
    dropout = params['dropout']
    l1_alpha = params['l1_alpha']
    l2_alpha = params['l2_alpha']

    data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)

    training_data, testing_data, adj = load_data(data_dir)

    model = models.GCNPerNode(n_feat = training_data.n_nodes, n_hid_1 = n_hid_1, 
                        n_hid_2 = n_hid_2, n_hid_3 = n_hid_3, 
                        out_dim = 1, dropout = dropout)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, 
                    weight_decay = l2_alpha) #weight decay is l2 loss
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator() 
    for epoch in range(50):
        epoch += 1
        
        model, epoch_results = train.train(training_data, model, 
                                    optimizer, adj, epoch, loss_function, 
                                    utils.accuracy, l1_alpha, 
                                    testing_data = testing_data)
        training_metrics.add(epoch_results)

        #if testing data accuracy has plateaued
        if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                if i < 0.1]) >= 10 and epoch > 30:
            break

    #returns average testing data loss in the last five epochs
    return sum(training_metrics.testing_data_loss[-5:])/5


def optimise_hps(Ab):

    def objective(params, Ab = Ab):
        return train_evaluate(Ab, params)

    space = {'n_hid_1': hyperopt.hp.uniform('n_hid_1', 10, 80),
            'n_hid_2': hyperopt.hp.uniform('n_hid_2', 10, 80),
            'n_hid_3': hyperopt.hp.uniform('n_hid_3', 0, 50),
            'dropout': hyperopt.hp.uniform('dropout', 0.2, 0.6),
            'l1_alpha': hyperopt.hp.uniform('l1_alpha', 0.01, 0.15),
            'l2_alpha': hyperopt.hp.uniform('l2_alpha', 1e-4, 1e-3)
            }
    
    trials = hyperopt.Trials()
    _ = hyperopt.fmin(objective, 
                    space, 
                    trials = trials, 
                    algo = hyperopt.tpe.suggest,
                    max_evals = 50)

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

    start_time = time.time()
    trials = optimise_hps(Ab)
    logging.info(f'Optimisation loop complete, time take = {time.time() - start_time}')

    with open(f'{Ab}_gnn_hp_optimisation_trials.pkl', 'wb') as a:
        pickle.dump(trials, a)

    plot_chains(trials, f'{Ab}_gnn_hp_optimisation_markov_chains.png')