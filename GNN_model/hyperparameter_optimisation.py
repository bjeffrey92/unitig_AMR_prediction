import time
import pickle
import hyperopt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from GNN_model import train, utils, models

logging.basicConfig()
logging.root.setLevel(logging.INFO)


def train_evaluate(Ab, params):

    print(params) 

    dropout = params['dropout']
    l2_alpha = params['l2_alpha']
    lr = params['lr']

    data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)
    k = 4

    training_data, testing_data = train.load_data(data_dir, k = k)
    adj = None

    torch.manual_seed(0)
    model = models.PreConvolutionNN(k = k + 1, n_nodes = training_data.n_nodes, 
                            out_dim = 1, dropout = dropout) #add one to k as 1st neighbour is

    optimizer = optim.Adam(model.parameters(), lr = lr, 
                    weight_decay = l2_alpha) #weight decay is l2 loss
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator() 
    for epoch in range(50):
        epoch += 1
        
        model, epoch_results = train.train(training_data, model, 
                                    optimizer, adj, epoch, loss_function, 
                                    utils.accuracy, l1_alpha = None, 
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

    space = {'dropout': hyperopt.hp.uniform('dropout', 0.2, 0.7),
            'l2_alpha': hyperopt.hp.uniform('l2_alpha', 1e-4, 1e-3),
            'lr': hyperopt.hp.uniform('lr', 1e-5, 5e-4)
            }
    
    trials = hyperopt.Trials()
    _ = hyperopt.fmin(objective, 
                    space, 
                    trials = trials, 
                    algo = hyperopt.tpe.suggest,
                    max_evals = 100)

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

    with open(f'{Ab}_preconvolution_trials.pkl', 'wb') as a:
        pickle.dump(trials, a)

    plot_chains(trials, f'{Ab}_preconvolution_hp_optimisation_markov_chains.png')