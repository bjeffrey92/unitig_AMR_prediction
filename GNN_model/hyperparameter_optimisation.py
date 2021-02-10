import sys
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from GNN_model import train, utils, models

# Ab = sys.argv[1]
Ab = 'log2_azm_mic'

def train_evaluate(dropout, l2_alpha, lr, n_hid_1, n_hid_2):

    data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)

    training_data, testing_data, distances = train.load_data(data_dir, 
                                                            distances = True,
                                                            adj = False)

    torch.manual_seed(0)
    model = models.GraphConnectionsNN(distances = distances, 
                            n_hid_1 = int(n_hid_1), n_hid_2 = int(n_hid_2),  
                            out_dim = 1, dropout = dropout) 

    optimizer = optim.Adam(model.parameters(), lr = lr, 
                    weight_decay = l2_alpha) #weight decay is l2 loss
    loss_function = nn.MSELoss()

    training_metrics = utils.MetricAccumulator() 
    for epoch in range(50):
        epoch += 1
        
        model, epoch_results = train.train(training_data, model, 
                                    optimizer, epoch, loss_function, 
                                    utils.accuracy, 
                                    testing_data = testing_data)
        training_metrics.add(epoch_results)

        #if testing data accuracy has plateaued
        if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                if i < 0.1]) >= 10 and epoch > 30:
            break

    #returns average testing data loss in the last five epochs
    return sum(training_metrics.testing_data_loss[-5:])/5


if __name__ == '__main__':

    pbounds = {
            'dropout': (0.2, 0.7),
            'l2_alpha': (1e-4, 1e-3),
            'lr': (1e-5, 5e-4),
            'n_hid_1': (25, 100),
            'n_hid_2': (10, 50)
    }

    optimizer = BayesianOptimization(
        f = train_evaluate,
        pbounds = pbounds,
        random_state = 1,
    )

    optimizer.maximize(n_iter = 2)