#!/usr/bin/env python3
import argparse

import time
import logging
import math
import os
from functools import lru_cache

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model.utils import load_training_data, load_testing_data, \
                            load_adjacency_matrix, accuracy,\
                            logcosh, write_epoch_results, DataGenerator, \
                            MetricAccumulator, add_global_node
from GNN_model.models import GCNMaxPooling, GCNPerNode, PreConvolutionNN


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
        loss_function, accuracy, testing_data = None, batch_size = 32):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    data.reset_generator()
    data.shuffle_samples()

    batch_losses = []
    batch_accuracies = []
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
        acc_train = accuracy(output_tensor, labels)
        batch_losses.append(float(loss_train))
        batch_accuracies.append(acc_train)
        
        loss_train.backward()
        optimizer.step()    
    
    if testing_data:
        loss_test, acc_test = test(testing_data, model, adj, 
                                    loss_function, accuracy)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'
    
    mean_loss_train = sum(batch_losses)/len(batch_losses)
    mean_acc_train = sum(batch_accuracies)/len(batch_accuracies)
    logging.info(f'Epoch {epoch} complete\n' + \
                f'\tTime taken = {time.time() - t}\n' + \
                f'\tMean Training Data Loss = {mean_loss_train}\n' + \
                f'\tMean Training Data Accuracy = {mean_acc_train}\n'
                f'\tTesting Data Loss = {loss_test}\n' + \
                f'\tTesting Data Accuracy = {acc_test}\n'
                )

    return model, (mean_loss_train, mean_acc_train, loss_test, acc_test)


def train(data, model, optimizer, adj, epoch, 
        loss_function, accuracy, l1_alpha = None, testing_data = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = epoch_(model, data, adj)
    loss_train = loss_function(output, data.labels)
    if l1_alpha is not None: #apply l1 regularisation
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if name.endswith('weight'):
                L1_reg = L1_reg + torch.norm(param, 1)
        regularised_loss_train = loss_train + l1_alpha * L1_reg
    else:
        regularised_loss_train = loss_train

    acc_train = accuracy(output, data.labels)
    
    if testing_data:
        loss_test, acc_test = test(testing_data, model, 
                                    adj, loss_function, accuracy)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'
    
    regularised_loss_train.backward()
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


def test(data, model, adj, loss_function, accuracy):
    data.reset_generator()
    model.train(False)
    
    with torch.no_grad():
        output = epoch_(model, data, adj)
    loss = float(loss_function(output, data.labels))
    acc = accuracy(output, data.labels)

    return loss, acc

@lru_cache(maxsize = 1)
def load_data(data_dir, k = None, 
            normed_adj_matrix = True, global_node = False):
    training_features, training_labels = load_training_data(data_dir, k = k)
    testing_features, testing_labels = load_testing_data(data_dir, k = k)
    
    if k is not None:
        training_data = DataGenerator(training_features, training_labels, 
                                    global_node = global_node,
                                    pre_convolved = True)
        testing_data = DataGenerator(testing_features, testing_labels, 
                                    global_node = global_node,
                                    pre_convolved = True)
        return training_data, testing_data
    else:
        assert training_features.shape[1] == testing_features.shape[1], \
            'Dimensions of training and testing data not equal'
        training_data = DataGenerator(training_features, training_labels, 
                                    global_node = global_node)
        testing_data = DataGenerator(testing_features, testing_labels, 
                                    global_node = global_node)
        adj = load_adjacency_matrix(data_dir, normed_adj_matrix)
        
        #adds node which is connected to all others
        #required for gcn without max pooling layers
        if global_node:
            adj = add_global_node(adj)
        
        return training_data, testing_data, adj


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


def train_multiple():
    root_dir = 'data/model_inputs/freq_5_95'

    Abs = ['log2_azm_mic',
        'log2_cip_mic',
        'log2_cro_mic',
        'log2_cfx_mic']

    Ab_l1_alphas = {'log2_azm_mic': 0.1,
                    'log2_cip_mic': 0.1,
                    'log2_cro_mic': 0.01,
                    'log2_cfx_mic': 0.01}

    for Ab in Abs:
        data_dir = os.path.join(root_dir, Ab)

        training_data, testing_data, adj = load_data(data_dir, 
                                                    global_node = True)

        torch.manual_seed(0)
        model = GCNPerNode(n_feat = training_data.n_nodes, n_hid_1 = 50, 
                        n_hid_2 = 50, out_dim = 1, dropout = 0.3)

        optimizer = optim.Adam(model.parameters(), lr = 0.001, 
                        weight_decay = 5e-4) #weight decay is l2 regularisation
        # loss_function = logcosh
        loss_function = nn.MSELoss()

        summary_file = Ab + '_l1_regularised.tsv'

        l1_alpha = Ab_l1_alphas[Ab]

        #records training metrics and logs the gradient after each epoch
        training_metrics = MetricAccumulator() 

        start_time = time.time()
        for epoch in range(300):
            epoch += 1
            

            model, epoch_results = train(training_data, model, 
                                        optimizer, adj, epoch, loss_function, 
                                        accuracy, l1_alpha, testing_data)
            
            training_metrics.add(epoch_results)
            if epoch >= 20:
                training_metrics.log_gradients(epoch)
            write_epoch_results(epoch, epoch_results, summary_file)
        
            if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                 if i < 0.1]) >= 10 and epoch > 50:
                logging.info('Gradient of testing data accuracy appears to have plateaued, terminating early')
                break

        logging.info(f'Model Fitting Complete. Time elapsed {time.time() - start_time}')

        torch.save(model, Ab + '_l1_regularised.pt')


def main(args):    

    Ab = 'log2_azm_mic'
    data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)
    k = 3

    training_data, testing_data = load_data(data_dir, k = k)
    adj = None #for consistency

    n_neighbours = training_data.next_sample()[0].shape[1]
    torch.manual_seed(0)
    model = PreConvolutionNN(n_nodes = training_data.n_nodes, 
                            n_neighbours = n_neighbours,
                            out_dim = 1, dropout = 0.3) #add one to k as 1st neighbour is

    # training_data, testing_data, adj = load_data(data_dir)
    # torch.manual_seed(0)
    # model = GCNPerNode(n_feat = training_data.n_nodes, n_hid_1 = 50, 
    #                     n_hid_2 = 50, out_dim = 1, dropout = 0.3)

    optimizer = optim.Adam(model.parameters(), lr = 0.0001, 
                    weight_decay = 5e-4) #weight decay is l2 loss
    loss_function = nn.MSELoss()

    summary_file = f'{Ab}_preconvolution_NN.tsv'

    #records training metrics and logs the gradient after each epoch
    training_metrics = MetricAccumulator() 

    start_time = time.time()
    for epoch in range(300):
        epoch += 1
        
        model, epoch_results = train(training_data, model, 
                                    optimizer, adj, epoch, loss_function, 
                                    accuracy, l1_alpha = None, 
                                    testing_data = testing_data)
        
        training_metrics.add(epoch_results)
        if epoch >= 20:
            training_metrics.log_gradients(epoch)
        write_epoch_results(epoch, epoch_results, summary_file)
    
        if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                if i < 0.1]) >= 10 and epoch > 50:
            logging.info('Gradient of testing data accuracy appears to have plateaued, terminating early')
            break

    logging.info(f'Model Fitting Complete. Time elapsed {time.time() - start_time}')

    torch.save(model, f'{Ab}_preconvolution_NN.pt')


# if __name__ == '__main__':
#     main(parse_args())
