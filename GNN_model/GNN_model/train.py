#!/usr/bin/env python3
import argparse

import time
import logging
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from GNN_model import utils
from GNN_model.models import GraphConnectionsNN, GCNPerNode


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def epoch_(model, data, adj = None):
    data.reset_generator()
   
    outputs = [None] * data.n_samples
    for i in range(data.n_samples):
        features = data.next_sample()[0]
        outputs[i] = model(features, adj).unsqueeze(0)

    output_tensor = torch.cat(outputs, dim = 0)
    
    return output_tensor


def batch_train(data, model, optimizer, epoch, loss_function, accuracy, 
            adj = None, testing_data = None, batch_size = 32):
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
        loss_test, acc_test = test(testing_data, model, 
                                    loss_function, accuracy, adj)
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


def train(data, model, optimizer, epoch, loss_function, accuracy,
        adj = None, l1_alpha = None, testing_data = None, 
        validation_data = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = epoch_(model, data, adj)
    loss_train = loss_function(output, data.labels)
    if l1_alpha is not None: #apply l1 regularisation
        L1_reg = torch.as_tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if name.endswith('weight'):
                L1_reg = L1_reg + torch.norm(param, 1)
        regularised_loss_train = loss_train + l1_alpha * L1_reg
    else:
        regularised_loss_train = loss_train

    acc_train = accuracy(output, data.labels)
    
    if testing_data:
        loss_test, acc_test = test(testing_data, model, 
                                    loss_function, accuracy, adj)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'
    if validation_data:
        loss_val, acc_val = test(validation_data, model, 
                                    loss_function, accuracy, adj)
    else:
        loss_val = 'N/A'
        acc_val = 'N/A'
    
    regularised_loss_train.backward()
    optimizer.step()
    loss_train = float(loss_train) #to write it to file

    logging.info(f'Epoch {epoch} complete\n' + \
                f'\tTime taken = {time.time() - t}\n' + \
                f'\tTraining Data Loss = {loss_train}\n' + \
                f'\tTraining Data Accuracy = {acc_train}\n'
                f'\tTesting Data Loss = {loss_test}\n' + \
                f'\tTesting Data Accuracy = {acc_test}\n'
                f'\tValidation Data Loss = {loss_val}\n' + \
                f'\tValidation Data Accuracy = {acc_val}\n'
                )

    return model, (loss_train, acc_train, loss_test, acc_test, loss_val, acc_val)


def test(data, model, loss_function, accuracy, adj = None):
    data.reset_generator()
    model.train(False)
    
    with torch.no_grad():
        output = epoch_(model, data, adj)
    loss = float(loss_function(output, data.labels))
    acc = accuracy(output, data.labels)

    return loss, acc


def load_data(data_dir: str, distances: bool, adj: bool, 
            left_out_clade = None, left_out_cluster = None):
    if distances == adj:
        raise ValueError('One of distances or adj must equal True')
    if left_out_clade is not None and left_out_cluster is not None:
        raise ValueError(
            'Only one of clusters  or clades can be used for cross validation')

    training_data = utils.load_training_data(data_dir)
    testing_data = utils.load_testing_data(data_dir)
    training_metadata, testing_metadata = utils.load_metadata(data_dir) #for CV split

    if left_out_clade is not None:
        logging.info(
            f'Formatting data for model with clade {left_out_clade} left out')
        training_indices = training_metadata.loc[
                            training_metadata.Clade != left_out_clade].index
        testing_indices = testing_metadata.loc[
                            testing_metadata.Clade != left_out_clade].index
        validation_indices_1 = training_metadata.loc[
                            training_metadata.Clade == left_out_clade].index #extract data from training set
        validation_indices_2 = testing_metadata.loc[
                            testing_metadata.Clade == left_out_clade].index #extract data from testing set
    if left_out_cluster is not None:
        logging.info(
            f'Formatting data for model with cluster {left_out_cluster} left out')
        training_indices = training_metadata.loc[
                            training_metadata.clusters != left_out_cluster].index
        testing_indices = testing_metadata.loc[
                            testing_metadata.clusters != left_out_cluster].index
        validation_indices_1 = training_metadata.loc[
                            training_metadata.clusters == left_out_cluster].index #extract data from training set
        validation_indices_2 = testing_metadata.loc[
                            testing_metadata.clusters == left_out_cluster].index #extract data from testing set

    training_features = torch.index_select(training_data[0], 0, 
                                        torch.as_tensor(training_indices))
    training_labels = torch.index_select(training_data[1], 0, 
                                        torch.as_tensor(training_indices))
    testing_features = torch.index_select(testing_data[0], 0,
                                        torch.as_tensor(testing_indices))
    testing_labels = torch.index_select(testing_data[1], 0, 
                                        torch.as_tensor(testing_indices))
    validation_features = torch.cat([
                            torch.index_select(training_data[0], 0, 
                                        torch.as_tensor(validation_indices_1)),
                            torch.index_select(testing_data[0], 0, 
                                        torch.as_tensor(validation_indices_2))
                            ])
    validation_labels = torch.cat([
                            torch.index_select(training_data[1], 0, 
                                        torch.as_tensor(validation_indices_1)),
                            torch.index_select(testing_data[1], 0,
                                        torch.as_tensor(validation_indices_2))
                            ])

    assert all([training_features.shape[1] == testing_features.shape[1],
                validation_features.shape[1] == testing_features.shape[1]]), \
        'Dimensions of training, testing and validation data are not equal'

    training_data = utils.DataGenerator(training_features, training_labels,
                                global_node = False)
    testing_data = utils.DataGenerator(testing_features, testing_labels, 
                                global_node = False)
    validation_data = utils.DataGenerator(validation_features, validation_labels, 
                                global_node = False)

    if distances:
        distances = utils.load_distances(data_dir)
        return training_data, testing_data, validation_data, distances
    elif adj:
        adj = utils.load_adjacency_matrix(data_dir, False)
        return training_data, testing_data, validation_data, adj


def main(args):    

    Ab = 'log2_cro_mic'
    for left_out_clade in [1,2,3]:
        start_time = time.time()

        data_dir = os.path.join('data/model_inputs/freq_5_95', Ab)

        all_data = load_data(data_dir, distances = False, adj = True, 
                        left_out_clade = left_out_clade)
        training_data, testing_data, validation_data, adj = all_data
        # adj = None

        torch.manual_seed(0)
        model = GCNPerNode(adj.shape[0], 50, 50, 1, 0.3) 

        optimizer = optim.Adam(model.parameters(), lr = 0.0001, 
                        weight_decay = 5e-4) #weight decay is l2 loss
        loss_function = nn.MSELoss()

        summary_file = f'{Ab}_clade_{left_out_clade}_left_out_GCNPerNode.tsv'

        #records training metrics and logs the gradient after each epoch
        training_metrics = utils.MetricAccumulator() 

        start_time = time.time()
        for epoch in range(300):
            epoch += 1
            
            model, epoch_results = train(training_data, model, 
                                        optimizer, epoch, loss_function, 
                                        utils.accuracy, adj = adj, l1_alpha = None, 
                                        testing_data = testing_data,
                                        validation_data = validation_data)
            
            training_metrics.add(epoch_results)
            if epoch >= 20:
                training_metrics.log_gradients(epoch)
            utils.write_epoch_results(epoch, epoch_results, summary_file)
        
            if len([i for i in training_metrics.testing_data_acc_grads[-10:] \
                    if i < 0.1]) >= 10 and epoch > 50:
                logging.info('Gradient of testing data accuracy appears to have plateaued, terminating early')
                break

        logging.info(f'Model Fitting Complete. Time elapsed {time.time() - start_time}')

        torch.save(model, f'{Ab}_clade_{left_out_clade}_left_out_GCNPerNode.pt')