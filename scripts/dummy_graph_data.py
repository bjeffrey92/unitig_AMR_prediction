import pickle
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
import logging
from scipy import sparse
from torch_sparse import SparseTensor

from scripts import gcn_vis
from GNN_model import utils
from GNN_model.models import GraphEdgeWiseAttention


logging.basicConfig()
logging.root.setLevel(logging.INFO)


def convert_to_tensor(matrix, torch_sparse_coo = True):
    
    shape = matrix.shape
    
    row = torch.LongTensor(matrix.row)
    col = torch.LongTensor(matrix.col)
    value = torch.Tensor(matrix.data)
    
    sparse_tensor = SparseTensor(row = row, col = col, 
                        value = value, sparse_sizes = shape)
    if torch_sparse_coo:
        return sparse_tensor.to_torch_sparse_coo_tensor()
    else:
        return sparse_tensor


def create_data(G, out_dir):
    n_nodes = G.number_of_nodes() 

    #extract adjacency matrix and convert to sparse tensor 
    adj = nx.adj_matrix(G)
    adj = adj.toarray()
    adj_sparse = sparse.csr_matrix(adj)
    adj_tensor = convert_to_tensor(adj_sparse.tocoo())

    #small graph 1 nodes
    # gene_1_nodes = [22, 5, 17, 18]
    # gene_2_nodes = [14, 1, 12, 10]

    #small graph 2 nodes
    gene_1_nodes = [116, 20, 6, 123]
    gene_2_nodes = [70, 69, 0, 49]

    labels = []
    features = []
    #make resistant samples with two resistance genes
    for i in range(250):
        for node in G:
            if node in gene_1_nodes + gene_2_nodes:
                G.nodes[node]['attr'] = 1.0 #label nodes of two 'genes'
            else:
                G.nodes[node]['attr'] = float(np.random.binomial(1, 0.5)) #noise
        labels.append(1)
        features.append(gcn_vis.get_attribute_vector(G))

    #make non resistant samples with random unitigs except for resistance unitigs
    for i in range(250):
        for node in G:
            if node in gene_1_nodes + gene_2_nodes:
                G.nodes[node]['attr'] = 0.0 #label nodes of two 'genes'
            else:
                G.nodes[node]['attr'] = float(np.random.binomial(1, 0.5)) #noise
        labels.append(0)
        features.append(gcn_vis.get_attribute_vector(G))

    indices = list(range(500))
    random.shuffle(indices)

    training_features = torch.as_tensor([features[i] for i in indices[:300]])
    testing_features = torch.as_tensor([features[i] for i in indices[300:]])
    training_labels = torch.as_tensor([labels[i] for i in indices[:300]])
    testing_labels = torch.as_tensor([labels[i] for i in indices[300:]])

    training_features = training_features.to(torch.float32)
    testing_features = testing_features.to(torch.float32)
    training_labels = training_labels.to(torch.float32)
    testing_labels = testing_labels.to(torch.float32)

    torch.save(adj_tensor, os.path.join(out_dir, 'unitig_adjacency_tensor.pt'))
    torch.save(training_features.to_sparse(), 
            os.path.join(out_dir, 'training_features.pt'))
    torch.save(testing_features.to_sparse(), 
            os.path.join(out_dir, 'testing_features.pt'))
    torch.save(training_labels, 
            os.path.join(out_dir, 'training_labels.pt'))
    torch.save(testing_labels, 
            os.path.join(out_dir, 'testing_labels.pt'))


def load_data(data_dir, normed_adj_matrix = False):
    adj = utils.load_adjacency_matrix(data_dir, normed_adj_matrix)
    
    training_features, training_labels = utils.load_training_data(data_dir)
    testing_features, testing_labels = utils.load_testing_data(data_dir)
    assert training_features.shape[1] == testing_features.shape[1], \
        'Dimensions of training and testing data not equal'
    
    training_data = utils.DataGenerator(training_features, training_labels, 
                                global_node = False)
    testing_data = utils.DataGenerator(testing_features, testing_labels, 
                                global_node = False)

    return training_data, testing_data, adj


def demo_perfect_attention(features, adj_tensor, G):
    #small graph 2 nodes
    gene_1_nodes = [116, 20, 6, 123]
    gene_2_nodes = [70, 69, 0, 49]

    r = adj_tensor.indices()[0].tolist()
    c = adj_tensor.indices()[1].tolist()

    indices = []
    for i in gene_1_nodes:
        l = [i for i, x in enumerate(r) if x in gene_1_nodes]
        indices += [i for i, x in enumerate(c) if x in gene_1_nodes and i in l]
    for i in gene_2_nodes:
        l = [i for i, x in enumerate(r) if x in gene_2_nodes]
        indices += [i for i, x in enumerate(c) if x in gene_2_nodes and i in l]

    values =  torch.Tensor([x if i in indices else 0 for i, x in 
                    enumerate(adj_tensor.values().tolist())])

    #the ideal weights for the learned adjacency matrix
    adj_perf = SparseTensor(row = torch.LongTensor(r), 
                            col = torch.LongTensor(c),
                            value = values).to_torch_sparse_coo_tensor()

    features_hat = torch.sparse.mm(adj_perf, features)
    G_hat = gcn_vis.set_attributes(G,  features_hat.squeeze(1).tolist())

    # gcn_vis.plot_graph(G_hat, )

def accuracy(predictions, labels):
    predictions = torch.round(predictions)
    return int(sum(labels == predictions))/len(predictions) * 100


def epoch_(model, data):
    data.reset_generator()
   
    outputs = [None] * data.n_samples
    for i in range(data.n_samples):
        features = data.next_sample()[0]
        outputs[i] = model(features).unsqueeze(0)

    output_tensor = torch.cat(outputs, dim = 0)
    
    return output_tensor


def test(data, model, adj, loss_function, accuracy):
    data.reset_generator()
    model.train(False)
    
    with torch.no_grad():
        output = epoch_(model, data)
    loss = float(loss_function(output, data.labels))
    acc = accuracy(output, data.labels)

    return loss, acc


def train(data, model, optimizer, adj, epoch, 
        loss_function, testing_data = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    data.shuffle_samples()

    output = epoch_(model, data)
    loss_train = loss_function(output, data.labels)
    acc_train = accuracy(output, data.labels)

    if testing_data:
        loss_test, acc_test = test(testing_data, model, 
                                    adj, loss_function, accuracy)
    else:
        loss_test = 'N/A'
        acc_test = 'N/A'        

    loss_train.backward(retain_graph = True)
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


if __name__ == '__main__':
    data_dir = 'dummy_data/small_graph_2/'
    
    # with open('dummy_data/small_graph_2.pkl', 'rb') as a:
    #     G = pickle.load(a)
    # mapping = {sorted(G)[i]:i for i in range(len(G))}
    # G = nx.relabel_nodes(G, mapping)
    # create_data(G, data_dir)

    training_data, testing_data, adj = load_data(data_dir)

    model = GraphEdgeWiseAttention(adj, training_data.n_nodes, 1)
    optimizer = optim.Adam(model.parameters(), lr = 0.001, 
                        weight_decay = 5e-4) 
    loss_function = nn.BCELoss()

    training_metrics = utils.MetricAccumulator() 
    for epoch in range(500):
        epoch += 1
    
        model, epoch_results = train(training_data, model, 
                                    optimizer, adj, epoch, loss_function,
                                    testing_data)

        training_metrics.add(epoch_results)