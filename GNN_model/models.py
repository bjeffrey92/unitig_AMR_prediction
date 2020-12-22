from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_model.layers import GraphConvolution, GraphConvolutionPerNode, \
                            PreSelectionConvolution


class _connectionSubModel(nn.Module):
    def __init__(self, in_dim, out_dim = 1):
        super(_connectionSubModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = torch.nn.Linear(in_dim, out_dim) #encodes weights and bias

    def forward(self, layer_input):
        output = self.lin(layer_input) #Y = WX^T + B
        return output 

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'        

class GraphConnectionsNN(nn.Module):
    def __init__(self, distances, n_hid_1, n_hid_2, out_dim, dropout):
        super(GraphConnectionsNN, self).__init__()

        self.distances = {str(k):v for k,v in distances.items()}
        self.dropout = dropout

        self.layer_1 = nn.ModuleDict({str(node):_connectionSubModel(len(path_lengths)) 
                            for node, path_lengths in distances.items()})
        self.layer_2 = nn.Linear(len(distances), n_hid_1)
        self.layer_3 = nn.Linear(n_hid_1, n_hid_2)
        self.layer_4 = nn.Linear(n_hid_2, out_dim)

    @lru_cache(maxsize = None)
    def _parse_inputs(self, model_input, i):
        return model_input[list(self.distances[i].keys())].squeeze(1)

    def forward(self, model_input, adj = None):
        self.latest_input = model_input
        x = torch.cat(
            [self.layer_1[i](self._parse_inputs(model_input, i)) 
                for i in self.distances.keys()],
            dim = 0)
        x = F.leaky_relu(x)
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.layer_2(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.layer_3(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = F.leaky_relu(self.layer_4(x))
        return out


class GraphEdgeWiseAttention(nn.Module):
    def __init__(self, adj, n_nodes, out_dim):
        super(GraphEdgeWiseAttention, self).__init__()
        
        self.n_nodes = n_nodes
        self.out_dim = out_dim
        adj = adj.coalesce()
        self.w = nn.Parameter(torch.empty(size=(len(adj.values()), 1)))
        nn.init.uniform_(self.w.data, 0, 1)
        self.adj = torch.sparse_coo_tensor(adj.indices(), self.w.squeeze(1)) 
        self.ones = torch.ones((out_dim,n_nodes))

    def forward(self, layer_input):
        x = torch.relu(torch.sparse.mm(self.adj, layer_input))
        out = torch.sigmoid(torch.mm(self.ones, x))[0][0]
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_nodes) + ' -> ' \
               + str(self.out_dim) + ')'

class GCNGlobalNode(nn.Module):
    def __init__(self, n_feat, conv_1, n_hid, out_dim, dropout):
        super(GCNGlobalNode, self).__init__()

        self.gc1 = GraphConvolution(n_feat, conv_1)
        # self.gc2 = GraphConvolution(conv_1, conv_2)
        self.linear1 = nn.Linear(conv_1, n_hid)
        self.linear2 = nn.Linear(n_hid, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))[-1] #select last node which is the global node
        # x = F.leaky_relu(self.gc2(x, adj))[-1] 
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = F.leaky_relu(self.linear2(x))[0]
        return out


class GCNMaxPooling(nn.Module):
    def __init__(self, n_feat, conv_1, conv_2, n_hid_1, n_hid_2, 
                out_dim, dropout):
        super(GCNMaxPooling, self).__init__()

        self.gc1 = GraphConvolution(n_feat, conv_1)
        
        #if more than one convolutional layer
        if conv_2 > 0:
            self.gc2 = GraphConvolution(n_feat, conv_2)
        else:
            self.gc2 = None
        
        self.linear1 = nn.Linear(n_feat, n_hid_1)
        
        #if more than one densely connected hidden layer
        if n_hid_2 > 0:
            self.linear2 = nn.Linear(n_hid_1, n_hid_2)
            self.linear3 = nn.Linear(n_hid_2, out_dim)
        else:
            self.linear2 = nn.Linear(n_hid_1, out_dim)
            self.linear3 = None
        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        
        if self.gc2 is not None:
            x, _ = torch.max(x, 1) #global max pooling
            x = x.unsqueeze(1)
            x = F.leaky_relu(self.gc2(x, adj))
        
        x, _ = torch.max(x, 1) #global max pooling
        x = x.unsqueeze(1)
        
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        
        x = F.leaky_relu(self.linear2(x))
        if self.linear3 is not None:
            F.dropout(x, self.dropout, inplace = True, training = True)
            x = F.leaky_relu(self.linear3(x))

        return x[0][0]


class GCNPerNode(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout, 
                n_hid_3 = 0):
        super(GCNPerNode, self).__init__()

        self.gc = GraphConvolutionPerNode(n_feat, n_hid_1)
        self.linear1 = nn.Linear(n_hid_1, n_hid_2)
        if n_hid_3 > 0:
            self.linear2 = nn.Linear(n_hid_2, n_hid_3)
            self.linear3 = nn.Linear(n_hid_3, out_dim)
        else:
            self.linear2 = nn.Linear(n_hid_2, out_dim)
            self.linear3 = None
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc(x, adj))
        # F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        if self.linear3 is not None:
            x = F.leaky_relu(self.linear2(x))
            F.dropout(x, self.dropout, inplace = True, training = True)
            out = F.leaky_relu(self.linear3(x)[0][0])
        else:
            out = F.leaky_relu(self.linear2(x)[0][0])
        return out


class PreConvolutionNN(nn.Module):
    def __init__(self, n_nodes, n_neighbours, dropout, out_dim = 1):
        super(PreConvolutionNN, self).__init__()

        self.conv = PreSelectionConvolution(n_nodes, n_neighbours)
        self.linear1 = nn.Linear(n_nodes, out_dim)
        self.dropout = dropout

    #adj included for consistency with training functions, simplifies training
    def forward(self, x, adj = None): 
        x = F.leaky_relu(self.conv(x)).unsqueeze(1)
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = self.linear1(x.transpose(0,1))[0][0]
        return out


class LinearConv(nn.Module):
    def __init__(self, k, n_nodes):
        super(LinearConv, self).__init__()
        self.linear1 = nn.Linear(k * n_nodes, 1)
    def forward(self, x, adj = None):
        x = x.flatten(0).unsqueeze(1)
        return self.linear1(x.transpose(0,1))[0][0]


class VanillaNN(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout):
        super(VanillaNN, self).__init__()

        self.linear1 = nn.Linear(n_feat, n_hid_1)
        self.linear2 = nn.Linear(n_hid_1, n_hid_2)
        self.linear3 = nn.Linear(n_hid_2, out_dim)
        self.dropout = dropout

    def forward(self, x, adj = None): #add unused adj argument so same training functions can be used for now
        x = F.leaky_relu(self.linear1(x.transpose(0,1)))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.linear2(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = F.leaky_relu(self.linear3(x)[0][0])
        return out

