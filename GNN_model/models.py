import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_model.layers import GraphConvolution, GraphConvolutionPerNode


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_feat, n_hid_1)
        self.gc2 = GraphConvolution(n_hid_1, n_hid_2)
        self.linear = nn.Linear(n_hid_2, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.gc2(x, adj))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = self.linear(x)
        out = x.sum()/len(x) #average across nodes
        return out


class GCNPerNode(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout):
        super(GCNPerNode, self).__init__()

        self.gc = GraphConvolutionPerNode(n_feat, n_hid_1)
        self.linear1 = nn.Linear(n_hid_1, n_hid_2)
        self.linear2 = nn.Linear(n_hid_2, out_dim) 
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc(x, adj))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        out = self.linear2(x)[0][0]
        return out


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
        out = self.linear3(x)[0][0]
        return out
