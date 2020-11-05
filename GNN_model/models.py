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
        return torch.tanh(out)


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
        return torch.tanh(out)