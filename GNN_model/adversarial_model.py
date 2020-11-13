import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN_model.layers import GraphConvolutionPerNode

class MICPredictor(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, n_hid_3, out_dim, dropout):
        super(MICPredictor, self).__init__()

        self.gc = GraphConvolutionPerNode(n_feat, n_hid_1)
        self.linear1 = nn.Linear(n_hid_1, n_hid_2)
        self.linear2 = nn.Linear(n_hid_2, n_hid_3)
        self.linear3 = nn.Linear(n_hid_3, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        #return MIC prediction and output of penultimate layer as input to adversary
        x = F.leaky_relu(self.gc(x, adj))
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        x = F.leaky_relu(self.linear2(x))
        pred = self.linear3(x)[0][0]
        return pred, x

class Adversary(nn.Module):
    def __init__(self, n_feat, n_hid, out_dim, dropout):
        super(Adversary, self).__init__()

        self.linear1 = nn.Linear(n_feat, n_hid)
        self.linear2 = nn.Linear(n_hid, out_dim)
        self.dropout = dropout

    def forward(self, x):
        #input is output of penultimate layer of MICPredictor
        x = F.leaky_relu(self.linear1(x))
        F.dropout(x, self.dropout, inplace = True, training = True)
        pred = self.linear2(x)[0]
        return pred