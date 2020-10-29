import torch.nn as nn
import torch.nn.functional as F
from GNN_model.layers import GCNConv

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(n_feat, n_hid)
        self.gc2 = GCNConv(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)