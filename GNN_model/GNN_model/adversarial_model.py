import torch.nn.functional as F
from GNN_model.models import VanillaNN


class MICPredictor(VanillaNN):
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x.transpose(0, 1)))
        F.dropout(x, self.dropout, inplace=True, training=True)
        x = F.leaky_relu(self.linear2(x))
        F.dropout(x, self.dropout, inplace=True, training=True)
        out = F.leaky_relu(self.linear3(x)[0][0])
        return out, x


Adversary = VanillaNN
