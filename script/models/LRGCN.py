from torch_geometric_temporal import EvolveGCNO, GCLSTM,LRGCN
import torch
import torch.nn.functional as F


class LRGCN(torch.nn.Module):
    """
        GCLSTM model from PyTorch Geometric Temporal
        reference:
        https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/gclstm_example.py
    """
    def __init__(self, node_feat_dim,hidden_dim,K = 1):
        super(LRGCN, self).__init__()
        self.recurrent = LRGCN(node_feat_dim, hidden_dim, K)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        return h_0, c_0