import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM 
from torch_geometric.utils.negative_sampling import negative_sampling
from models.tgn.decoder import LinkPredictor



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(in_channels=node_feat_dim, 
                                out_channels=32, 
                                K=1,) #K is the Chebyshev filter size

    def forward(self, x, edge_index, edge_weight, h, c):
        r"""
        forward function for the model, 
        this is used for each snapshot
        h: node hidden state matrix from previous time
        c: cell state matrix from previous time
        """
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        return h_0, c_0




if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader

    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)


    #! add support for node features in the future
    node_feat_dim = 16 #all 0s for now
    edge_feat_dim = 16 

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    num_epochs = 200
    lr = 0.01

    #* initialization of the model to prep for training
    model = RecurrentGCN(node_feat_dim=node_feat_dim)
    node_feat = torch.zeros((num_nodes, node_feat_dim))
    link_pred = LinkPredictor(in_channels=edge_feat_dim)
    optimizer = torch.optim.Adam(
        set(model.parameters()) | set(link_pred.parameters()), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()



    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        h, c = None, None
        snapshot_list = train_data['edge_index']
        for snapshot_idx in range(train_data['time_length']):
            optimizer.zero_grad()
            pos_index = snapshot_list[snapshot_idx]
            pos_index = pos_index.long()
            neg_edges = negative_sampling(pos_index, num_nodes=num_nodes, num_neg_samples=(pos_index.size(1)*1))

            if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                edge_index = pos_index
                if ('edge_attr' not in train_data):
                    edge_attr = torch.ones(edge_index.size(1), edge_feat_dim)
                h, c = model(node_feat, edge_index, edge_attr, h, c)
            else: #subsequent snapshot, feed the previous snapshot
                prev_index = snapshot_list[snapshot_idx-1]
                edge_index = prev_index.long()
                if ('edge_attr' not in train_data):
                    edge_attr = torch.ones(edge_index.size(1), edge_feat_dim)
                h, c = model(node_feat, edge_index, edge_attr, h, c)

            pos_out = link_pred(h[edge_index[0]], h[edge_index[1]])
            neg_out = link_pred(h[neg_edges[0]], h[neg_edges[1]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * edge_index.size(1)






