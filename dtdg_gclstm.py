import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM 
from torch_geometric.utils.negative_sampling import negative_sampling
from models.tgn.decoder import LinkPredictor
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler


#! not training properly, the model prediction doesn't change


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, K=1):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(in_channels=node_feat_dim, 
                                out_channels=hidden_dim, 
                                K=K,) #K is the Chebyshev filter size
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight, h, c):
        r"""
        forward function for the model, 
        this is used for each snapshot
        h: node hidden state matrix from previous time
        c: cell state matrix from previous time
        """
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)




if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader

    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)


    #! add support for node features in the future
    node_feat_dim = 16 #all 0s for now
    edge_feat_dim = 1 #for edge weights
    hidden_dim = 256

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    num_epochs = 200
    lr = args.lr

    #* initialization of the model to prep for training
    model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=3).to(args.device)
    node_feat = torch.zeros((num_nodes, node_feat_dim)).to(args.device)
    # link_pred = LinkPredictor(in_channels=hidden_dim).to(args.device)
    link_pred = LinkPredictor(hidden_dim, hidden_dim, 1,
                              2, 0.2).to(args.device)


    optimizer = torch.optim.Adam(
        set(model.parameters()) | set(link_pred.parameters()), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        link_pred.train()
        snapshot_list = train_data['edge_index']
        h_0, c_0, h = None, None, None
        loss = 0
        for snapshot_idx in range(train_data['time_length']):
            pos_index = snapshot_list[snapshot_idx]
            pos_index = pos_index.long().to(args.device)
            neg_edges = negative_sampling(pos_index, num_nodes=num_nodes, num_neg_samples=(pos_index.size(1)*1))

            if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                edge_index = pos_index
                # TODO, also need to support edge attributes correctly in TGX
                if ('edge_attr' not in train_data):
                    edge_attr = torch.ones(edge_index.size(1), edge_feat_dim).to(args.device)
                else:
                    raise NotImplementedError("Edge attributes are not yet supported")
                h, h_0, c_0 = model(node_feat, edge_index, edge_attr, h_0, c_0)
            else: #subsequent snapshot, feed the previous snapshot
                #prev_index = snapshot_list[snapshot_idx-1]
                prev_index = pos_index
                edge_index = prev_index.long().to(args.device)
                # TODO, also need to support edge attributes correctly in TGX
                if ('edge_attr' not in train_data):
                    edge_attr = torch.ones(edge_index.size(1), edge_feat_dim).to(args.device)
                else:
                    raise NotImplementedError("Edge attributes are not yet supported")
                h, h_0, c_0 = model(node_feat, edge_index, edge_attr, h_0, c_0)

            pos_out = link_pred(h[edge_index[0]], h[edge_index[1]])
            neg_out = link_pred(h[neg_edges[0]], h[neg_edges[1]])

            loss += torch.mean(criterion(pos_out, torch.ones_like(pos_out)))
            loss += torch.mean(criterion(neg_out, torch.zeros_like(neg_out)))

        #due to being recurrent model and takes in recurrent input, loss is outside the loop
        loss.backward()
        total_loss = float(loss)
        optimizer.step()
        optimizer.zero_grad()
        print (f'Epoch {epoch}/{num_epochs}, Loss: {total_loss/num_nodes}')

        #! Evaluation starts here
        #! need to optimize code to have train, test function, maybe in a class
        model.eval()
        link_pred.eval()
        evaluator = Evaluator(name="tgbl-wiki") #reuse MRR evaluator from TGB
        metric = "mrr"
        neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")

        #* load the val negative samples
        neg_sampler.load_eval_set(fname=args.dataset + "_val_ns.pkl", split_mode="val")

        val_snapshots = val_data['edge_index'] #converted to undirected, also removes self loops as required by HTGN
        val_edges = val_data['original_edges'] #original edges unmodified
        ts_min = min(val_snapshots.keys())

        h_0 = h_0.detach()
        c_0 = c_0.detach()
        h = h.detach()

        perf_list = {}
        perf_idx = 0

        for snapshot_idx in val_snapshots.keys():
            pos_index = torch.from_numpy(val_edges[snapshot_idx])
            pos_index = pos_index.long().to(args.device)
            #* update the node embeddings with edges from previous snapshot
            if (snapshot_idx > ts_min):
                #* update the snapshot embedding
                prev_index = val_snapshots[snapshot_idx-1]
                prev_index = prev_index.long().to(args.device)
                if ('edge_attr' not in val_data):
                    edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                else:
                    raise NotImplementedError("Edge attributes are not yet supported")
                h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)
            
            for i in range(pos_index.shape[0]):
                pos_src = pos_index[i][0].item()
                pos_dst = pos_index[i][1].item()
                pos_t = snapshot_idx
                neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode='val')
                
                for idx, neg_batch in enumerate(neg_batch_list):
                    query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                    query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                    query_src = torch.from_numpy(query_src).long().to(args.device)
                    query_dst = torch.from_numpy(query_dst).long().to(args.device)
                    edge_index = torch.stack((query_src, query_dst), dim=0)
                    y_pred = link_pred(h[edge_index[0]], h[edge_index[1]])
                    y_pred = y_pred.detach().cpu().numpy()
                    input_dict = {
                            "y_pred_pos": np.array([y_pred[0]]),
                            "y_pred_neg": y_pred[1:],
                            "eval_metric": [metric],
                        }
                    perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
                    perf_idx += 1

        result = list(perf_list.values())
        perf_list = np.array(result)
        perf_metrics = float(np.mean(perf_list))

        print(f"Epoch {epoch} : Val {metric}: {perf_metrics}")







