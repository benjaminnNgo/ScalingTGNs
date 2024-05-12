import math
import os

import networkx as nx
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
import wandb
import timeit
import pandas as pd

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def readout_function(embeddings, readout_scheme='mean'):
    """
    Read out function to generate a representation for the whole graph
    reference:
    https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification/blob/master/readouts/basic_readout.py
    """
    # note: x.size(): [#nodes, args.n_out]
    if readout_scheme == 'max':
        readout = torch.max(embeddings, dim=0)[0].squeeze()  # max readout
    elif readout_scheme == 'mean':
        readout = torch.mean(embeddings, dim=0).squeeze()  # mean readout
    elif readout_scheme == 'sum':
        readout = torch.sum(embeddings, dim=0).squeeze()  # sum readout
    else:
        readout = None
        raise ValueError("Readout Method Undefined!")
    return readout

def get_node_id_int(node_id_dict,lookup_node,curr_idx):
    if lookup_node not in node_id_dict:
        node_id_dict[lookup_node] = curr_idx
        curr_idx += 1
    return node_id_dict[lookup_node],curr_idx




def extra_dataset_attributes_loading(args, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    partial_path = f'../data/input/raw/'

    # load graph lables
    label_filename = f'{partial_path}/labels/{args.dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)

    # load and process graph-pooled (node-level) features
    edgelist_filename = f'{partial_path}/edgelists/{args.dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)
    uniq_ts_list = np.unique(edgelist_df['snapshot'])

    TG_feats = []
    for ts in uniq_ts_list:
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]

        ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight',
                                       create_using=nx.MultiDiGraph)
        node_list = list(ts_G.nodes)
        indegree_list = np.array(ts_G.in_degree(node_list))
        weighted_indegree_list = np.array(ts_G.in_degree(node_list, weight='weight'))
        outdegree_list = np.array(ts_G.out_degree(node_list))
        weighted_outdegree_list = np.array(ts_G.out_degree(node_list, weight='weight'))

        if readout_scheme == 'max':
            TG_this_ts_feat = np.array([np.max(indegree_list), np.max(weighted_indegree_list),
                                        np.max(outdegree_list), np.max(weighted_outdegree_list)])
        elif readout_scheme == 'mean':
            TG_this_ts_feat = np.array([np.mean(indegree_list[:, 1].astype(float)),
                                        np.mean(weighted_indegree_list[:, 1].astype(float)),
                                        np.mean(outdegree_list[:, 1].astype(float)),
                                        np.mean(weighted_outdegree_list[:, 1].astype(float))])
        elif readout_scheme == 'sum':
            TG_this_ts_feat = np.array([np.sum(indegree_list), np.sum(weighted_indegree_list),
                                        np.sum(outdegree_list), np.sum(weighted_outdegree_list)])
        else:
            TG_this_ts_feat = None
            raise ValueError("Readout scheme is Undefined!")

        TG_feats.append(TG_this_ts_feat)

    # scale the temporal graph features to have a reasonable range
    scalar = MinMaxScaler()
    TG_feats = scalar.fit_transform(TG_feats)

    return TG_labels, TG_feats

def data_loader_egcn(dataset):
    partial_path = f'../data/input/raw/'

    data_root = '../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}_egcn.data'.format(dataset)  # the data will be saved here after generation.
    print("INFO: Dataset: {}".format(dataset))
    print("DEBUG: Look for data at {}.".format(filepath))
    if os.path.isfile(filepath):
        print('INFO: Loading {} directly.'.format(dataset))
        return torch.load(filepath)

    edgelist_filename = f'{partial_path}/edgelists/{args.dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)

    max_transfer = float(edgelist_df['weight'].max())
    min_transfer = float(edgelist_df['weight'].min())
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    # value normalization

    edgelist_df['weight'] = edgelist_df['weight'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    uniq_node = set()
    uniq_node.update(edgelist_df['source'].tolist())
    uniq_node.update(edgelist_df['destination'].tolist())

    edge_idx_list = []
    edge_att_list = []
    node_id_dict = {}
    curr_idx = 0
    for ts in uniq_ts_list:
        ts_edges_idx = []
        ts_edges_atts = []
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
        for idx, row in ts_edges.iterrows():
            source_node,curr_idx = get_node_id_int(node_id_dict,row['source'],curr_idx)
            destination_node, curr_idx = get_node_id_int(node_id_dict, row['destination'], curr_idx)
            ts_edges_idx.append(torch.tensor(np.array([source_node, destination_node]),dtype=torch.long))
            ts_edges_atts.append(row['weight'])

        assert len(ts_edges_atts) == len(ts_edges_idx)

        edge_idx_list.append(torch.tensor(np.transpose(np.array(ts_edges_idx))))
        edge_att_list.append(torch.tensor(np.array(ts_edges_atts),dtype=torch.long))

    print(node_id_dict)
    data = {}
    data['edge_index'] = edge_idx_list
    data['edge_attribute'] = edge_att_list
    data['time_length'] = len(uniq_ts_list)
    data['num_nodes'] = curr_idx


    torch.save(data, filepath)
    return data



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_feat_dim)
        self.linear = torch.nn.Linear(node_feat_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class MLP(torch.nn.Module):
    """
  Binary Classifier to work as a decoder

  reference:
    https://github.com/twitter-research/tgn/blob/master/utils/utils.py
  """

    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, drop=0.5):
        super().__init__()
        self.fc_1 = torch.nn.Linear(in_dim, hidden_dim_1)
        self.fc_2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc_3 = torch.nn.Linear(hidden_dim_1, 1)
        self.act = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)
class Runner():
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="egcn",

                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": "egcn",
                    "dataset": args.dataset,
                    "time granularity": args.time_scale,
                }
            )

        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)
        self.data = data_loader_egcn(args.dataset)
        self.edge_idx_list = self.data['edge_index']
        self.edge_att_list = self.data['edge_attribute']
        self.num_nodes = self.data['num_nodes'] + 1
        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr
        self.len = self.data['time_length']
        self.testlength = math.floor(self.len * args.test_ratio)  # Re-calculate number of test snapshots
        self.start_train = 0
        self.evalLength = math.floor(self.len * args.eval_ratio)

        start_train =0
        self.train_shots_mask = list(range(start_train, self.len - self.testlength - self.evalLength))
        self.eval_shots_mask = list(range(self.len - self.testlength - self.evalLength, self.len - self.testlength))
        self.test_shots_mask = list(range(self.len - self.testlength, self.len))

        self.node_feat_dim = 256 #@TODO: Replace with args to config it easily
        self.node_feat = torch.randn((self.num_nodes, self.node_feat_dim)).to(args.device)
        self.edge_feat_dim = 1 #@TODO: Replace with args to config it easily
        self.hidden_dim = args.nhid

        self.model = RecurrentGCN(node_feat_dim=self.node_feat_dim, hidden_dim=self.hidden_dim).to(args.device)


        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=self.hidden_dim + num_extra_feat, hidden_dim_1=self.hidden_dim + num_extra_feat,
                          hidden_dim_2=self.hidden_dim + num_extra_feat,
                          drop=0.1).to(args.device)  # @NOTE: these hyperparameters may need to be changed

    def tgclassification_test(self,epoch,readout_scheme):
        tg_labels, tg_preds = [], []

        for t_test_idx in self.test_shots_mask:
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_idx = self.edge_idx_list[t_test_idx].to(args.device)
                edge_att = self.edge_att_list[t_test_idx].float().to(args.device)

                h = self.model(self.node_feat, edge_idx, edge_att)
                tg_readout = readout_function(h, readout_scheme)

                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_test_idx + len(self.train_shots_mask) + len(
                                              self.eval_shots_mask)]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(
                    self.t_graph_labels[t_test_idx + len(self.train_shots_mask) + len(self.eval_shots_mask)].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

    def tgclassification_val(self,epoch,readout_scheme):

        tg_labels, tg_preds = [], []

        for t_eval_idx in self.test_shots_mask:
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_idx = self.edge_idx_list[t_eval_idx].to(args.device)
                edge_att = self.edge_att_list[t_eval_idx].float().to(args.device)

                h = self.model(self.node_feat, edge_idx, edge_att)

                # graph readout
                tg_readout = readout_function(h, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_eval_idx + len(self.train_shots)]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_eval_idx + len(self.train_shots)].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())


            auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
            return epoch, auc, ap


    def run(self):
        optimizer = torch.optim.Adam(
            set(self.model.parameters()) | set(self.tgc_decoder.parameters()), lr=0.0001)
        criterion = torch.nn.MSELoss()
        train_avg_epoch_loss_dict = {}
        t_total_start = timeit.default_timer()
        min_loss = 10
        train_avg_epoch_loss_dict = {}

        best_model = self.model.state_dict()
        best_MLP = self.tgc_decoder.state_dict()

        best_epoch = -1
        patience = 0
        best_eval_auc = -1  # Set previous evaluation result to very small number
        best_test_auc = -1
        best_test_ap = -1

        for param in self.model.parameters():
            param.retain_grad()

        for epoch in range(20):
        # for epoch in range(1, args.max_epoch + 1):
            print("------------------------------------------")
            train_start_time = timeit.default_timer()
            optimizer.zero_grad()
            total_loss = 0
            self.model.train()
            self.tgc_decoder.train()
            h = None
            tg_labels = []
            tg_preds = []
            epoch_losses = []

            for snapshot_idx in self.train_shots_mask:

                optimizer.zero_grad()

                edge_idx = self.edge_idx_list[snapshot_idx].to(args.device)
                edge_att = self.edge_att_list[snapshot_idx].float().to(args.device)

                h = self.model(self.node_feat, edge_idx, edge_att)
                tg_readout = readout_function(h, "mean")
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[snapshot_idx]).to(args.device)))
                # print(tg_embedding)
                #
                # graph classification
                tg_label = self.t_graph_labels[snapshot_idx].float().view(1, )
                tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                tg_labels.append(tg_label.cpu().numpy())
                tg_preds.append(tg_pred.cpu().detach().numpy())
                t_loss = criterion(tg_pred, tg_label)
                t_loss.backward(retain_graph=True)
                optimizer.step()
                epoch_losses.append(t_loss.item())

            avg_epoch_loss = np.mean(epoch_losses)
            print(avg_epoch_loss)
            train_avg_epoch_loss_dict[epoch] = avg_epoch_loss
            train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
            print(train_auc,train_ap)





if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare

    args.dataset = "CMT0xf85feea2fdd81d51177f6b8f35f0e6734ce45f5f"
    set_random(args.seed)
    runner = Runner()
    runner.run()
    # t_graph_labels, t_graph_feat = extra_dataset_attributes_loading(args)
    # data = data_loader_egcn(args.dataset)
    #
    # print(data['edge_index'][2])
    # args.device = 'cuda'
    # # get masks
    #
    # edge_idx_list = data['edge_index']
    # edge_att_list = data['edge_attribute']
    # num_nodes = data['num_nodes'] + 1
    #
    #
    # readout_scheme = 'mean'
    # tgc_lr = args.lr
    # len = data['time_length']
    # testlength = math.floor(len * args.test_ratio)  # Re-calculate number of test snapshots
    # start_train = 0
    # evalLength = math.floor(len * args.eval_ratio)
    #
    # train_shots_mask = list(range(start_train, len - testlength - evalLength))
    # eval_shots_mask = list(range(len - testlength - evalLength, len - testlength))
    # test_shots_mask = list(range(len - testlength, len))
    # # train_mask = dataset.train_mask
    # # val_mask = dataset.val_mask
    # # test_mask = dataset.test_mask
    # # train_edges = full_data[train_mask]
    # # val_edges = full_data[val_mask]
    # # test_edges = full_data[test_mask]
    #
    # if args.wandb:
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="egcn",
    #
    #         # track hyperparameters and run metadata
    #         config={
    #             "learning_rate": args.lr,
    #             "architecture": "egcn",
    #             "dataset": args.dataset,
    #             "time granularity": args.time_scale,
    #         }
    #     )
    # # ! set up node features
    # node_feat_dim = 256
    # node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)
    # edge_feat_dim = 1
    # hidden_dim = 256
    #
    # # * load the discretized version
    #
    # num_epochs = args.max_epoch
    # lr = args.lr
    #
    # args.seed = 710
    # args.num_runs = 1
    #
    # for seed in range(args.seed, args.seed + args.num_runs):
    #     set_random(seed)
    #     print(f"Run {seed}")
    #
    #     # * initialization of the model to prep for training
    #     model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim).to(args.device)
    #     node_feat = torch.randn((num_nodes, node_feat_dim)).float().to(args.device)
    #
    #     num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
    #     tgc_decoder = MLP(in_dim=hidden_dim + num_extra_feat, hidden_dim_1=hidden_dim + num_extra_feat,
    #                            hidden_dim_2=hidden_dim + num_extra_feat,
    #                            drop=0.1).to(args.device)  # @NOTE: these hyperparameters may need to be changed
    #
    #     optimizer = torch.optim.Adam(
    #         set(model.parameters()) | set(tgc_decoder.parameters()), lr=lr)
    #     criterion = torch.nn.MSELoss()
    #
    #     best_val = 0
    #     best_test = 0
    #     best_epoch = 0
    #
    #     for epoch in range(10):
    #         print("------------------------------------------")
    #         train_start_time = timeit.default_timer()
    #         optimizer.zero_grad()
    #         total_loss = 0
    #         model.train()
    #         tgc_decoder.train()
    #         h = None
    #         total_loss = 0
    #         tg_labels = []
    #         tg_preds = []
    #         epoch_losses = []
    #         for snapshot_idx in train_shots_mask:
    #
    #             optimizer.zero_grad()
    #             edge_idx = edge_idx_list[snapshot_idx].to(args.device)
    #             edge_att = edge_att_list[snapshot_idx].float().to(args.device)
    #
    #             h = model(node_feat, edge_idx, edge_att)
    #             tg_readout = readout_function(h, "mean")
    # #             #@=========================Need t_graph_feat here=====================
    #             tg_embedding = torch.cat((tg_readout, torch.from_numpy(t_graph_feat[snapshot_idx]).to(args.device)))
    # #
    #             # graph classification
    #             tg_label = t_graph_labels[snapshot_idx].float().view(1, )
    #             tg_pred = tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()
    #
    #             tg_labels.append(tg_label.cpu().numpy())
    #             tg_preds.append(tg_pred.cpu().detach().numpy())
    #             t_loss = criterion(tg_pred, tg_label)
    #             t_loss.backward()
    #             optimizer.step()
    #             epoch_losses.append(t_loss.item())
    # #             # update the models====================================
    #         avg_epoch_loss = np.mean(epoch_losses)
    #         print(avg_epoch_loss)
    #
    #
    #         train_time = timeit.default_timer() - train_start_time
    #         print(f'Epoch {epoch}/{num_epochs}, Loss: {total_loss}')
    #         print("Train time: ", train_time)
    #
    #
    #     #===================================================================
    #     #     # ? Evaluation starts here
    #     #     #@Change all validation step below here for as training and apply early stopping
    #     #     val_snapshots = data['val_data']['edge_index']
    #     #     ts_list = data['val_data']['ts_map']
    #     #     val_loader = TemporalDataLoader(val_edges, batch_size=batch_size)
    #     #     evaluator = Evaluator(name=args.dataset)
    #     #     neg_sampler = dataset.negative_sampler
    #     #     dataset.load_val_ns()
    #     #
    #     #     start_epoch_val = timeit.default_timer()
    #     #     val_metrics, h = test_tgb(h, val_loader, val_snapshots, ts_list,
    #     #                               node_feat, model, link_pred, neg_sampler, evaluator, metric, split_mode='val')
    #     #     val_time = timeit.default_timer() - start_epoch_val
    #     #     print(f"Val {metric}: {val_metrics}")
    #     #     print("Val time: ", val_time)
    #     #     if (args.wandb):
    #     #         wandb.log({"train_loss": (total_loss),
    #     #                    "val_" + metric: val_metrics,
    #     #                    "train time": train_time,
    #     #                    "val time": val_time,
    #     #                    })
    #     #
    #     #     # ! report test results when validation improves
    #     #     if (val_metrics > best_val):
    #     #         dataset.load_test_ns()
    #     #         test_snapshots = data['test_data']['edge_index']
    #     #         ts_list = data['test_data']['ts_map']
    #     #         test_loader = TemporalDataLoader(test_edges, batch_size=batch_size)
    #     #         neg_sampler = dataset.negative_sampler
    #     #         dataset.load_test_ns()
    #     #
    #     #         test_start_time = timeit.default_timer()
    #     #         test_metrics, h = test_tgb(h, test_loader, test_snapshots, ts_list,
    #     #                                    node_feat, model, link_pred, neg_sampler, evaluator, metric,
    #     #                                    split_mode='test')
    #     #         test_time = timeit.default_timer() - test_start_time
    #     #         best_val = val_metrics
    #     #         best_test = test_metrics
    #     #
    #     #         print("test metric is ", test_metrics)
    #     #         print("test elapsed time is ", test_time)
    #     #         print("--------------------------------")
    #     #         if ((epoch - best_epoch) >= args.patience and epoch > 1):
    #     #             best_epoch = epoch
    #     #             break
    #     #         best_epoch = epoch
    #     # print("run finishes")
    #     # print("best epoch is, ", best_epoch)
    #     # print("best val performance is, ", best_val)
    #     # print("best test performance is, ", best_test)
    #     # print("------------------------------------------")