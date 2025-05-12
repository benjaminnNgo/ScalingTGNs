import os
import math
import sys
import time
import timeit
import torch
import copy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import random
import wandb
from torch_geometric_temporal import EvolveGCNO, GCLSTM
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from script.utils.tgf_param import supported_properties
from script.configs.data_spec import DATA_PATH

model_file_path = f"{DATA_PATH}/output/ckpts/gclstm/social_models/"

partial_path = f"{DATA_PATH}/input/"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

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
    """
        Because node addresses are string, so function map string to a int
        This function also makes 2 node have the same address have the same node id
    """
    if lookup_node not in node_id_dict:
        node_id_dict[lookup_node] = curr_idx
        curr_idx += 1
    return node_id_dict[lookup_node],curr_idx



def extra_dataset_attributes_loading(args, readout_scheme='mean', task="lconcomp_gs"):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """

    TG_labels_data = []
    TG_feats_data = []

    logger.info("INFO: Extracting extra dataset attributes")
    for dataset in args.dataset:
        print("Loading features for {}".format(dataset))
        task_name = supported_properties[task]
        print("Loading Label for task {}".format(task))
        label_filename = f'{partial_path}/raw/labels/{task_name}/{dataset}_{task_name}_labels.csv'
        label_df = pd.read_csv(label_filename, header=None, names=['label'])
        TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)
        TG_labels_data.append(TG_labels)

        # load and process graph-pooled (node-level) features
        edgelist_filename = f'{partial_path}/raw/edgelists/{dataset}_edgelist.txt'
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
        TG_feats_data.append(TG_feats)

    logger.info("INFO: Extracting extra dataset attributes done!")
    return TG_labels_data, TG_feats_data

def data_loader_geometric_temporal(dataset):
    """
        Data loader for traing models with PyTorch geometric temporal
        data is a dictionary has following attributes:
            - edge index for each snapshot ( to get edge index for snapshot 0: data['edge_index'][0])
            - edge attribute for each snapshot ( similar above)
            - time length : number of snapshots
            - number of nodes
    """
    # partial_path = f'../data/input/raw/'

    data_root = '{}/cached/{}/'.format(partial_path, dataset)
    filepath = mkdirs(data_root) + '{}_pyTorchGeoTemp.data'.format(
        dataset)  # the data will be saved here after generation.
    print("INFO: Dataset: {}".format(dataset))
    print("DEBUG: Look for data at {}".format(filepath))
    if os.path.isfile(filepath):
        print('INFO: Loading {} directly.'.format(dataset))
        return torch.load(filepath)

    edgelist_filename = f'{partial_path}/raw/edgelists/{dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)

    max_transfer = float(edgelist_df['weight'].max())
    min_transfer = float(edgelist_df['weight'].min())
    if max_transfer == min_transfer:
        max_transfer = min_transfer + 1

    # Normalization is needed to solve NaN lost problem
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
            source_node, curr_idx = get_node_id_int(node_id_dict, row['source'], curr_idx)
            destination_node, curr_idx = get_node_id_int(node_id_dict, row['destination'], curr_idx)
            ts_edges_idx.append(torch.tensor(np.array([source_node, destination_node]), dtype=torch.long))
            ts_edges_atts.append(row['weight'])

        assert len(ts_edges_atts) == len(ts_edges_idx)

        edge_idx_list.append(torch.tensor(np.transpose(np.array(ts_edges_idx))))
        edge_att_list.append(torch.tensor(np.array(ts_edges_atts), dtype=torch.long))

    data = {}
    data['edge_index'] = edge_idx_list
    data['edge_attribute'] = edge_att_list
    data['time_length'] = len(uniq_ts_list)
    data['num_nodes'] = curr_idx
    torch.save(data, filepath)
    return data

def save_results(dataset, test_auc, test_ap, lr, train_snapshot, test_snapshot, best_epoch, time):
    partial_path = model_file_path
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    result_path = f"{partial_path}/{args.results_file}"
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(
            columns=["dataset", "test_auc", "test_ap", "lr", "train_snapshot", "test_snapshot", "seed", "best_epoch",
                     "time"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'dataset': dataset,
                                  'test_auc': test_auc,
                                  'test_ap': test_ap,
                                  "lr": lr,
                                  "train_snapshot": train_snapshot,
                                  "test_snapshot": test_snapshot,
                                  "seed": args.seed,
                                  "best_epoch": best_epoch,
                                  "time": time
                                  }, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def save_results(model_name, mode, dataset, test_auc, test_ap, bias=False):
    partial_path = model_file_path
    if bias:
        result_path = "{}/test_result/{}_results_bias.csv".format(partial_path,args.model, model_name)
    else:
        result_path = "{}/test_result/{}_results.csv".format(partial_path,args.model, model_name)
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["dataset", "mode", "test_auc", "test_ap", "pack"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'dataset': dataset, 'mode' : mode, 'test_auc': test_auc, 'test_ap': test_ap, "pack":args.pack}, ignore_index=True)
    result_df.to_csv(result_path, index=False)


class RecurrentGCN(torch.nn.Module):
    """
        GCLSTM model from PyTorch Geometric Temporal
        reference:
        https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/gclstm_example.py
    """
    def __init__(self, node_feat_dim,hidden_dim):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(node_feat_dim, hidden_dim, 1)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

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
        self.data = data
        # self.data = []
        # args.dataset = []
        # text_path = "../data/input/data_list/{}".format(datasets_package_path)
        # with open(text_path, 'r') as file:
        #     for line in file:
        #         print("INFO: Dataset: {}".format(line))
        #         self.data.append(data_loader_geometric_temporal(dataset=line.strip()))
        #         args.dataset.append(line.strip())
        

        
        # self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)
        self.t_graph_labels, self.t_graph_feat = t_graph_labels, t_graph_feat
        self.num_datasets = len(self.data)
        # self.data = data_loader_geometric_temporal(args.dataset)
        self.edge_idx_list = [self.data[i]['edge_index'] for i in range(self.num_datasets)]
        self.edge_att_list = [self.data[i]['edge_attribute'] for i in range(self.num_datasets)]
        self.num_nodes_per_data = [self.data[i]['num_nodes'] + 1 for i in range(self.num_datasets)]
        self.num_nodes = max(self.num_nodes_per_data)
        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr
        self.len = [self.data[i]['time_length'] for i in range(self.num_datasets)]
        self.testlength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)]  # Re-calculate number of test snapshots
        self.evalLength = [math.floor(self.len[i] * args.val_ratio) for i in range(self.num_datasets)]

        self.train_shots_mask = [list(range(0, self.len[i] - self.testlength[i] - self.evalLength[i])) for i in range(self.num_datasets)] #Changed
        self.eval_shots_mask = [list(range(self.len[i] - self.testlength[i] - self.evalLength[i], self.len[i] - self.testlength[i])) for i in range(self.num_datasets)] #Changed
        self.test_shots_mask = [list(range(self.len[i] - self.testlength[i], self.len[i])) for i in range(self.num_datasets)]

        self.node_feat_dim = 256 #@TODO: Replace with args to config it easily
        self.node_feat = torch.randn((self.num_nodes, self.node_feat_dim)).to(args.device)
        # print(self.node_feat.size())
        # self.node_feat_dim = self.node_feat.size(1)  # @TODO: Replace with args to config it easily
        self.edge_feat_dim = 1 #@TODO: Replace with args to config it easily
        self.hidden_dim = args.nhid

        self.model_path = '{}/{}_{}_seed_{}'.format(model_file_path,
                                                    args.model,
                                                    args.pack,
                                                    args.seed)

        self.model = RecurrentGCN(node_feat_dim=self.node_feat_dim, hidden_dim=self.hidden_dim).to(args.device)
        logger.info("The models is going to be loaded from {}".format(self.model_path))
        self.model.load_state_dict(torch.load('{}.pth'.format(self.model_path)))
        

        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=self.hidden_dim + num_extra_feat, hidden_dim_1=self.hidden_dim + num_extra_feat,
                          hidden_dim_2=self.hidden_dim + num_extra_feat,
                          drop=0.1).to(args.device)  # @NOTE: these hyperparameters may need to be changed
        self.tgc_decoder.load_state_dict(torch.load('{}_mlp.pth'.format(self.model_path)))
        #Hidden for gclstm
        self.h = None
        self.h_0 = None
        self.c_0 = None

    # Hidden for gclstm
    def init_hidden(self):
        self.h = None
        self.h_0 = None
        self.c_0 = None

    # Detach gradient
    def detach_hidden(self):
        self.h =  self.h.detach()
        self.h_0 =  self.h_0.detach()
        self.c_0 = self.c_0.detach()

    def train(self):
        self.model.train()
        self.tgc_decoder.train()

    def eval(self):
        self.model.eval()
        self.tgc_decoder.eval()

    def tgclassification_test(self,readout_scheme, dataset_idx):
        self.detach_hidden()
        tg_labels, tg_preds = [], []
        # self.test_shots_mask = [list(range(self.len[i] - self.testlength[i] -self.evalLength[i], self.len[i])) for i in range(self.num_datasets)]
        for t_test_idx in self.test_shots_mask[dataset_idx]:
            self.eval()
            with torch.no_grad():
                edge_idx = self.edge_idx_list[dataset_idx][t_test_idx].to(args.device)
                edge_att = self.edge_att_list[dataset_idx][t_test_idx].float().to(args.device)

                self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)
                tg_readout = readout_function(self.h, readout_scheme)

                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_test_idx]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(
                    self.t_graph_labels[dataset_idx][t_test_idx].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                # self.detach_hidden()

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap

    def tgclassification_val(self, readout_scheme, dataset_idx):
        self.detach_hidden()

        tg_labels, tg_preds = [], []
        for t_eval_idx in self.eval_shots_mask[dataset_idx]:
            self.eval()
            with torch.no_grad():
                edge_idx = self.edge_idx_list[dataset_idx][t_eval_idx].to(args.device)
                edge_att = self.edge_att_list[dataset_idx][t_eval_idx].float().to(args.device)

                self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)


                # graph readout
                tg_readout = readout_function(self.h, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_eval_idx]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_eval_idx].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap


    def test(self):
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)        
        self.model.eval()
        self.tgc_decoder.eval()
        for dataset_idx in range(self.num_datasets):
                self.init_hidden()
                data_name = args.dataset[dataset_idx]
                # # print(data_name)
                
                for snapshot_idx in self.train_shots_mask[dataset_idx]:
                    with torch.no_grad():
                        edge_idx = self.edge_idx_list[dataset_idx][snapshot_idx].to(args.device)
                        edge_att = self.edge_att_list[dataset_idx][snapshot_idx].float().to(args.device)
                        self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)
                        self.detach_hidden()

                # test_auc, test_ap = self.tgclassification_val(self.readout_scheme, dataset_idx)
                # logger.info("Final Test Data {}: AUC: {:.4f}, AP: {:.4f}".format(
                #     data_name, 
                #     test_auc, test_ap))
                # save_results(model_name, 
                #              "Eval", 
                #              data_name, 
                #              test_auc, 
                #              test_ap)
                
                for snapshot_idx in self.eval_shots_mask[dataset_idx]:
                    with torch.no_grad():
                        edge_idx = self.edge_idx_list[dataset_idx][snapshot_idx].to(args.device)
                        edge_att = self.edge_att_list[dataset_idx][snapshot_idx].float().to(args.device)
                        self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)
                        self.detach_hidden()

                test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx)
                print("Final Test Data {}: AUC: {:.4f}, AP: {:.4f}".format(
                    data_name, 
                    test_auc, test_ap))
                save_results(model_name, 
                             "Test", 
                             data_name, 
                             test_auc, 
                             test_ap)
           

if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import logger
    args.model = "GCLSTM"
    datasets_package_path = "mix_dataset_package_test.txt"
    text_path = "{}/input/data_list/{}".format(DATA_PATH,datasets_package_path)
    data = []
    task = "lconcomp_gs"
    args.dataset = []
    with open(text_path, 'r') as file:
        for line in file:
            print("INFO: Dataset: {}".format(line))
            data.append(data_loader_geometric_temporal(dataset=line.strip()))
            args.dataset.append(line.strip())
    

        
    t_graph_labels, t_graph_feat = extra_dataset_attributes_loading(args)

    model_name = "{}_{}_seed_{}".format(args.model, args.pack, args.seed)
    runner = Runner()
    runner.test()
