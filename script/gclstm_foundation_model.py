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

model_file_path = '/network/scratch/r/razieh.shirzadkhani/fm'
partial_path = '/network/scratch/r/razieh.shirzadkhani/fm/fm_data/data_lt_70/all_data/'

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



def extra_dataset_attributes_loading_old(args, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    # partial_path = f'../data/input/raw/'
    partial_path = "/network/scratch/r/razieh.shirzadkhani/fm/fm_data/data_lt_70/all_data/raw/"
    TG_labels_data = []
    TG_feats_data = []
    logger.info("INFO: Extracting extra dataset attributes")
    for dataset in args.dataset:
        print("Loading features for {}".format(dataset))
        # load graph lables
        label_filename = f'{partial_path}/labels/{dataset}_labels.csv'
        label_df = pd.read_csv(label_filename, header=None, names=['label'])
        TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)
        TG_labels_data.append(TG_labels)
        
        # load and process graph-pooled (node-level) features 
        edgelist_filename = f'{partial_path}/edgelists/{dataset}_edgelist.txt'
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
                TG_this_ts_feat = np.array([np.mean(indegree_list[: , 1].astype(float)), 
                                            np.mean(weighted_indegree_list[: , 1].astype(float)), 
                                            np.mean(outdegree_list[: , 1].astype(float)), 
                                            np.mean(weighted_outdegree_list[: , 1].astype(float))])
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

    data_root = '{}/cached_iclr_rebuttal/{}/'.format(partial_path, dataset)
    filepath = mkdirs(data_root) + '{}_pyTorchGeoTemp.data'.format(dataset)  # the data will be saved here after generation.
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
            source_node,curr_idx = get_node_id_int(node_id_dict,row['source'],curr_idx)
            destination_node, curr_idx = get_node_id_int(node_id_dict, row['destination'], curr_idx)
            ts_edges_idx.append(torch.tensor(np.array([source_node, destination_node]),dtype=torch.long))
            ts_edges_atts.append(row['weight'])

        assert len(ts_edges_atts) == len(ts_edges_idx)

        edge_idx_list.append(torch.tensor(np.transpose(np.array(ts_edges_idx))))
        edge_att_list.append(torch.tensor(np.array(ts_edges_atts),dtype=torch.long))

    data = {}
    data['edge_index'] = edge_idx_list
    data['edge_attribute'] = edge_att_list
    data['time_length'] = len(uniq_ts_list)
    data['num_nodes'] = curr_idx
    torch.save(data, filepath)
    return data

def save_results(dataset, test_auc, test_ap,lr,train_snapshot,test_snapshot,best_epoch,time):
    partial_path =  "../data/output/single_model_egcn/"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    result_path = f"{partial_path}/{args.results_file}"
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["dataset", "test_auc", "test_ap","lr","train_snapshot","test_snapshot","seed","best_epoch","time"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'dataset': dataset,
                                   'test_auc': test_auc,
                                   'test_ap': test_ap,
                                   "lr":lr,
                                   "train_snapshot":train_snapshot,
                                   "test_snapshot":test_snapshot,
                                   "seed":args.seed,
                                   "best_epoch":best_epoch,
                                   "time":time
                                   }, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def save_epoch_results(epoch, test_auc, test_ap, time, dataset=None):
    if dataset is None:
        result_path = "../data/output/epoch_result/average/{}/{}_epochResult.csv".format(args.model,filename)
    else:
        # data_name = args.data_name[args.dataset[dataset]] if args.dataset[dataset] in args.data_name else args.dataset[dataset]
        # print(data_name)
        result_folder = "../data/output/epoch_result/data/{}".format(dataset)
        result_path = result_folder + "/{}_epochResult.csv".format(filename)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
    
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)
    result_df = result_df.append({'epoch': int(epoch), 'test_auc': test_auc, 'test_ap': test_ap, 'time': time}, ignore_index=True)
    result_df.to_csv(result_path, index=False)


def save_epoch_training(epoch, train_auc, train_ap, loss, time, dataset=None):
    if dataset is None:
        result_path = "../data/output/training_test/average/{}/{}_epochResult.csv".format(args.model, filename)
    else:
        # data_name = args.data_name[args.dataset[dataset]] if args.dataset[dataset] in args.data_name else args.dataset[dataset]
        # print(data_name)
        result_folder = "../data/output/training_test/{}/data/{}".format(dataset)
        result_path = result_folder + "/{}_epochResult.csv".format(filename)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)
    result_df = result_df.append({'epoch': int(epoch), 'train_loss': loss, 'train_auc': train_auc, 'train_ap': train_ap, 'time': time}, ignore_index=True)
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
        self.data = []
        args.dataset = []
        text_path = "../data/input/data_list/{}".format(datasets_package_path)
        with open(text_path, 'r') as file:
            for line in file:
                print("INFO: Dataset: {}".format(line))
                self.data.append(data_loader_geometric_temporal(dataset=line.strip()))
                args.dataset.append(line.strip())
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="gclstm-log2_init",
                # Set name of the run:
                name="{}_{}_{}_{}".format(len(self.data), args.model, args.seed, args.curr_time),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,

                }
            )

        # init_logger('../data/output/log/{}/{}_{}_seed_{}_{}_{}_log.txt'.format(args.model, args.model, args.seed, len(args.dataset), args.nhid, args.curr_time))
        logger.info("INFO: Args: {}".format(args))
        # self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)
        self.t_graph_labels, self.t_graph_feat = [], []
        for dataset in args.dataset:
            t_graph_label_i, t_graph_feat_i = extra_dataset_attributes_loading(args, dataset)
            self.t_graph_labels.append(t_graph_label_i)
            self.t_graph_feat.append(t_graph_feat_i)
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
        self.hidden_dim = args.nhid * 20
        self.model_path = '{}/saved_models/fm/{}/{}_20'.format(model_file_path, 
                                                                        args.model,
                                                                        filename)
        
        self.model_chkp_path = '{}/saved_models/fm/{}/checkpoint/{}_20'.format(model_file_path, 
                                                                        args.model,
                                                                        filename)
        self.model = RecurrentGCN(node_feat_dim=self.node_feat_dim, hidden_dim=self.hidden_dim).to(args.device)


        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=self.hidden_dim + num_extra_feat, hidden_dim_1=self.hidden_dim + num_extra_feat,
                          hidden_dim_2=self.hidden_dim + num_extra_feat,
                          drop=0.1).to(args.device)  # @NOTE: these hyperparameters may need to be changed

        #Hidden for gclstm
        self.h = None
        self.h_0 = None
        self.c_0 = None

        # if os.path.exists("{}.pth".format(self.model_chkp_path)):
        #     logger.info("INFO: Model already exist and will be loaded from {}".format(self.model_chkp_path))
        #     checkpoint = torch.load("{}.pth".format(self.model_chkp_path))
        #     self.model.load_state_dict(checkpoint['model_state_dict'])

        #     mlp_checkpoint = torch.load("{}_mlp.pth".format(self.model_chkp_path))
        #     self.tgc_decoder.load_state_dict(mlp_checkpoint['model_state_dict'])
        #     self.start_epoch = checkpoint['epoch']
        #     logger.info("INFO: Epochs will start from: ".format(self.start_epoch))

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

    def tgclassification_test(self,epoch,readout_scheme):
        self.detach_hidden()
        tg_labels, tg_preds = [], []

        for t_test_idx in self.test_shots_mask:
            self.eval()
            with torch.no_grad():
                edge_idx = self.edge_idx_list[t_test_idx].to(args.device)
                edge_att = self.edge_att_list[t_test_idx].float().to(args.device)

                self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)
                tg_readout = readout_function(self.h, readout_scheme)

                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_test_idx]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(
                    self.t_graph_labels[t_test_idx].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

    def tgclassification_val(self, epoch, readout_scheme, dataset_idx):
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
        return epoch, auc, ap


    def run(self):
        optimizer = torch.optim.Adam(
            set(self.model.parameters()) | set(self.tgc_decoder.parameters()), lr=self.tgc_lr)
        criterion = torch.nn.MSELoss()

        train_avg_epoch_loss_dict = {}

        best_model = self.model.state_dict()
        best_mlp = self.tgc_decoder.state_dict()

        # For eval model to choose the best model
        best_epoch = -1
        patience = 0
        best_eval_auc = -1  # Set previous evaluation result to very small number
        best_eval_ap = -1


        t_total_start = timeit.default_timer()
        for epoch in range(1, args.max_epoch + 1):
            t_epoch_start = timeit.default_timer()
            self.init_hidden()
            optimizer.zero_grad()
            self.train()

            train_aucs, train_aps, eval_aucs, eval_aps = [], [], [], []
            epoch_losses = []
            dataset_rnd = random.sample(range(self.num_datasets), self.num_datasets)
            for dataset_idx in dataset_rnd:
                self.init_hidden()
                data_name = args.data_name[args.dataset[dataset_idx]] if args.dataset[dataset_idx] in args.data_name else args.dataset[dataset_idx]
                # print(data_name)
                tg_labels, tg_preds = [], []
                dataset_losses = []
                for snapshot_idx in self.train_shots_mask[dataset_idx]:
                    optimizer.zero_grad()

                    #Get edge list and edge attributes
                    edge_idx = self.edge_idx_list[dataset_idx][snapshot_idx].to(args.device)
                    edge_att = self.edge_att_list[dataset_idx][snapshot_idx].float().to(args.device)
                    # print(self.node_feat.size(), edge_idx.size(), edge_att.size())
                    self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)
                    tg_readout = readout_function(self.h, self.readout_scheme)
                    tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[dataset_idx][snapshot_idx]).to(args.device)))

                    # graph classification
                    tg_label = self.t_graph_labels[dataset_idx][snapshot_idx].float().view(1, )
                    tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                    t_loss = criterion(tg_pred, tg_label)
                    t_loss.backward()
                    optimizer.step()

                    tg_labels.append(tg_label.cpu().numpy())
                    tg_preds.append(tg_pred.cpu().detach().numpy())
                    dataset_losses.append(t_loss.item())
                    # epoch_losses.append(t_loss.item())
                    self.detach_hidden()

                avg_dataset_loss = np.mean(dataset_losses)
                epoch_losses.append(avg_dataset_loss)
                train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
                eval_epoch, eval_auc, eval_ap = self.tgclassification_val(epoch, self.readout_scheme, dataset_idx)
                train_aucs.append(train_auc)
                train_aps.append(train_ap)
                eval_aucs.append(eval_auc)
                eval_aps.append(eval_ap)
                # save_epoch_results(epoch,eval_auc,eval_ap, 0, dataset=data_name)
                # save_epoch_training(epoch,train_auc,train_ap, avg_dataset_loss, 0, dataset=data_name)

                # if (args.wandb):
                #     wandb.log({"Data {} Train Loss".format(data_name): avg_dataset_loss,
                #                "Data {} Train AUC".format(data_name) : train_auc,
                #                "Data {} Train AP".format(data_name) : train_ap,
                #                "Data {} Eval AUC".format(data_name) : eval_auc,
                #                "Data {} Eval AP".format(data_name) : eval_ap,
                #                "Epoch" : epoch                                
                #         })
                    
            avg_epoch_loss = np.mean(epoch_losses)
            train_avg_epoch_loss_dict[epoch] = avg_epoch_loss 
            avg_train_auc = np.mean(train_aucs)
            avg_train_ap = np.mean(train_aps)
            avg_eval_auc = np.mean(eval_aucs)
            avg_eval_ap = np.mean(eval_aps) 

            # Saving model checkpoint:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()}, 
                        "{}.pth".format(self.model_chkp_path))
            
            torch.save({'epoch': epoch,
                        'model_state_dict': self.tgc_decoder.state_dict()}, 
                        "{}_mlp.pth".format(self.model_chkp_path))
            

            # Only apply early stopping when after min_epoch of training
            # patience = 0 #Reset patience
            if best_eval_auc < avg_eval_auc or epoch <= args.min_epoch: #Use AUC as metric to define early stoping
                    # print("average eval auc: ", avg_eval_auc)
                    # print("difference: ", best_eval_auc - avg_eval_auc)
                    patience = 0
                    best_eval_auc = avg_eval_auc
                    best_eval_ap = avg_eval_ap
                    best_model = self.model.state_dict() #Saved the best model for testing
                    best_mlp = self.tgc_decoder.state_dict()
                    # best_test_results = [test_aucs, test_aps]
                    best_epoch = epoch
            else:
                # print("difference: ", best_eval_auc - avg_eval_auc)
                if best_eval_auc - avg_eval_auc > 0.0005:
                    patience += 1
                    # print("patience at epoch {}: {}".format(epoch, patience))
                if patience > args.patience:  
                    logger.info("Early stopping at epoch: {}".format(epoch))
                    break

            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            total_epoch_time = timeit.default_timer() - t_epoch_start
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                          total_epoch_time,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Train: Epoch:{}, AUC: {:.4f}, AP: {:.4f}, Loss: {:.4f}".format(epoch, avg_train_auc, avg_train_ap, avg_epoch_loss))
                logger.info(
                    "Eval: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(eval_epoch, avg_eval_auc, avg_eval_ap))

                
            if (args.wandb):
                wandb.log({"Avg Train Loss": avg_epoch_loss,
                            "Avg Train AUC" : avg_train_auc,
                            "Avg Train AP" : avg_train_ap,
                            "Avg Eval AUC" : avg_eval_auc,
                            "Avg Eval AP" : avg_eval_ap,
                            "Epoch" : epoch                            
                    })
            
            save_epoch_results(epoch, avg_eval_auc, avg_eval_ap, total_epoch_time)
            save_epoch_training(epoch, avg_train_auc, avg_train_ap, avg_epoch_loss, total_epoch_time)

        total_time = timeit.default_timer() - t_total_start
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # Save the model
        logger.info("INFO: Saving the model in {}.pth".format(self.model_path))
        # if not os.path.exists(self.model_path):
        #     os.makedirs(self.model_path)
        torch.save(best_model, "{}.pth".format(self.model_path))
        torch.save(best_mlp, "{}_mlp.pth".format(self.model_path))
        logger.info("INFO: The models is saved. Done.")

        # ------------ DEBUGGING ------------
        # save the training loss values
        # partial_results_path = f'../data/output/log/single_model/{args.dataset}/{args.model}/'
        # loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        # if os.path.exists(partial_results_path) == False:
        #     os.makedirs(partial_results_path)

        # with open(loss_log_filename, 'wb') as file:
        #     dump(train_avg_epoch_loss_dict, file)

        # Final Test
        logger.info("Best Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(best_epoch, best_eval_auc, best_eval_ap))
        # save_results(args.dataset, best_test_auc, best_test_ap, self.tgc_lr, len(self.train_shots_mask), len(self.test_shots_mask),
        #              best_epoch, total_time)






if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import set_random, logger, init_logger
    from script.utils.data_util import extra_dataset_attributes_loading
    # args.seed = 710
    args.max_epoch = 300
    # args.wandb = True
    args.min_epoch = 100
    args.model = "GCLSTM"
    args.patience = 30
    args.log_interval = 10
    args.lr = 0.0001
    # t = time.localtime()
    # args.curr_time = time.strftime("%Y-%m-%d-%H:%M:%S", t)
    
    category = "GCLSTM"
    
    for args.seed in [710]:
        # for data_count in [16]:
                num_data = 64
                datasets_package_path = "dataset_package_{}.txt".format(num_data)
                filename = "{}_{}_seed_{}".format(category, num_data, args.seed)
                init_logger('../data/output/{}/log/{}_log.txt'.format(category, filename))
                logger.info("Start Logging")
                set_random(args.seed)
                # args.nout = nout
                # args.nhid = nout
                runner = Runner()
                runner.run()
        