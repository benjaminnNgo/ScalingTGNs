"""
Assumption:
    Train and test temporal graph classification task 
    without having a pre-trained models

July 14, 2023
"""

import os
import math
import sys
import time
import torch
import copy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import matplotlib.pyplot as plt
import random
import wandb
# wandb.login(key="29968c684c2e412ed650ce0b5b52db584d572b86")
model_file_path = '/network/scratch/r/razieh.shirzadkhani/fm'
# model_file_path = ".."
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
  

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
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


def readout_function(embeddings, readout_scheme='mean'):
    """
    Read out function to generate a representation for the whole graph
    reference:    
    https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification/blob/master/readouts/basic_readout.py
    """
    # note: x.size(): [#nodes, args.n_out]
    if readout_scheme == 'max':
      readout = torch.max(embeddings, dim=0)[0].squeeze() # max readout
    elif readout_scheme == 'mean':
      readout = torch.mean(embeddings, dim=0).squeeze() # mean readout
    elif readout_scheme == 'sum':
      readout = torch.sum(embeddings, dim=0).squeeze() # sum readout
    else:
      readout = None
      raise ValueError("Readout Method Undefined!")
    return readout
  

def extra_dataset_attributes_loading(args, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    partial_path = f'../data/input/raw/'
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


def save_results(model_name, mode, dataset, test_auc, test_ap, bias=False):
    if bias:
        result_path = "../data/output/{}/test_result/{}_results_bias.csv".format(category, model_name)
    else:
        result_path = "../data/output/{}/test_result/{}_results.csv".format(category, model_name)
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["dataset", "mode", "test_auc", "test_ap"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'dataset': dataset, 'mode' : mode, 'test_auc': test_auc, 'test_ap': test_ap}, ignore_index=True)
    result_df.to_csv(result_path, index=False)


class Runner(object):
    def __init__(self):

        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr

        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]
        self.testlength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)] 
        self.evalLength = [math.floor(self.len[i] * args.eval_ratio) for i in range(self.num_datasets)]
        self.start_train = 0
        self.train_shots = [list(range(0, self.len[i] - self.testlength[i] - self.evalLength[i])) for i in range(self.num_datasets)] #Changed
        self.val_shots = [list(range(self.len[i] - self.testlength[i] - self.evalLength[i], self.len[i] - self.testlength[i])) for i in range(self.num_datasets)] #Changed
        self.test_shots = [list(range(self.len[i] - self.testlength[i] - self.evalLength[i], self.len[i])) for i in range(self.num_datasets)]
        self.criterion = torch.nn.BCELoss()
        self.load_feature()

        self.model = load_model(args).to(args.device)
        self.model_path = '{}/saved_models/fm/{}/{}.pth'.format(model_file_path, category, model_path)
        logger.info("The models is going to be loaded from {}".format(self.model_path))
        self.model.load_state_dict(torch.load(self.model_path))
        # load the graph labels
        # self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)
        self.t_graph_labels, self.t_graph_feat = t_graph_labels, t_graph_feat
        # define decoder: graph classifier
        num_extra_feat = 4  
        self.mlp_path = '{}/saved_models/fm/{}/{}_mlp.pth'.format(model_file_path, category, model_path)
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 
        self.tgc_decoder.load_state_dict(torch.load(self.mlp_path))

        self.optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("INFO: Using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('INFO: using pre-defined feature')
            else:
                # self.x = torch.arange(args.max_node_id + 1).float().view(args.max_node_id + 1,1).to(args.device)

                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('INFO: using one-hot feature')
            args.nfeat = self.x.size(1)



    def tgclassification_test(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds, test_loss = [], [], []
        # self.test_shots = [list(range(self.len[i] - self.testlength[i] -self.evalLength[i], self.len[i])) for i in range(self.num_datasets)]
        for t_test_idx, t in enumerate(self.test_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                
                edge_index, pos_edge, neg_edge = prepare(data[dataset_idx], t)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_test_idx + 
                                                                                          len(self.train_shots[dataset_idx])]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_test_idx + len(self.train_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
                # test_loss.append(self.criterion(tg_preds[-1], tg_labels[-1]))
        # total_test_loss = np.men(test_loss)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap

    def tgclassification_val(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_eval_idx, t in enumerate(self.val_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                
                edge_index, pos_edge, neg_edge = prepare(data[dataset_idx], t)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_eval_idx + 
                                                                                          len(self.train_shots[dataset_idx])]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_eval_idx + len(self.train_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap

    def test_bias(self, dataset_idx):
        # for dataset_idx in range(self.num_datasets):
            # inference_model = self.model.state_dict()
            inference_model = copy.deepcopy(self.model)
            inference_mlp = copy.deepcopy(self.tgc_decoder)
            epoch_losses = []
            tg_labels = []
            tg_preds =  []
            inference_model.train()
            inference_mlp.train()
            # forward_pass_data = self.train_shots[dataset_idx] + self.val_shots[dataset_idx]
            for t_train_idx, t_train in enumerate(self.train_shots[dataset_idx]):
                self.optimizer.zero_grad()

                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data[dataset_idx], t_train)
                embeddings = inference_model(edge_index, self.x)
                
                # graph readout
                tg_readout = readout_function(embeddings, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, 
                                            torch.from_numpy(self.t_graph_feat[dataset_idx][t_train_idx]).to(args.device)))
                
                # graph classification
                tg_label = self.t_graph_labels[dataset_idx][t_train_idx].float().view(1, )
                tg_pred = inference_mlp(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                tg_labels.append(tg_label.cpu().numpy())
                tg_preds.append(tg_pred.cpu().detach().numpy())
                train_loss = self.criterion(tg_pred, tg_label)
                train_loss.backward()
                self.optimizer.step()

                # update the models
                inference_model.update_hiddens_all_with(embeddings)

            save_results(model_path, 
                             args.data_name[args.dataset[dataset_idx]], 
                             args.data_name[args.dataset[dataset_idx]], 
                             args.data_name[args.dataset[dataset_idx]], 
                             args.data_name[args.dataset[dataset_idx]],
                             bias=True)
            inference_model.eval()
            inference_mlp.eval()
            for dataset_idx_i in range(self.num_datasets):
                for t_train in self.train_shots[dataset_idx_i]:
                    with torch.no_grad():
                        edge_index, _, _, _, _, _, _ = prepare(data[dataset_idx_i], t_train)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                
                # val_auc, val_ap = self.tgclassification_val(self.readout_scheme, dataset_idx_i)
                # logger.info("Final Val Data {}: AUC: {:.4f}, AP: {:.4f}".format(
                #     args.data_name[args.dataset[dataset_idx_i]], 
                #     val_auc, val_ap))
                # save_results(model_path, 
                #              "Val", 
                #              args.data_name[args.dataset[dataset_idx_i]], 
                #              val_auc, 
                #              val_ap,
                #              bias=True)

                for t_val in self.val_shots[dataset_idx_i]:
                    with torch.no_grad():
                        edge_index, _, _, _, _, _, _ = prepare(data[dataset_idx_i], t_val)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx_i)
                save_results(model_path, 
                             "Test", 
                             args.data_name[args.dataset[dataset_idx_i]], 
                             test_auc, 
                             test_ap,
                             bias=True)

    # def forward_pass(self, )

    def run(self):
        """
        Run the temporal graph classification task
        """
        # load the TG-models
        self.model.init_hiddens()
        logger.info("Start testing the temporal graph classification models.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)        
        self.model.eval()
        self.tgc_decoder.eval()
        for dataset_idx in range(self.num_datasets):
            data_name = args.data_name[args.dataset[dataset_idx]] if args.dataset[dataset_idx] in args.data_name else args.dataset[dataset_idx]

            if args.test_bias:
                self.test_bias(dataset_idx)
            else:
                # Passing through train data to get the embeddings
                for t_train in self.train_shots[dataset_idx]:
                    with torch.no_grad():
                        edge_index, _, _, _, _, _, _ = prepare(data[dataset_idx], t_train)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                
                # val_auc, val_ap = self.tgclassification_val(self.readout_scheme, dataset_idx)
                # logger.info("Final Val Data {}: AUC: {:.4f}, AP: {:.4f}".format(
                #     data_name, 
                #     val_auc, val_ap))
                # save_results(model_path, 
                #              "Val", 
                #              data_name, 
                #              val_auc, 
                #              val_ap)
                
                # Passing through validation set to get the embeddings
                for t_train in self.val_shots[dataset_idx]:
                    with torch.no_grad():
                        edge_index, _, _, _, _, _, _ = prepare(data[dataset_idx], t_train)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)

                
                test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx)
                logger.info("Final Test Data {}: AUC: {:.4f}, AP: {:.4f}".format(
                    data_name, 
                    test_auc, test_ap))
                save_results(model_path, 
                             "Test", 
                             data_name, 
                             test_auc, 
                             test_ap)





if __name__ == '__main__':
    from script.utils.config import args, dataset_names
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.utils.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir, load_multiple_datasets
    from script.utils.inits import prepare

    args.data_name = dataset_names
    args.test_bias = False
    
    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("======================================")
    print("INFO: Dataset: {}".format(args.dataset))
    print("INFO: Model: {}".format(args.model))
    args.dataset, data = load_multiple_datasets("dataset_package_test.txt")
    t_graph_labels, t_graph_feat = extra_dataset_attributes_loading(args)
    # num_nodes_per_data = [data[i]['num_nodes'] for i in range(len(data))]
    # args.num_nodes = 183714#max(num_nodes_per_data)
    # args.max_node_id = 183713
    # args.num_nodes = args.max_node_id + 1
    category = "nout"
    
    for n_data in [1]:
        for seed in [710, 720]:
            for nout in [32, 64]:
                args.nhid = nout
                args.nout = nout
                # model_path = "rand_data/rr/{}".format(n_data)
                # model_path = "rand_data/rr/{}_{}_seed_{}_{}".format(args.model, n_data, seed, nr)
                model_path = "HTGN_16_seed_{}_{}".format(seed, nout)
                # model_path = "node_id/HTGN_seed_{}_{}".format(seed, n_data)
                # result_path = "../data/output/{}/test_result/{}_results.csv".format(category, model_path)
                # print(result_path)
                runner = Runner()
                runner.run()

        for seed in [800]:
            for nout in [64]:
                args.nhid = nout
                args.nout = nout
                model_path = "HTGN_16_seed_{}_{}".format(seed, nout)
                runner = Runner()
                runner.run()

        for seed in [710]:
            for nout in [128]:
                args.nhid = nout
                args.nout = nout
                model_path = "HTGN_16_seed_{}_{}".format(seed, nout)
                runner = Runner()
                runner.run()


    # average_results(result_path)