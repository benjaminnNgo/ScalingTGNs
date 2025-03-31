"""
Assumption:
    Train and test temporal graph classification task 
    without having a pre-trained models

April 2024
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
from sklearn.metrics import roc_auc_score, average_precision_score

model_file_path = "/network/scratch/r/razieh.shirzadkhani/fm"
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
  

def extra_dataset_attributes_loading2(args, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for multi-network TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    partial_path = f'../data/input/raw/'
    partial_path = "/network/scratch/r/razieh.shirzadkhani/fm/fm_data/data_lt_70/all_data/raw/"
    TG_labels_data = []
    TG_feats_data = []
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
                TG_this_ts_feat = np.array([np.max(indegree_list[: , 1].astype(float)), 
                                            np.max(weighted_indegree_list[: , 1].astype(float)), 
                                            np.max(outdegree_list[: , 1].astype(float)), 
                                            np.max(weighted_outdegree_list[: , 1].astype(float))])
            elif readout_scheme == 'mean':
                TG_this_ts_feat = np.array([np.mean(indegree_list[: , 1].astype(float)), 
                                            np.mean(weighted_indegree_list[: , 1].astype(float)), 
                                            np.mean(outdegree_list[: , 1].astype(float)), 
                                            np.mean(weighted_outdegree_list[: , 1].astype(float))])
            elif readout_scheme == 'sum':
                TG_this_ts_feat = np.array([np.sum(indegree_list[: , 1].astype(float)), 
                                            np.sum(weighted_indegree_list[: , 1].astype(float)), 
                                            np.sum(outdegree_list[: , 1].astype(float)), 
                                            np.sum(weighted_outdegree_list[: , 1].astype(float))])
            else:
                TG_this_ts_feat = None
                raise ValueError("Readout scheme is Undefined!")
            
            TG_feats.append(TG_this_ts_feat)
    
        # scale the temporal graph features to have a reasonable range
        scalar = MinMaxScaler()
        TG_feats = scalar.fit_transform(TG_feats)
        TG_feats_data.append(TG_feats)
        return TG_labels_data, TG_feats_data


def save_inference_results(model_name, mode, dataset, test_auc, test_ap, bias=False):
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

        # Calculate the length of validation and test sets and split the datasets individually
        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]
        self.testLength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)] 
        self.valLength = [math.floor(self.len[i] * args.val_ratio) for i in range(self.num_datasets)]
        self.start_train = 0
        self.train_shots = [list(range(0, self.len[i] - self.testLength[i] - self.valLength[i])) for i in range(self.num_datasets)] 
        self.val_shots = [list(range(self.len[i] - self.testLength[i] - self.valLength[i], self.len[i] - self.testLength[i])) for i in range(self.num_datasets)] 
        self.test_shots = [list(range(self.len[i] - self.testLength[i], self.len[i])) for i in range(self.num_datasets)]
        self.criterion = torch.nn.BCELoss()
        self.load_feature()

        # load the TG-models
        self.model = load_model(args).to(args.device)
        self.model_path = '{}/saved_models/fm/{}/{}.pth'.format(model_file_path, category, model_path)
        print("The models is going to be loaded from {}".format(self.model_path))
        self.model.load_state_dict(torch.load(self.model_path))
        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = [], []
        for dataset in args.dataset:
            t_graph_label_i, t_graph_feat_i = extra_dataset_attributes_loading(args, dataset)
            self.t_graph_labels.append(t_graph_label_i)
            self.t_graph_feat.append(t_graph_feat_i)
        # self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args, dataset)

        # define decoder: graph classifier
        num_extra_feat = 0  
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
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                # self.x = torch.arange(args.max_node_id + 1).float().view(args.max_node_id + 1,1).to(args.device)
            args.nfeat = self.x.size(1)



    def tgclassification_test(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds, test_loss = [], [], []
        for t_test_idx, t in enumerate(self.test_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                
                edge_index = prepare(data[dataset_idx], t)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                # tg_embedding = torch.cat((tg_readout,
                #                           torch.from_numpy(self.t_graph_feat[dataset_idx][t_test_idx + 
                #                                                                           len(self.train_shots[dataset_idx])
                #                                                                            +
                #                                                   len(self.val_shots[dataset_idx])]).to(args.device)))
                tg_embedding = tg_readout.to(args.device)
                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_test_idx + len(self.train_shots[dataset_idx])
                                                                   +
                                                                  len(self.val_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
                
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap

    def tgclassification_val(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_val_idx, t in enumerate(self.val_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                
                edge_index = prepare(data[dataset_idx], t)[:3]
                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                # tg_embedding = torch.cat((tg_readout,
                #                           torch.from_numpy(self.t_graph_feat[dataset_idx][t_val_idx + 
                #                                                                           len(self.train_shots[dataset_idx])]).to(args.device)))
                tg_embedding = tg_readout.to(args.device)
                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_val_idx + len(self.train_shots[dataset_idx])].cpu().numpy())
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

                edge_index = prepare(data[dataset_idx], t_train)
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

            save_inference_results(model_path, 
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
                        edge_index = prepare(data[dataset_idx_i], t_train)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                
                val_auc, val_ap = self.tgclassification_val(self.readout_scheme, dataset_idx_i)
                save_inference_results(model_path, 
                             "Val", 
                             args.data_name[args.dataset[dataset_idx_i]], 
                             val_auc, 
                             val_ap,
                             bias=True)

                for t_val in self.val_shots[dataset_idx_i]:
                    with torch.no_grad():
                        edge_index = prepare(data[dataset_idx_i], t_val)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx_i)
                save_inference_results(model_path, 
                             "Test", 
                             args.data_name[args.dataset[dataset_idx_i]], 
                             test_auc, 
                             test_ap,
                             bias=True)

    # def forward_pass(self, )

    def test(self):
        """
        Test the temporal graph classification task
        """
        
        self.model.init_hiddens()

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)     

        # Set model and decoder to evaluation mode   
        self.model.eval()
        self.tgc_decoder.eval()
        for dataset_idx in range(self.num_datasets):
            self.model.init_hiddens()
            data_name = args.data_name[args.dataset[dataset_idx]] if args.dataset[dataset_idx] in args.data_name else args.dataset[dataset_idx]

            if args.test_bias:
                self.test_bias(dataset_idx)
            else:

                # Forwad pass through train data to get the embeddings
                for t_train in self.train_shots[dataset_idx]:
                    with torch.no_grad():
                        edge_index = prepare(data[dataset_idx], t_train)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)
                
                # Inference testing on validation set
                # val_auc, val_ap = self.tgclassification_val(self.readout_scheme, dataset_idx)
                # save_inference_results(model_path, 
                #              "Val", 
                #              data_name, 
                #              val_auc, 
                #              val_ap)
                
                # Forward pass through validation set to get the embeddings
                for t_val in self.val_shots[dataset_idx]:
                    with torch.no_grad():
                        edge_index = prepare(data[dataset_idx], t_val)
                        embeddings = self.model(edge_index, self.x)
                        self.model.update_hiddens_all_with(embeddings)

                # Inference testing on test set
                test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx)
                # print(test_auc)
                save_inference_results(model_path, 
                             "Test", 
                             data_name, 
                             test_auc, 
                             test_ap)





if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import disease_path
    from script.models.load_model import load_model
    from script.utils.data_util import load_multiple_datasets, extra_dataset_attributes_loading
    from script.utils.inits import prepare
    
    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("======================================")
    print("INFO: Dataset: {}".format(args.dataset))
    print("INFO: Model: {}".format(args.model))
    args.dataset, data = load_multiple_datasets("dataset_package_test.txt")
    for args.seed in [710, 800]:
        category = "HTGN"
        model_path = "{}_{}_seed_{}_high".format(args.model, 4, args.seed)
        runner = Runner()
        runner.test()
