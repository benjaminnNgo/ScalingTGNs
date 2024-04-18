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


def save_results(dataset, test_auc, test_ap,lr,train_snapshot,test_snapshot):
    result_path = "../data/output/{}_{}_{}".format(dataset, args.results_file, args.curr_time)
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["dataset", "test_auc", "test_ap","lr","train_snapshot","test_snapshot"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append({'dataset': int(dataset), 'test_auc': test_auc, 'test_ap': test_ap,"lr":lr,"train_snapshot":train_snapshot,"test_snapshot":test_snapshot}, ignore_index=True)
    result_df.to_csv(result_path, index=False)


def save_epoch_results(epoch,test_auc, test_ap, dataset=0):
    if dataset == 0:
        result_path = "../data/output/epoch_result/average/{}_{}_{}_{}_epochResult".format(len(args.dataset),args.model,args.seed, args.curr_time)
    else:
        result_path = "../data/output/epoch_result/data/{}/{}_{}_{}_{}_epochResult".format(args.data_name[args.dataset[dataset]], 
                                                                                      len(args.dataset),
                                                                                      args.model,args.seed, 
                                                                                      args.curr_time)
    
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append({'epoch': int(epoch), 'test_auc': test_auc, 'test_ap': test_ap}, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def save_epoch_training(epoch, train_auc, train_ap, loss, dataset=0):
    if dataset == 0:
        result_path = "../data/output/training_test/average/{}_{}_{}_{}_epochResult".format(len(args.dataset),args.model,args.seed, args.curr_time)
    else:
        result_path = "../data/output/training_test/data/{}/{}_{}_{}_{}_epochResult".format(args.data_name[args.dataset[dataset]], 
                                                                                            len(args.dataset),
                                                                                            args.model,args.seed, 
                                                                                            args.curr_time)
        
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_ap"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append({'epoch': int(epoch), 'train_loss': loss, 'train_auc': train_auc, 'train_ap': train_ap}, ignore_index=True)
    result_df.to_csv(result_path, index=False)



class Runner(object):
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="ScalingTGNs",
                name="{}_{}_{}".format(len(data), args.model, args.curr_time),
                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": args.dataset,
                }
            )

        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr

        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]
        self.testlength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)] 
        self.evalLength = [math.floor(self.len[i] * args.eval_ratio) for i in range(self.num_datasets)]
        # print(self.evalLength, self.testlength)
        self.start_train = 0
        self.train_shots = [list(range(0, self.len[i] - self.testlength[i] - self.evalLength[i])) for i in range(self.num_datasets)] #Changed
        self.eval_shots = [list(range(self.len[i] - self.testlength[i] - self.evalLength[i], self.len[i] - self.testlength[i])) for i in range(self.num_datasets)] #Changed
        self.test_shots = [list(range(self.len[i] - self.testlength[i], self.len[i])) for i in range(self.num_datasets)]
        # print(len(self.train_shots[0]),len(self.train_shots[1]), len(self.eval_shots[0]), len(self.eval_shots[1]),len(self.test_shots[0]),len(self.test_shots[1]))
        # args.num_nodes = max(args.num_nodes)
        self.criterion = torch.nn.BCELoss()
        self.load_feature()
        # logger.info('INFO: total length: {}, train length: {}, test length: {}'.format(self.len, len(self.train_shots), args.testlength))

        self.model = load_model(args).to(args.device)
        self.model_path = '{}/saved_models/fm/{}.pth'.format(model_file_path, model_path)
        logger.info("The models is going to be loaded from {}".format(self.model_path))
        self.model.load_state_dict(torch.load(self.model_path))

        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 

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
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('INFO: using one-hot feature')
            args.nfeat = self.x.size(1)



    def tgclassification_test(self, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

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
                                                                                          len(self.train_shots[dataset_idx]) + 
                                                                                          len(self.eval_shots[dataset_idx])]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_test_idx + len(self.train_shots[dataset_idx]) + len(self.eval_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return auc, ap

   

    def run(self):
        """
        Run the temporal graph classification task
        """
        # load the TG-models
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification models.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()
        
        t_total_start = time.time()
        test_aucs, test_aps = [], []

        for dataset_idx in range(self.num_datasets):
            # embeddings = self.model(edge_index, self.x)

            for t_train in self.train_shots[dataset_idx]:
                with torch.no_grad():
                    edge_index, _, _, _, _, _, _ = prepare(data[dataset_idx], t_train)
                    embeddings = self.model(edge_index, self.x)
                    self.model.update_hiddens_all_with(embeddings)
        
            self.model.eval()
            test_auc, test_ap = self.tgclassification_test(self.readout_scheme, dataset_idx)
            test_aucs.append(test_auc)
            test_aps.append(test_ap)
            logger.info("Final Test Data {}: AUC: {:.4f}, AP: {:.4f}".format(
                args.data_name[args.dataset[dataset_idx]], 
                test_auc, test_ap))
            # save_results(dataset_idx, test_auc, test_ap,self.tgc_lr, len(self.train_shots[dataset_idx]),len(self.test_shots[dataset_idx]))
        logger.info("Final Test Average: AUC: {:.4f}, AP: {:.4f}".format(np.mean(test_aucs), np.mean(test_aps)))
        # save_results(self.num_datasets, test_auc, test_ap, self.tgc_lr, len(self.train_shots),len(self.test_shots))




if __name__ == '__main__':
    from script.config import args, dataset_names
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir, load_multiple_datasets
    from script.inits import prepare

    args.model = "HTGN"
    args.seed = 710
    args.max_epoch=400
    args.testlength=30
    args.lr = 0.0001
    args.log_interval=5
    args.patience = 30
    args.data_name = dataset_names
    # args.wandb = True
    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("======================================")
    print("INFO: Dataset: {}".format(args.dataset))
    print("INFO: Model: {}".format(args.model))

    # use time of run for saving results
    # t = time.localtime()
    model_path = "HTGN_seed_710_2_2024-04-16-15:30:57"
    args.dataset, data = load_multiple_datasets("two_data")
    # args.num_nodes = 15000#[data[i]['num_nodes'] for i in range(len(data))]
    print("INFO: Number of nodes:", args.num_nodes)
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.model + '_' + "Test2" + '_seed_' + str(args.seed) + '_log.txt')
    runner = Runner()
    runner.run()


# ----------------------
# commands to run:
# cd script
# python train_tgc_end_to_end.py --models=HTGN --seed=710  --dataset=dgd --max_epoch=200