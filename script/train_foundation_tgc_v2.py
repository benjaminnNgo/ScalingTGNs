"""
Assumption:
    Train foundation temporal graph classification task 
    without having a pre-trained models

June 10, 2024
"""

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

# model_file_path =  '..'
model_file_path = '/network/scratch/r/razieh.shirzadkhani/fm'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
wandb.login(key="29968c684c2e412ed650ce0b5b52db584d572b86")  

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
  

def save_train_results(epoch, train_auc, train_ap, loss, time, dataset=None):
    """
    Saving Loss, AUC and AP for training per data and on average.
    If dataset is None, average over all datasets for each epoch is saved.
    """
    if dataset is None:
        result_folder = "../data/output/training_result/average"
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                                args.seed,
                                                                                len(args.dataset))
    else:
        result_folder = "../data/output/training_result/data/{}".format(dataset)
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                                 args.seed,
                                                                                 len(args.dataset))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)
    result_df = result_df.append({'epoch': int(epoch), 'train_loss': loss, 'train_auc': train_auc, 'train_ap': train_ap, 'time': time}, ignore_index=True)
    result_df.to_csv(result_path, index=False)



def save_val_results(epoch, val_auc, val_ap, time, dataset=None):
    """
    Saving Validation AUC and Validation AP for each epoch, per data and average
    If dataset in None average is saved.
    """
    if dataset is None:
        result_folder = "../data/output/val_result/average"
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                                args.seed,
                                                                                len(args.dataset))
    else:
        result_folder = "../data/output/val_result/data/{}".format(dataset)
        result_path = result_folder + "/{}_seed_{}_{}_epochResult.csv".format(args.model,
                                                                                 args.seed,
                                                                                 len(args.dataset))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)
    result_df = result_df.append({'epoch': int(epoch), 'test_auc': val_auc, 'test_ap': val_ap, 'time': time}, ignore_index=True)
    result_df.to_csv(result_path, index=False)



class Runner(object):
    def __init__(self):
        # Initialize Wandb for this project and run
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="ScalingTGNs_features",
                name="{}_{}_{}".format(args.model, args.seed, len(data)),
                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": args.dataset,
                }
            )

        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr
        self.start_epoch = 0       

        # Calculate the length of validation and test sets and split the datasets individually
        self.num_datasets = len(data)
        self.len = [data[i]['time_length'] for i in range(self.num_datasets)]
        self.testLength = [math.floor(self.len[i] * args.test_ratio) for i in range(self.num_datasets)] 
        self.valLength = [math.floor(self.len[i] * args.val_ratio) for i in range(self.num_datasets)]
        
        self.start_train = 0
        self.train_shots = [list(range(0, self.len[i] - self.testLength[i] - self.valLength[i])) for i in range(self.num_datasets)] #Changed
        self.val_shots = [list(range(self.len[i] - self.testLength[i] - self.valLength[i], self.len[i] - self.testLength[i])) for i in range(self.num_datasets)] #Changed
        self.test_shots = [list(range(self.len[i] - self.testLength[i], self.len[i])) for i in range(self.num_datasets)]

        # Use Binary Cross Entropy for calculatin loss
        self.criterion = torch.nn.BCELoss()

        
        # Loading graph features
        # self.load_feature()
        self.model = load_model(args).to(args.device)
        self.model_path = '{}/saved_models/fm/{}/{}_{}_seed_{}'.format(model_file_path, 
                                                                        category,
                                                                        args.model,
                                                                        self.num_datasets,
                                                                        args.seed)
        
        self.model_chkp_path = '{}/saved_models/fm/{}/checkpoint/{}_{}_seed_{}'.format(model_file_path, 
                                                                        category,
                                                                        args.model,
                                                                        self.num_datasets,
                                                                        args.seed)
       
        # load the graph labels
        # self.t_graph_labels, self.t_graph_feat = [], []
        # for dataset in args.dataset:
        #     t_graph_label_i, t_graph_feat_i = extra_dataset_attributes_loading(args, dataset)
        #     self.t_graph_labels.append(t_graph_label_i)
        #     self.t_graph_feat.append(t_graph_feat_i)
        # a, b = extra_dataset_attributes_loading2(args)
        # print(b)
        # print(self.t_graph_feat)
        
        # self.t_graph_labels = t_graph_labels
        # self.t_graph_feat = t_graph_feat
        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 

        self.optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr)
        logger.info("{}".format(self.model_chkp_path))
        # total_params = [p.numel() for p in self.tgc_decoder.parameters()]
        # print(f"Number of parameters: {total_params}")
        # total_params2 = [p.numel() for p in self.model.parameters()]
        # print(f"Number of parameters2: {total_params2}")
        if os.path.exists("{}.pth".format(self.model_chkp_path)):
            logger.info("INFO: Model already exist and will be loaded from {}".format(self.model_chkp_path))
            checkpoint = torch.load("{}.pth".format(self.model_chkp_path))
            self.model.load_state_dict(checkpoint['model_state_dict'])

            mlp_checkpoint = torch.load("{}_mlp.pth".format(self.model_chkp_path))
            self.tgc_decoder.load_state_dict(mlp_checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']
            logger.info("INFO: Epochs will start from: {}".format(self.start_epoch))
            
        
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
                print("here")
                self.x = torch.eye(args.num_nodes).to(args.device)
                # self.x = np.fill_diagonal(torch.zeros(args.num_nodes, args.num_nodes), 
                #                           args.node_ids).to(args.device)
                logger.info('INFO: using one-hot feature')
            # args.nfeat = self.x.size(1)

    def tgclassification_val(self, epoch, readout_scheme, dataset_idx):
        """
        Final inference on the validation set
        """
        tg_labels, tg_preds = [], []

        for t_val_idx, t in enumerate(self.val_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                self.x = torch.from_numpy(self.t_node_feat[t_val_idx]).to(torch.float32).to(args.device)
                edge_index = prepare(data[dataset_idx], t)[:3]
                # new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_val_idx + len(self.train_shots[dataset_idx])]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_val_idx + len(self.train_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

    def tgclassification_test(self, epoch, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_test_idx, t in enumerate(self.test_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                self.x = torch.from_numpy(self.t_node_feat[t_test_idx]).to(torch.float32).to(args.device)
                edge_index = prepare(data[dataset_idx], t)[:3]

                embeddings = self.model(edge_index, list(self.x))

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_test_idx + len(self.train_shots[dataset_idx])]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_test_idx + len(self.train_shots[dataset_idx]) +
                                                                  len(self.val_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap
    

    def run(self):
        """
        Run the temporal graph classification task
        """
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification models.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        # Set the model and decoder to train mode
        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()
        t_total_start = timeit.default_timer()
        

        t_total_start = timeit.default_timer()
        best_model = self.model.state_dict()
        patience = 0
        best_val_auc = -1
        
        for epoch in range(self.start_epoch + 1, args.max_epoch + 1):
            print("Epoch: ", epoch)
            t_epoch_start = timeit.default_timer()
            epoch_losses = [] # Saving average losses of datasets in each epoch
            train_aucs, train_aps, val_aucs, val_aps = [], [], [], []
            test_aucs, test_aps = [], []

            # Shuffling order of datasets for each epoch
            dataset_rnd = random.sample(range(self.num_datasets), self.num_datasets)
            for dataset_idx in dataset_rnd:
                self.t_graph_labels, self.t_graph_feat, self.t_node_feat = extra_node_attributes_loading(args, args.dataset[dataset_idx])
                tg_labels, tg_preds = [], []
                self.model.train()
                self.tgc_decoder.train()
                self.model.init_hiddens()
                dataset_losses = [] # Store losses for each dataset in one epoch

                for t_train_idx, t_train in enumerate(self.train_shots[dataset_idx]):
                    self.x = torch.from_numpy(self.t_node_feat[t_train_idx]).to(torch.float32).to(args.device)
                    
                    self.optimizer.zero_grad()
                    edge_index = prepare(data[dataset_idx], t_train)
                    embeddings = self.model(edge_index, self.x)
                    
                    # graph readout
                    tg_readout = readout_function(embeddings, self.readout_scheme)
                    tg_embedding = torch.cat((tg_readout, 
                                              torch.from_numpy(self.t_graph_feat[t_train_idx]).to(args.device)))
                    
                    
                    # graph classification
                    tg_label = self.t_graph_labels[t_train_idx].float().view(1, )
                    tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                    tg_labels.append(tg_label.cpu().numpy())
                    tg_preds.append(tg_pred.cpu().detach().numpy())
                    train_loss = self.criterion(tg_pred, tg_label)
                    train_loss.backward()
                    self.optimizer.step()
                    dataset_losses.append(train_loss.item())
                    # update the models
                    self.model.update_hiddens_all_with(embeddings)

                if isnan(train_loss):
                    print('ATTENTION: nan loss')
                    break

                # --------------------Evaluation------------------------
                # Foundational model evaluation:
                avg_dataset_loss = np.mean(dataset_losses)
                epoch_losses.append(avg_dataset_loss)

                self.model.eval()
                self.tgc_decoder.eval()
                train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
                val_epoch, val_auc, val_ap = self.tgclassification_val(epoch, self.readout_scheme, dataset_idx)
                train_aucs.append(train_auc)
                train_aps.append(train_ap)
                val_aucs.append(val_auc)
                val_aps.append(val_ap)
                # Save train and validation results for each dataset
                save_val_results(epoch,val_auc,val_ap, 0, dataset=args.dataset[dataset_idx])
                save_train_results(epoch,train_auc,train_ap, avg_dataset_loss, 0, dataset=args.dataset[dataset_idx])
                
                

            # Calculte average of Loss, AUC and AP for train and validations among datasets for each epoch
            avg_epoch_loss = np.mean(epoch_losses)
            avg_train_auc = np.mean(train_aucs)
            avg_train_ap = np.mean(train_aps)
            avg_val_auc = np.mean(val_aucs)
            avg_val_ap = np.mean(val_aps)
            total_epoch_time = timeit.default_timer() - t_epoch_start

            # Saving model checkpoint:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()}, 
                        "{}.pth".format(self.model_chkp_path))
            
            torch.save({'epoch': epoch,
                        'model_state_dict': self.tgc_decoder.state_dict()}, 
                        "{}_mlp.pth".format(self.model_chkp_path))
            
            # Saving best model based on validation AUC
            if best_val_auc < avg_val_auc or epoch <= args.min_epoch: 
                    patience = 0
                    best_val_auc = avg_val_auc
                    best_model = self.model.state_dict() #Saved the best model for testing
                    best_mlp = self.tgc_decoder.state_dict()
                    best_test_results = [test_aucs, test_aps]
                    best_epoch = epoch
            else: # Check for early stopping
                if best_val_auc - np.mean(val_aucs) > 0.005:
                    patience += 1
                    print("patience at epoch {}: {}".format(epoch, patience))
                if patience > args.patience:  
                    logger.info("Early stopping at epoch: {}".format(epoch))
                    break

            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                          total_epoch_time,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Train: Epoch:{}, AUC: {:.4f}, AP: {:.4f}, Loss: {:.4f}".format(epoch, avg_train_auc, avg_train_ap, avg_epoch_loss))
                logger.info(
                    "Val: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(val_epoch, avg_val_auc, avg_val_ap))

                
            if (args.wandb):
                wandb.log({"Avg Train Loss": avg_epoch_loss,
                            "Avg Train AUC" : avg_train_auc,
                            "Avg Train AP" : avg_train_ap,
                            "Avg Val AUC" : avg_val_auc,
                            "Avg Val AP" : avg_val_ap,
                            "Epoch" : epoch                            
                    })
            
            
            # Save average train and validation results for each epoch
            save_val_results(epoch, avg_val_auc, avg_val_ap, total_epoch_time)
            save_train_results(epoch, avg_train_auc, avg_train_ap, avg_epoch_loss, total_epoch_time)

        total_time = timeit.default_timer() - t_total_start
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # Saving best model and decoder
        logger.info("INFO: Saving best model from epoch {}...".format(best_epoch))
        logger.info("File name: {}_seed_{}_{}.pth".format(args.model, args.seed, self.num_datasets))
        torch.save(best_model, "{}.pth".format(self.model_path))
        torch.save(best_mlp, "{}_mlp.pth".format(self.model_path))
        logger.info("Best test results: {}".format(best_test_results))
        


if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.utils.data_util import load_multiple_datasets, extra_dataset_attributes_loading, loader, extra_node_attributes_loading
    from script.utils.inits import prepare
    
    args.model = "HTGN"
    args.seed = 800
    args.max_epoch=300
    args.lr = 0.0001
    args.log_interval=10
    args.patience = 30
    args.min_epoch = 100
    # args.wandb = True
    print("INFO: >>> Temporal Graph Classification <<<")
    print("======================================")
    print("INFO: Model: {}".format(args.model))
    
    # use time of run for saving results
    t = time.localtime()
    args.curr_time = time.strftime("%Y-%m-%d-%H:%M:%S", t)

    args.dataset, data = load_multiple_datasets("dataset_package_4.txt")
    # num_nodes = [data[i]['num_nodes'] for i in range(len(data))]
    # args.num_nodes = max(num_nodes)
    # args.num_nodes = 100
    category = "features" #"nout" #"rand_data" "HTGN"
    # data_number = 3
    for nout in [32]:
        # args.dataset, data = load_multiple_datasets("{}/dataset_package_16_{}.txt".format(category, data_number))            
        # init_logger('../data/output/{}/log/{}_{}_seed_{}_{}_log.txt'.format(category, args.model, len(args.dataset), args.seed, data_number))
        init_logger('../data/output/{}/log/{}_{}_seed_{}_{}_log.txt'.format(category, args.model, args.seed, len(args.dataset), nout))
        set_random(args.seed)
        # args.nout = nout
        # args.nhid = nout
        runner = Runner()
        runner.run()
    # import scipy.sparse as sp
    # a = sp.load_npz("/home/mila/r/razieh.shirzadkhani/ScalingTGNs/data/input/raw/disease/disease_lp.feats.npz").toarray()
    # print(a[0])
    # print(a.shape)
    # edgelist_df = pd.read_csv("/network/scratch/r/razieh.shirzadkhani/fm/fm_data/data_lt_70/all_data/raw/edgelists/unnamedtoken222080x7e77dcb127f99ece88230a64db8d595f31f1b068_edgelist.txt")
    # unique_nodes = pd.unique(edgelist_df[['source', 'destination']].values.ravel('K'))
    # num_unique_nodes = len(unique_nodes)
    # print(num_unique_nodes)

    # data = loader(dataset="unnamedtoken222080x7e77dcb127f99ece88230a64db8d595f31f1b068", neg_sample=args.neg_sample)
    # print(data["num_nodes"])