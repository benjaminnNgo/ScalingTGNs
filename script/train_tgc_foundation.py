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
wandb.login(key="29968c684c2e412ed650ce0b5b52db584d572b86")

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
        result_path = "../data/output/epoch_result/{}_{}_{}_{}_epochResult".format(len(args.dataset),args.model,args.seed, args.curr_time)
    else:
        result_path = "../data/output/epoch_result/data_{}/{}_{}_{}_{}_epochResult".format(dataset, len(args.dataset),args.model,args.seed, args.curr_time)
    
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append({'epoch': int(epoch), 'test_auc': test_auc, 'test_ap': test_ap}, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def save_epoch_training(epoch, train_auc, train_ap, loss, dataset=0):
    if dataset == 0:
        result_path = "../data/output/training_test/{}_{}_{}_{}_epochResult".format(len(args.dataset),args.model,args.seed, args.curr_time)
    else:
        result_path = "../data/output/training_test/data_{}/{}_{}_{}_{}_epochResult".format(dataset, len(args.dataset),args.model,args.seed, args.curr_time)
        
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
        args.num_nodes = max(args.num_nodes)
        self.criterion = torch.nn.BCELoss()
        self.load_feature()
        # logger.info('INFO: total length: {}, train length: {}, test length: {}'.format(self.len, len(self.train_shots), args.testlength))

        self.model = load_model(args).to(args.device)
        self.model_path = '../saved_models/fm/{}_seed_{}_{}_{}.pth'.format(args.model, args.seed, self.num_datasets, args.curr_time)
        # logger.info("The models is going to be loaded from {}".format(self.model_path))
        # self.models.load_state_dict(torch.load(self.model_path))

        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
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

    def tgclassification_eval(self, epoch, readout_scheme, dataset_idx):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_eval_idx, t in enumerate(self.eval_shots[dataset_idx]):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_index, pos_edge, neg_edge = prepare(data[dataset_idx], t)[:3]
                # new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_eval_idx + len(self.train_shots[dataset_idx])]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_eval_idx + len(self.train_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        # print(tg_labels)
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
            for t_train_idx, t_train in enumerate(self.train_shots):
                with torch.no_grad():
                    edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t_train)
                    embeddings = self.model(edge_index, self.x)
                    self.model.update_hiddens_all_with(embeddings)

            with torch.no_grad():
                
                edge_index, pos_edge, neg_edge = prepare(data[dataset_idx], t)[:3]
                # new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[dataset_idx][t_test_idx + len(self.train_shots[dataset_idx]) + len(self.eval_shots[dataset_idx])]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[dataset_idx][t_test_idx + len(self.train_shots[dataset_idx]) + len(self.eval_shots[dataset_idx])].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())
                self.model.update_hiddens_all_with(embeddings)
        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

   

    def run(self):
        """
        Run the temporal graph classification task
        """
        # define optimizer and criterion
        optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )
        # criterion = torch.nn.BCELoss() # Moved to init function

        # load the TG-models
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification models.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()
        
        t_total_start = time.time()
        # min_loss = 10

        best_model = self.model.state_dict()
        patience = 0
        best_eval_auc = -1
        train_avg_epoch_loss_dict = {} # Stores the average of epoch loss over all datasets
        epoch_losses_per_dataset = {} # Stores each dataset loss at each epoch: {"data_1" : [e_1, e_2, ..], ...}
        test_auc, test_ap = [], []
        for epoch in range(1, args.max_epoch + 1):
            epoch_losses = [] # Used for saving average losses of datasets in each epoch
            train_aucs, train_aps, eval_aucs, eval_aps = [], [], [], []
            dataset_rnd = random.sample(range(self.num_datasets), self.num_datasets)
            for dataset_idx in dataset_rnd:
                tg_labels, tg_preds = [], []
                
                
                # initialize a list to save the dataset losses for each epoch
                if epoch == 1:
                    epoch_losses_per_dataset[dataset_idx] = []

                self.model.train()
                self.model.init_hiddens()
                t_epoch_start = time.time()
                dataset_losses = []
                for t_train_idx, t_train in enumerate(self.train_shots[dataset_idx]):
                    optimizer.zero_grad()

                    edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data[dataset_idx], t_train)
                    embeddings = self.model(edge_index, self.x)
                    
                    # graph readout
                    tg_readout = readout_function(embeddings, self.readout_scheme)
                    tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[dataset_idx][t_train_idx]).to(args.device)))
                    
                    # graph classification
                    tg_label = self.t_graph_labels[dataset_idx][t_train_idx].float().view(1, )
                    tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                    tg_labels.append(tg_label.cpu().numpy())
                    tg_preds.append(tg_pred.cpu().detach().numpy())
                    train_loss = self.criterion(tg_pred, tg_label)
                    train_loss.backward()
                    optimizer.step()
                    dataset_losses.append(train_loss.item())
                    # update the models
                    self.model.update_hiddens_all_with(embeddings)

                # --------------------Evaluation------------------------
                # Foundational model evaluation:
                avg_dataset_loss = np.mean(dataset_losses)
                epoch_losses_per_dataset[dataset_idx].append(avg_dataset_loss)
                epoch_losses.append(avg_dataset_loss)

                save_epoch_results(epoch,np.mean(eval_aucs),np.mean(eval_aps), dataset=dataset_idx+1)
                save_epoch_training(epoch,np.mean(train_aucs),np.mean(train_ap), avg_epoch_loss, dataset=dataset_idx+1)

                if isnan(train_loss):
                    print('ATTENTION: nan loss')
                    break
            
                self.model.eval()
                
                train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
                eval_epoch, eval_auc, eval_ap = self.tgclassification_eval(epoch, self.readout_scheme, dataset_idx)
                train_aucs.append(train_auc)
                train_aps.append(train_ap)
                eval_aucs.append(eval_auc)
                eval_aps.append(eval_ap)

                
                # print(avg_dataset_loss, train_auc, train_ap, eval_auc, eval_ap)
                if (args.wandb):
                    wandb.log({"Data {} Train Loss".format(dataset_idx+1): avg_dataset_loss,
                               "Data {} Train AUC".format(dataset_idx+1) : train_auc,
                               "Data {} Train AP".format(dataset_idx+1) : train_ap,
                               "Data {} Eval AUC".format(dataset_idx+1) : eval_auc,
                               "Data {} Eval AP".format(dataset_idx+1) : eval_ap                                
                        })

            avg_epoch_loss = np.mean(epoch_losses)
            train_avg_epoch_loss_dict[epoch] = avg_epoch_loss 
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            # print(avg_epoch_loss)
            # if avg_epoch_loss < min_loss:
            #         min_loss = avg_epoch_loss
            #         test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, self.readout_scheme)
            #         patience = 0
            # else:
            #         patience += 1
            #         if epoch > args.min_epoch and patience > args.patience:  
            #             print('INFO: Early Stopping...')
            #             break

            if best_eval_auc < np.mean(eval_aucs): #Use AUC as metric to define early stoping
                    patience = 0
                    best_eval_auc = np.mean(eval_aucs)
                    best_model = self.model.state_dict() #Saved the best model for testing
            else:
                    patience += 1
                    if epoch > args.min_epoch and patience > args.patience:  
                        print('INFO: Early Stopping...')
                        break


            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                          time.time() - t_epoch_start,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Train: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(epoch, np.mean(train_aucs), np.mean(train_aps)))
                logger.info(
                    "Eval: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(eval_epoch, np.mean(eval_aucs), np.mean(eval_aps)))

                

            if (args.wandb):
                wandb.log({"Avg Train Loss": avg_epoch_loss,
                            "Avg Train AUC" : np.mean(train_aucs),
                            "Avg Train AP" : np.mean(train_aps),
                            "Avg Eval AUC" : np.mean(eval_aucs),
                            "Avg Eval AP" : np.mean(eval_aps)                            
                    })
            
            
                
            save_epoch_results(epoch,np.mean(eval_aucs),np.mean(eval_aps))
            save_epoch_training(epoch,np.mean(train_aucs),np.mean(train_ap), avg_epoch_loss)
        
        logger.info('>> Total time : %6.2f' % (time.time() - t_total_start))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        logger.info("INFO: Saving the models...")
        torch.save(best_model, self.model_path)
        logger.info("INFO: The models is saved. Done.")

        # ------------ DEBUGGING ------------
        # save the training loss values
        partial_results_path = f'../data/output/log/{args.model}/'
        loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        if os.path.exists(partial_results_path)==False:
            os.makedirs(partial_results_path)

        with open(loss_log_filename, 'wb') as file:
            dump(train_avg_epoch_loss_dict, file)

        # Final Test
        # self.model.load_state_dict(best_model) #Load the best model for testing
        # test_aucs, test_aps = [], []
        # for dataset_idx in range(self.num_datasets):
        #     test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, self.readout_scheme, dataset_idx)
        #     test_aucs.append(test_auc)
        #     test_aps.append(test_ap)
        #     logger.info("Final Test Data {}: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(dataset_idx, test_epoch, test_auc, test_ap))
        #     save_results(dataset_idx, test_auc, test_ap,self.tgc_lr, len(self.train_shots[dataset_idx]),len(self.test_shots[dataset_idx]))
        # logger.info("Final Test Average: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(test_epoch, np.mean(test_aucs), np.mean(test_aps)))
        # save_results(self.num_datasets, test_auc, test_ap, self.tgc_lr, len(self.train_shots),len(self.test_shots))

if __name__ == '__main__':
    from script.config import args
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
    args.patience = 20
    # args.wandb = True
    print("INFO: >>> Temporal Graph Classification <<<")
    print("INFO: Args: ", args)
    print("======================================")
    print("INFO: Dataset: {}".format(args.dataset))
    print("INFO: Model: {}".format(args.model))

    # use time of run for saving results
    t = time.localtime()
    args.curr_time = time.strftime("%Y-%m-%d-%H:%M:%S", t)

    args.dataset, data = load_multiple_datasets("five_data")
    args.num_nodes = [data[i]['num_nodes'] for i in range(len(data))]
    print("INFO: Number of nodes:", args.num_nodes)
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.model + '_' + "Test2" + '_seed_' + str(args.seed) + '_log.txt')
    runner = Runner()
    runner.run()


# ----------------------
# commands to run:
# cd script
# python train_tgc_end_to_end.py --models=HTGN --seed=710  --dataset=dgd --max_epoch=200