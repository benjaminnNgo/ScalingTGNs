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

def data_loader_geometric_temporal(dataset):
    """
        Data loader for traing models with PyTorch geometric temporal
        data is a dictionary has following attributes:
            - edge index for each snapshot ( to get edge index for snapshot 0: data['edge_index'][0])
            - edge attribute for each snapshot ( similar above)
            - time length : number of snapshots
            - number of nodes
    """
    partial_path = f'../data/input/raw/'

    data_root = '../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}_pyTorchGeoTemp.data'.format(dataset)  # the data will be saved here after generation.
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
    partial_path = "../data/output/single_model_gclstm_test/"
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

def save_epoch_results(epoch,test_auc, test_ap,loss,train_auc,train_ap,time):
    partial_path = "../data/output/epoch_result/single_model_gclstm_test/"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)

    result_path = "{}/{}_{}_{}_epochResult".format(partial_path,args.dataset,args.model,args.seed)
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap","loss","train_auc","train_ap","time"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'epoch': epoch,
                                   'test_auc': test_auc,
                                   'test_ap': test_ap,
                                   "loss":loss,
                                   "train_auc":train_auc,
                                   "train_ap":train_ap,
                                   "time":time
                                   }, ignore_index=True)
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
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="gclstm_test_set",
                # Set name of the run:
                name="{}_{}_{}".format(args.dataset, args.model, args.seed),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,

                }
            )

        self.model_path = '../saved_models/single_model_gclstm_test/{}_{}_seed_{}/'.format(args.dataset,
                                                                               args.model, args.seed)
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)
        self.data = data_loader_geometric_temporal(args.dataset)
        self.edge_idx_list = self.data['edge_index']
        self.edge_att_list = self.data['edge_attribute']
        self.num_nodes = self.data['num_nodes'] + 1
        args.num_nodes = self.data['num_nodes'] + 1
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



        # self.node_feat = torch.eye(5351).to(args.device)
        self.node_feat_dim = self.node_feat.size(1)  # @TODO: Replace with args to config it easily
        self.edge_feat_dim = 1 #@TODO: Replace with args to config it easily
        self.hidden_dim = args.nhid

        self.model = RecurrentGCN(node_feat_dim=self.node_feat_dim, hidden_dim=self.hidden_dim).to(args.device)


        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=self.hidden_dim + num_extra_feat, hidden_dim_1=self.hidden_dim + num_extra_feat,
                          hidden_dim_2=self.hidden_dim + num_extra_feat,
                          drop=0.1).to(args.device)  # @NOTE: these hyperparameters may need to be changed

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

    def tgclassification_val(self,epoch,readout_scheme):
        self.detach_hidden()

        tg_labels, tg_preds = [], []
        for t_eval_idx in self.eval_shots_mask:
            self.eval()
            with torch.no_grad():
                edge_idx = self.edge_idx_list[t_eval_idx].to(args.device)
                edge_att = self.edge_att_list[t_eval_idx].float().to(args.device)

                self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)


                # graph readout
                tg_readout = readout_function(self.h, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_eval_idx]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_eval_idx].cpu().numpy())
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
        best_MLP = self.tgc_decoder.state_dict()

        # For eval model to choose the best model
        best_epoch = -1
        patience = 0
        best_eval_auc = -1  # Set previous evaluation result to very small number
        best_test_auc = -1
        best_test_ap = -1


        t_total_start = timeit.default_timer()
        for epoch in range(1, args.max_epoch + 1):
            t_epoch_start = timeit.default_timer()
            self.init_hidden()
            optimizer.zero_grad()
            self.train()

            tg_labels = []
            tg_preds = []
            epoch_losses = []
            for snapshot_idx in self.train_shots_mask:
                optimizer.zero_grad()

                #Get edge list and edge attributes
                edge_idx = self.edge_idx_list[snapshot_idx].to(args.device)
                edge_att = self.edge_att_list[snapshot_idx].float().to(args.device)

                self.h, self.h_0, self.c_0 = self.model(self.node_feat, edge_idx, edge_att, self.h_0, self.c_0)
                tg_readout = readout_function(self.h, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[snapshot_idx]).to(args.device)))

                # graph classification
                tg_label = self.t_graph_labels[snapshot_idx].float().view(1, )
                tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                t_loss = criterion(tg_pred, tg_label)
                t_loss.backward()
                optimizer.step()

                tg_labels.append(tg_label.cpu().numpy())
                tg_preds.append(tg_pred.cpu().detach().numpy())
                epoch_losses.append(t_loss.item())
                self.detach_hidden()

            avg_epoch_loss = np.mean(epoch_losses)
            train_avg_epoch_loss_dict[epoch] = avg_epoch_loss
            train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
            eval_epoch, eval_auc, eval_ap = self.tgclassification_val(epoch, self.readout_scheme)

            # Only apply early stopping when after min_epoch of training
            patience = 0 #Reset patience
            if best_eval_auc < eval_auc:  # Use AUC as metric to define early stoping
                patience = 0
                best_model = self.model.state_dict()  # Saved the best model for testing
                best_MLP = self.tgc_decoder.state_dict()

                best_eval_auc = eval_auc
                best_epoch, best_test_auc, best_test_ap = self.tgclassification_test(epoch, self.readout_scheme)
            else:
                if epoch < args.min_epoch:  # If it is less than min_epoch, reset best AUC to current AUC and save current model as best model
                    patience = 0
                    best_eval_auc = eval_auc
                    best_model = self.model.state_dict()
                    best_MLP = self.tgc_decoder.state_dict()
                    best_epoch, best_test_auc, best_test_ap = self.tgclassification_test(epoch, self.readout_scheme)

                    best_epoch = epoch
                if best_eval_auc - eval_auc > 0.05:
                    patience += 1
                if epoch > args.min_epoch and patience > args.patience:  # NOTE: args.min_epoch prevents it from stopping early in most cases
                    print('INFO: Early Stopping...')
                    logger.info("Early stopping at epoch: {}".format(epoch))
                    break

            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            total_epoch_time = timeit.default_timer() - t_epoch_start

            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 30)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                          total_epoch_time,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Test: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(eval_epoch, eval_auc, eval_ap))

                logger.info(
                    "Train: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(epoch, train_auc, train_ap))

            if isnan(t_loss):
                print('ATTENTION: nan loss')
                break
            if (args.wandb):
                wandb.log({"train_loss": avg_epoch_loss,
                           "eval AUC": eval_auc,
                           "eval AP": eval_ap,
                           "train AUC": train_auc,
                           "train AP": train_ap
                           })
            save_epoch_results(epoch, eval_auc, eval_ap, avg_epoch_loss, train_auc, train_ap, total_epoch_time)
            # save_epoch_traing(epoch,train_auc,train_ap)

        total_time = timeit.default_timer() - t_total_start
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # Save the model
        logger.info("INFO: Saving the models...")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(best_model, "{}{}.pth".format(self.model_path, args.model))
        torch.save(best_MLP, "{}{}_MLP.pth".format(self.model_path, args.model))
        logger.info("INFO: The models is saved. Done.")

        # ------------ DEBUGGING ------------
        # save the training loss values
        partial_results_path = f'../../data/output/log/single_model_gclstm_test/{args.dataset}/{args.model}/'
        loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        if os.path.exists(partial_results_path) == False:
            os.makedirs(partial_results_path)

        with open(loss_log_filename, 'wb') as file:
            dump(train_avg_epoch_loss_dict, file)

        # Final Test
        logger.info("Best Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(best_epoch, best_test_auc, best_test_ap))
        save_results(args.dataset, best_test_auc, best_test_ap, self.tgc_lr, len(self.train_shots_mask), len(self.test_shots_mask),
                     best_epoch, total_time)






if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare

    # args.seed = 710
    # args.max_epoch = 250
    # args.wandb = True
    # args.min_epoch = 100
    # args.dataset = 'unnamedtoken216620x429881672b9ae42b8eba0e26cd9c73711b891ca5'
    # args.model = "GCLSTM"
    # args.log_interval = 10
    # args.lr = 0.00015
    # set_random(args.seed)
    # init_logger(
    #     prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(args.seed) + '_log.txt')
    # runner = Runner()
    # runner.run()

    datasets = [
        # "unnamedtoken223250xf2ec4a773ef90c58d98ea734c0ebdb538519b988",
        # "unnamedtoken222800xa49d7499271ae71cd8ab9ac515e6694c755d400c",
        # "unnamedtoken223030x4ad434b8cdc3aa5ac97932d6bd18b5d313ab0f6f",
        # "unnamedtoken220850x9fa69536d1cda4a04cfb50688294de75b505a9ae",
        # "unnamedtoken220220xade00c28244d5ce17d72e40330b1c318cd12b7c3",
        # "unnamedtoken223090xc4ee0aa2d993ca7c9263ecfa26c6f7e13009d2b6",
        # "unnamedtoken221090x5de8ab7e27f6e7a1fff3e5b337584aa43961beef",
        # "unnamedtoken220240x235c8ee913d93c68d2902a8e0b5a643755705726",
        # "unnamedtoken221150xa2cd3d43c775978a96bdbf12d733d5a1ed94fb18",
        # "unnamedtoken218340xaa6e8127831c9de45ae56bb1b0d4d4da6e5665bd",
        # "unnamedtoken220960x4da27a545c0c5b758a6ba100e3a049001de870f5",
        # "unnamedtoken217780x7dd9c5cba05e151c895fde1cf355c9a1d5da6429",
        # "unnamedtoken220250xa71d0588eaf47f12b13cf8ec750430d21df04974",
        # "unnamedtoken218270x5026f006b85729a8b14553fae6af249ad16c9aab",
        # "unnamedtoken221900x49642110b712c1fd7261bc074105e9e44676c68f",
        # "unnamedtoken216900x9e32b13ce7f2e80a01932b42553652e053d6ed8e",
        # "unnamedtoken218450x221657776846890989a759ba2973e427dff5c9bb",
        # "TRAC0xaa7a9ca87d3694b5755f213b5d04094b8d0f0a6f",
        "unnamedtoken220280xcf3c8be2e2c42331da80ef210e9b1b307c03d36a",
    ]
    # datasets = ['unnamedtoken216300xcc4304a31d09258b0029ea7fe63d032f52e44efe']
    seeds = [710,720,800]

    for dataset in datasets:
        for seed in seeds:
            # print(dataset,seed)
            args.seed = seed
            args.max_epoch = 250
            args.wandb = True
            args.min_epoch = 100
            args.dataset = dataset
            args.model = "GCLSTM"
            args.log_interval = 10
            args.lr = 0.00015
            set_random(args.seed)
            init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
            runner = Runner()
            runner.run()
            try:
                wandb.finish()
            except Exception as e:
                print("Can't finish run with wandb: {}".format(e))