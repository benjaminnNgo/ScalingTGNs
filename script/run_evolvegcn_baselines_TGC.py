import os
import sys
import random
import time
import math
import timeit

import torch
import numpy as np
from math import isnan
import torch.optim as optim
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import wandb
from pickle import dump, load
import matplotlib.pyplot as plt

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


def save_results(dataset, test_auc, test_ap,lr,train_snapshot,test_snapshot,best_epoch,time):
    partial_path =  "../data/output/single_model/"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    result_path = f"{partial_path}/{args.results_file}"
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["dataset", "test_auc", "test_ap","lr","train_snapshot","test_snapshot","seed","best_epoch","time"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append({'dataset': dataset,
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
    partial_path = "../data/output/epoch_result/single_model/"
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
class Runner(object):
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="single_models_egcn",
                # Set name of the run:
                name="{}_{}_{}".format(args.dataset, args.model, args.seed),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,

                }
            )
        self.readout_scheme = 'mean'
        self.tgc_lr = args.lr

        self.len = data['time_length']
        args.testlength = math.floor(self.len * args.test_ratio)  # Re-calculate number of test snapshots
        self.start_train = 0
        args.evalLength = math.floor(self.len * args.eval_ratio)

        self.train_shots = list(range(self.start_train, self.len - args.testlength - args.evalLength))
        self.eval_shots = list(range(self.len - args.testlength - args.evalLength, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))

        # load features
        if args.trainable_feat:
            self.x = None
            logger.info("INFO:using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                feature = sp.load_npz('../data/input/raw/disease/disease_lp.feats.npz').toarray()
                self.x = [torch.from_numpy(feature).float().to(args.device)] * len(self.train_shots)
                logger.info('using pre-defined feature')
            else:
                self.x = [torch.eye(args.num_nodes).to(args.device)] * len(self.train_shots)
                logger.info('using one-hot feature')
            args.nfeat = self.x[0].size(1)

        # load the temporal graph embedding models
        self.model = load_model(args).to(args.device)
        logger.info('INFO: total length: {}, train length: {},eval length:{}, test length: {}'.format(self.len,
                                                                                                      len(self.train_shots),
                                                                                                      args.evalLength,
                                                                                                      args.testlength))

        self.model_path = '../saved_models/single_model/{}_{}_seed_{}/'.format(args.dataset,
                                                                               args.model, args.seed)
        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout+num_extra_feat, hidden_dim_1=args.nout+num_extra_feat, 
                               hidden_dim_2=args.nout+num_extra_feat, drop=0.1)  # @NOTE: these hyperparameters may need to be changed 



    def run(self):
        # define optimizer and criterion
        optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )
        criterion = torch.nn.BCELoss()

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()

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
        # for epoch in range(1, args.max_epoch + 1):
        for epoch in range(1, 2):
            self.model.train()
            self.model.init_hiddens()
            self.tgc_decoder.train()
            t_epoch_start = timeit.default_timer()
            epoch_losses = []
            optimizer.zero_grad()
            embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in self.train_shots], self.x)
            self.model.update_hiddens_all_with(embeddings[-1])
            # compute loss
            for t, z in enumerate(embeddings):
                # edge_index = prepare(data, t)[0]
                # epoch_loss = self.loss(z, edge_index)
                pos_edge_index, neg_edge_index = prepare(data, t)[1], prepare(data, t)[2]
                # epoch_loss = self.loss(z, pos_edge_index, neg_edge_index)
                # epoch_losses.append(epoch_loss)

                # graph readout
                tg_readout = readout_function(z, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t]).to(args.device)))
                
                # graph classification
                tg_label = self.t_graph_labels[t].float().view(1, )
                tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()
                epoch_loss = criterion(tg_pred, tg_label)
                epoch_losses.append(epoch_loss)

            sum(epoch_losses).backward()
            optimizer.step()

            # update the best results.
            average_epoch_loss = np.mean([epoch_loss.item() for epoch_loss in epoch_losses])
            train_avg_epoch_loss_dict[epoch] = average_epoch_loss
            train_auc, train_ap = roc_auc_score(tg_label, tg_pred), average_precision_score(tg_label, tg_pred)
            eval_epoch, eval_auc, eval_ap = self.tgclassification_eval(epoch, self.readout_scheme)

            # Only apply early stopping when after min_epoch of training
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
            if epoch % args.log_interval == 0:
                    logger.info('==' * 27)
                    logger.info("Epoch:{}, Loss: {:.3f}, Time: {:.2f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss,
                                                                                              time.time() - t_epoch_start,
                                                                                              gpu_mem_alloc))
                    logger.info(
                        "Test: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(eval_epoch, eval_auc, eval_ap))

                    logger.info(
                        "Train: Epoch:{}, AUC: {:.4f}, AP: {:.4f}".format(epoch, train_auc, train_ap))

            if isnan(epoch_loss):
                break

            if (args.wandb):
                wandb.log({"train_loss": average_epoch_loss,
                           "eval AUC": eval_auc,
                           "eval AP": eval_ap,
                           "train AUC": train_auc,
                           "train AP": train_ap
                           })
            save_epoch_results(epoch, eval_auc, eval_ap, average_epoch_loss, train_auc, train_ap, total_epoch_time)
        total_time = timeit.default_timer() - t_total_start
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (
            args.lr, args.nhid, args.nb_window))

        # Save the model
        logger.info("INFO: Saving the models...")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(best_model, "{}{}.pth".format(self.model_path, args.model))
        torch.save(self.tgc_decoder, "{}{}_MLP.pth".format(self.model_path, args.model))
        logger.info("INFO: The models is saved. Done.")

        # ------------ DEBUGGING ------------
        # save the training loss values
        partial_results_path = f'../data/output/log/single_model/{args.dataset}/{args.model}/'
        loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        if os.path.exists(partial_results_path) == False:
            os.makedirs(partial_results_path)

        with open(loss_log_filename, 'wb') as file:
            dump(train_avg_epoch_loss_dict, file)

        # Final Test
        logger.info("Best Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(best_epoch, best_test_auc, best_test_ap))
        save_results(args.dataset, best_test_auc, best_test_ap, self.tgc_lr, len(self.train_shots),
                     len(self.test_shots), best_epoch, total_time)

    def tgclassification_eval(self, epoch):
        tg_labels, tg_preds = [], []

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in self.eval_shots],
                                    self.x)

        for t_eval, z in zip(self.eval_shots, embeddings):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                # graph readout
                tg_readout = readout_function(z, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t_eval]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_eval].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap
    def tgclassification_test(self, epoch):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in self.test_shots], self.x)

        for t_test, z in zip(self.test_shots, embeddings):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():

                # graph readout
                tg_readout = readout_function(z, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t_test]).to(args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_test].cpu().numpy())
                tg_preds.append(self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap


if __name__ == '__main__':
    from script.inits import prepare
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.config import args
    from script.utils.util import init_logger, logger
    from script.utils.data_util import loader, prepare_dir
    from script.utils.util import MLP, readout_function
    random.seed(args.seed)  # random seed
    args.model = "EGCN"
    args.dataset = "CMT0xf85feea2fdd81d51177f6b8f35f0e6734ce45f5f"
    data = loader(dataset=args.dataset)  # enron10, fb, dblp
    args.num_nodes = data['num_nodes']
    log_folder = prepare_dir(args.output_folder)  # 2.create folder
    init_logger(log_folder + args.dataset + '_seed_' + str(args.seed) + '.txt')
    run = Runner()
    run.run()
