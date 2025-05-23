"""
Assumption:
    Train and test temporal graph classification task 
    without having a pre-trained models

June 10, 2024
"""
import math
import os
import sys
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import matplotlib.pyplot as plt
import wandb
import warnings
import timeit

# Filter out a specific warning
warnings.filterwarnings("ignore")

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
        readout = torch.max(embeddings, dim=0)[0].squeeze()  # max readout
    elif readout_scheme == 'mean':
        readout = torch.mean(embeddings, dim=0).squeeze()  # mean readout
    elif readout_scheme == 'sum':
        readout = torch.sum(embeddings, dim=0).squeeze()  # sum readout
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


def save_results(dataset, test_auc, test_ap, lr, train_snapshot, test_snapshot, best_epoch, time):
    partial_path = "../data/output/single_model/"
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


def save_epoch_results(epoch, test_auc, test_ap, loss, train_auc, train_ap, time):
    partial_path = "../data/output/epoch_result/single_model/"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)

    result_path = "{}/{}_{}_{}_epochResult".format(partial_path, args.dataset, args.model, args.seed)
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=["epoch", "test_auc", "test_ap", "loss", "train_auc", "train_ap", "time"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'epoch': epoch,
                                  'test_auc': test_auc,
                                  'test_ap': test_ap,
                                  "loss": loss,
                                  "train_auc": train_auc,
                                  "train_ap": train_ap,
                                  "time": time
                                  }, ignore_index=True)
    result_df.to_csv(result_path, index=False)


class Runner(object):
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="single_models",
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
        args.evalLength = math.floor(self.len * args.val_ratio)

        self.train_shots = list(range(self.start_train, self.len - args.testlength - args.evalLength))
        self.eval_shots = list(range(self.len - args.testlength - args.evalLength, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))

        self.load_feature()
        logger.info('INFO: total length: {}, train length: {},eval length:{}, test length: {}'.format(self.len,
                                                                                                      len(self.train_shots),
                                                                                                      args.evalLength,
                                                                                                      args.testlength))

        self.model = load_model(args).to(args.device)
        self.model_path = '../saved_models/single_model/{}_{}_seed_{}/'.format(args.dataset,
                                                                                        args.model, args.seed)
        # logger.info("The models is going to be loaded from {}".format(self.model_path))
        # self.models.load_state_dict(torch.load(self.model_path))

        # load the graph labels
        self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=args.nout + num_extra_feat, hidden_dim_1=args.nout + num_extra_feat,
                               hidden_dim_2=args.nout + num_extra_feat,
                               drop=0.1)  # @NOTE: these hyperparameters may need to be changed

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

    def tgclassification_eval(self, epoch, readout_scheme):
        """
        Final inference on the validation set
        """
        tg_labels, tg_preds = [], []

        for t_eval_idx, t in enumerate(self.eval_shots):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
                new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_eval_idx + len(self.train_shots)]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(self.t_graph_labels[t_eval_idx + len(self.train_shots)].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

                # update the models
                self.model.update_hiddens_all_with(embeddings)  # Added

        auc, ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
        return epoch, auc, ap

    def tgclassification_test(self, epoch, readout_scheme):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []

        for t_test_idx, t in enumerate(self.test_shots):
            self.model.eval()
            self.tgc_decoder.eval()
            with torch.no_grad():
                edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
                new_pos_edge, new_neg_edge = prepare(data, t)[-2:]

                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, readout_scheme)
                tg_embedding = torch.cat((tg_readout,
                                          torch.from_numpy(self.t_graph_feat[t_test_idx + len(self.train_shots) + len(
                                              self.eval_shots)]).to(
                                              args.device)))

                # graph classification
                tg_labels.append(
                    self.t_graph_labels[t_test_idx + len(self.train_shots) + len(self.eval_shots)].cpu().numpy())
                tg_preds.append(
                    self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid().cpu().numpy())

                # update the models
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
        criterion = torch.nn.BCELoss()

        # load the TG-models
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification models.")

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

        for epoch in range(1, args.max_epoch + 1):
            # for epoch in range(1, 5):
            self.model.train()
            self.model.init_hiddens()  # Just added
            self.tgc_decoder.train()
            t_epoch_start = timeit.default_timer()
            epoch_losses = []
            tg_labels = []
            tg_preds = []
            for t_train_idx, t_train in enumerate(self.train_shots):
                optimizer.zero_grad()

                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(data, t_train)

                embeddings = self.model(edge_index, self.x)

                # graph readout
                tg_readout = readout_function(embeddings, self.readout_scheme)
                tg_embedding = torch.cat((tg_readout, torch.from_numpy(self.t_graph_feat[t_train_idx]).to(args.device)))

                # graph classification
                tg_label = self.t_graph_labels[t_train_idx].float().view(1, )
                tg_pred = self.tgc_decoder(tg_embedding.view(1, tg_embedding.size()[0]).float()).sigmoid()

                tg_labels.append(tg_label.cpu().numpy())
                tg_preds.append(tg_pred.cpu().detach().numpy())
                t_loss = criterion(tg_pred, tg_label)
                t_loss.backward()
                optimizer.step()
                epoch_losses.append(t_loss.item())
                # update the models
                self.model.update_hiddens_all_with(embeddings)

            avg_epoch_loss = np.mean(epoch_losses)
            train_avg_epoch_loss_dict[epoch] = avg_epoch_loss
            train_auc, train_ap = roc_auc_score(tg_labels, tg_preds), average_precision_score(tg_labels, tg_preds)
            eval_epoch, eval_auc, eval_ap = self.tgclassification_eval(epoch, self.readout_scheme)

            # Only apply early stopping when after min_epoch of training
            if best_eval_auc < eval_auc:  # Use AUC as metric to define early stoping
                patience = 0
                best_model = self.model.state_dict()  # Saved the best model for testing
                best_MLP = self.tgc_decoder.state_dict()

                best_eval_auc = eval_auc
                best_epoch, best_test_auc, best_test_ap = self.tgclassification_test(epoch, self.readout_scheme)
            else:
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
        partial_results_path = f'../data/output/log/single_model_loss/{args.dataset}/{args.model}/'
        loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        if os.path.exists(partial_results_path) == False:
            os.makedirs(partial_results_path)

        with open(loss_log_filename, 'wb') as file:
            dump(train_avg_epoch_loss_dict, file)

        # Final Test
        logger.info("Best Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(best_epoch, best_test_auc, best_test_ap))
        save_results(args.dataset, best_test_auc, best_test_ap, self.tgc_lr, len(self.train_shots),
                     len(self.test_shots), best_epoch, total_time)


if __name__ == '__main__':
    from script.utils.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.utils.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.utils.inits import prepare

    #This array can be replaced by a list of datasets readed from a specific file
    datasets = [
        "ARC"
    ]

    seeds = [710, 720, 800]

    args.max_epoch = 250
    args.wandb = False #Set this to true if you want to use wandb as a training debug tool
    args.min_epoch = 100
    args.model = "HTGN"
    args.log_interval = 10
    args.lr = 0.00015
    args.patience = 20

    for dataset in datasets:
        for seed in seeds:
            args.dataset = dataset
            args.seed = seed

            print("INFO: >>> Temporal Graph Classification <<<")
            print("INFO: Args: ", args)
            print("======================================")
            print("INFO: Dataset: {}".format(args.dataset))
            print("INFO: Model: {}".format(args.model))

            data = loader(dataset=args.dataset, neg_sample=args.neg_sample)
            args.num_nodes = data['num_nodes']
            print("INFO: Number of nodes:", args.num_nodes)
            set_random(args.seed)

            args.output_folder = '../data/output/log/console_log/{}/{}/'.format(args.dataset, args.model)
            init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
            runner = Runner()
            runner.run()
            wandb.finish()
