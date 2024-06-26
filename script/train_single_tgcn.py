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




def save_results(dataset, test_auc, test_ap,lr,train_snapshot,test_snapshot,best_epoch,time):
    partial_path = "../data/output/single_model_{}/".format(args.model)
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
    partial_path = "../data/output/epoch_result/single_model_{}/".format(args.model)
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


class Runner():
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="gclstm_i1",
                # Set name of the run:
                name="{}_{}_{}".format(args.dataset, args.model, args.seed),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,

                }
            )

        self.model_path = '../saved_models/single_model/{}_{}_seed_{}/'.format(args.dataset,
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

        self.nfeat = args.nfeat   # @TODO: Replace with args to config it easily
        self.node_feat = torch.randn((self.num_nodes, self.nfeat)).to(args.device)



        # self.node_feat = torch.eye(5351).to(args.device)
        # self.nfeat = self.node_feat.size(1)  # @TODO: Replace with args to config it easily
        self.edge_feat_dim = 1 #@TODO: Replace with args to config it easily
        self.hidden_dim = args.nhid

        self.model = load_model(args).to(args.device)


        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_dim=self.hidden_dim + num_extra_feat, hidden_dim_1=self.hidden_dim + num_extra_feat,
                          hidden_dim_2=self.hidden_dim + num_extra_feat,
                          drop=0.1).to(args.device)  # @NOTE: these hyperparameters may need to be changed

        #Hidden for gclstm
        self.h = None

    # Hidden for gclstm
    def init_hidden(self):
        self.h = None

    # Detach gradient
    def detach_hidden(self):
        self.h =  self.h.detach()

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

                self.h = self.model(self.node_feat, edge_idx, edge_att, self.h)
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

                self.h = self.model(self.node_feat, edge_idx, edge_att, self.h)


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
                self.h = self.model(self.node_feat, edge_idx, edge_att, self.h)
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
        partial_results_path = f'../../data/output/log/single_model/{args.dataset}/{args.model}/'
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
    from script.utils.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.utils.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir, extra_dataset_attributes_loading, \
        data_loader_geometric_temporal
    from script.utils.inits import prepare,prepare_TGS_for_TGC

    #This array can be replaced by a list of datasets readed from a specific file
    datasets = [
        "unnamedtoken18980x00a8b738e453ffd858a7edf03bccfe20412f0eb0"
    ]

    seeds = [710]

    args.max_epoch = 5
    args.wandb = False #Set this to true if you want to use wandb as a training debug tool
    args.min_epoch = 100
    args.model = "TGCN"
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

            data = data_loader_geometric_temporal(dataset=args.dataset)
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
